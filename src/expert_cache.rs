pub type ModuleFactoryResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Debug, Clone)]
pub struct ExpertCacheEntry<T: candle_nn::Module> {
    module: Option<T>,
    accesses: usize,
    last_accessed: usize,
    evictable: bool,
}

pub struct ExpertCache<T: candle_nn::Module> {
    entries: Vec<ExpertCacheEntry<T>>,
    occupancy_limit: usize,
    occupancy_current: usize,
    module_factory: Box<dyn Fn(usize, usize) -> ModuleFactoryResult<T>>,
    n_experts_per_layer: usize,
    n_layers: usize,
    /// The current layer of the forward pass. Used by the eviction policy to determine which expert(s) to evict.
    evictor_current_layer: usize,
}

impl<T: candle_nn::Module + std::fmt::Debug> std::fmt::Debug for ExpertCache<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertCache")
            .field("entries", &self.entries)
            .field("occupancy_limit", &self.occupancy_limit)
            .field("occupancy_current", &self.occupancy_current)
            .field("n_experts_per_layer", &self.n_experts_per_layer)
            .field("n_layers", &self.n_layers)
            .field("evictor_current_layer", &self.evictor_current_layer)
            .finish()
    }
}

/// Modulo operation that works for a < 0. Example:
/// neg_mod(-1, 3) = 2.
/// neg_mod(0, 3) = 0.
/// neg_mod(1, 3) = 1.
/// neg_mod(2, 3) = 2.
/// neg_mod(3, 3) = 0.
fn neg_mod(a: i64, b: i64) -> i64 {
    if a < 0 { b - ((-a) % b) } else { a % b }
}

impl<T: candle_nn::Module> ExpertCache<T> {
    pub fn new(
        n_layers: usize,
        n_experts_per_layer: usize,
        module_factory: Box<dyn Fn(usize, usize) -> ModuleFactoryResult<T>>,
        occupancy_limit: usize,
    ) -> Self {
        let mut entries = Vec::with_capacity(n_layers * n_experts_per_layer);

        for _ in 0..n_layers {
            for _ in 0..n_experts_per_layer {
                entries.push(ExpertCacheEntry {
                    module: None,
                    accesses: 0,
                    last_accessed: 0,

                    evictable: true,
                });
            }
        }

        Self {
            entries,
            occupancy_limit,
            occupancy_current: 0,
            module_factory,
            n_experts_per_layer,
            n_layers,
            evictor_current_layer: 0,
        }
    }

    pub fn set_current_layer(&mut self, layer: usize) {
        self.evictor_current_layer = layer;
    }

    /// super naive eviction policy: evict the first evictable expert in the previous layer.
    fn get_eviction_candidate(&mut self) -> Result<(usize, usize), ModuleCacheError> {
        let mut curr_layer =
            neg_mod(self.evictor_current_layer as i64 - 1, self.n_layers as i64) as usize;

        while curr_layer != self.evictor_current_layer {
            let entry = self.get_entry(curr_layer, 0)?;
            if entry.evictable {
                return Ok((curr_layer, 0));
            }
            curr_layer = neg_mod(curr_layer as i64 - 1, self.n_layers as i64) as usize;
        }

        Err(ModuleCacheError::NoEvictionCandidate)
    }

    fn evict(&mut self, idx_layer: usize, idx_expert: usize) -> Result<(), ModuleCacheError> {
        let entry = self.get_entry_mut(idx_layer, idx_expert)?;
        entry.module = None;
        self.occupancy_current -= 1;
        println!("Evicted layer {} expert {}", idx_layer, idx_expert);
        Ok(())
    }

    fn ensure_occupancy(&mut self) -> Result<(), ModuleCacheError> {
        if self.occupancy_current > self.occupancy_limit {
            let (idx_layer, idx_expert) = self.get_eviction_candidate()?;
            self.evict(idx_layer, idx_expert)?;
            Ok(())
        } else {
            Ok(())
        }
    }

    fn get_entry(
        &self,
        idx_layer: usize,
        idx_expert: usize,
    ) -> Result<&ExpertCacheEntry<T>, ModuleCacheError> {
        Ok(self
            .entries
            .get(idx_layer * self.n_experts_per_layer + idx_expert)
            .ok_or(ModuleCacheError::ModuleNotFound(format!(
                "Layer {} expert {} not found",
                idx_layer, idx_expert
            )))?)
    }

    fn get_entry_mut(
        &mut self,
        idx_layer: usize,
        idx_expert: usize,
    ) -> Result<&mut ExpertCacheEntry<T>, ModuleCacheError> {
        Ok(self
            .entries
            .get_mut(idx_layer * self.n_experts_per_layer + idx_expert)
            .ok_or(ModuleCacheError::ModuleNotFound(format!(
                "Layer {} expert {} not found",
                idx_layer, idx_expert
            )))?)
    }

    pub fn get(&mut self, idx_layer: usize, idx_expert: usize) -> Result<&T, ModuleCacheError> {
        let entry = self.get_entry_mut(idx_layer, idx_expert)?;

        if !entry.module.is_some() {
            self.get_entry_mut(idx_layer, idx_expert)?.module =
                Some((self.module_factory)(idx_layer, idx_expert)?);
            self.ensure_occupancy();
            self.occupancy_current += 1;
        }

        Ok(self
            .get_entry_mut(idx_layer, idx_expert)?
            .module
            .as_ref()
            .unwrap())
    }

    pub fn set(
        &mut self,
        idx_layer: usize,
        idx_expert: usize,
        module: T,
    ) -> Result<(), ModuleCacheError> {
        let entry = self.get_entry_mut(idx_layer, idx_expert)?;

        if !entry.module.is_some() {
            self.ensure_occupancy();
            self.occupancy_current += 1;
        }

        self.get_entry_mut(idx_layer, idx_expert)?.module = Some(module);
        Ok(())
    }

    pub fn set_evictable(
        &mut self,
        idx_layer: usize,
        idx_expert: usize,
        evictable: bool,
    ) -> Result<(), ModuleCacheError> {
        let entry = self.get_entry_mut(idx_layer, idx_expert)?;
        entry.evictable = evictable;
        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ModuleCacheError {
    #[error("Module not found: {0}")]
    ModuleNotFound(String),
    #[error("Module cache full")]
    ModuleCacheFull,
    #[error("No eviction candidate")]
    NoEvictionCandidate,
    #[error("Module factory error: {0}")]
    ModuleFactoryError(Box<dyn std::error::Error + Send + Sync>),
}


impl From<Box<dyn std::error::Error + Send + Sync>> for ModuleCacheError {
    fn from(error: Box<dyn std::error::Error + Send + Sync>) -> Self {
        ModuleCacheError::ModuleFactoryError(error)
    }
}

impl From<ModuleCacheError> for candle_core::Error {
    fn from(error: ModuleCacheError) -> Self {
        candle_core::Error::Wrapped(Box::new(error))
    }
}