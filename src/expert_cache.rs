use std::collections::{HashMap, HashSet};

use candle_core::{Device, quantized::QTensor};

use crate::{
    pread_loader::PreadTensorLoader,
    prefetcher::{Prefetcher, Primitive},
};

pub type ExpertFactoryResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Debug, Clone)]
pub struct ExpertCacheEntry<T: candle_nn::Module> {
    state: ExpertCacheEntryState<T>,
    accesses: usize,
    last_accessed: usize,
    evictable: bool,
}

#[derive(Debug, Clone)]
pub enum ExpertCacheEntryState<T: candle_nn::Module> {
    Empty,
    Pending,
    Loaded(T),
}

pub struct ExpertPackage {
    pub tensors: HashMap<String, QTensor>,
    pub expert: (usize, usize),
}

impl Primitive for (usize, usize) {}

pub struct ExpertCache<T: candle_nn::Module> {
    entries: Vec<ExpertCacheEntry<T>>,
    occupancy_limit: usize,
    occupancy_current: usize,
    expert_spec: Box<dyn Fn(usize, usize) -> Vec<String>>,
    expert_factory: Box<dyn Fn(ExpertPackage) -> ExpertFactoryResult<T>>,
    n_experts_per_layer: usize,
    n_layers: usize,
    /// The current layer of the forward pass. Used by the eviction policy to determine which expert(s) to evict.
    evictor_current_layer: usize,
    prefetcher: Prefetcher<(usize, usize)>,
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

pub struct ExpertCacheConfig<T: candle_nn::Module> {
    /// The number of layers in the model.
    pub n_layers: usize,
    /// The number of experts per layer.
    pub n_experts_per_layer: usize,
    /// The factory function to create an expert from a set of tensors.
    pub expert_factory: Box<dyn Fn(ExpertPackage) -> ExpertFactoryResult<T>>,
    /// Maps an expert index to a list of tensor names for that expert.
    pub expert_spec: Box<dyn Fn(usize, usize) -> Vec<String>>,
    /// The occupancy limit of the cache.
    pub occupancy_limit: usize,
    /// The device to load the tensors on.
    pub device: Device,
    /// The model loader to load the tensors from.
    pub loader: PreadTensorLoader,
}

impl<T: candle_nn::Module> ExpertCache<T> {
    pub fn new(cfg: ExpertCacheConfig<T>) -> Self {
        let mut entries = Vec::with_capacity(cfg.n_layers * cfg.n_experts_per_layer);

        for _ in 0..cfg.n_layers {
            for _ in 0..cfg.n_experts_per_layer {
                entries.push(ExpertCacheEntry {
                    state: ExpertCacheEntryState::Empty,
                    accesses: 0,
                    last_accessed: 0,

                    evictable: true,
                });
            }
        }

        Self {
            entries,
            occupancy_limit: cfg.occupancy_limit,
            occupancy_current: 0,
            expert_factory: cfg.expert_factory,
            expert_spec: cfg.expert_spec,
            prefetcher: Prefetcher::new(cfg.loader, cfg.device),
            evictor_current_layer: 0,
            n_experts_per_layer: cfg.n_experts_per_layer,
            n_layers: cfg.n_layers,
        }
    }

    /// Sets the current prediction layer. This is used by the eviction policy to determine which expert(s) to evict.
    pub fn set_current_layer(&mut self, layer: usize) {
        self.evictor_current_layer = layer;
    }

    /// super naive eviction policy: evict the first evictable expert in the previous layer.
    fn get_eviction_candidate(&mut self) -> Result<(usize, usize), ExpertCacheError> {
        let mut curr_layer =
            neg_mod(self.evictor_current_layer as i64 - 1, self.n_layers as i64) as usize;

        while curr_layer != self.evictor_current_layer {
            for expert in 0..self.n_experts_per_layer {
                let entry = self.get_entry(curr_layer, expert)?;
                if entry.evictable && matches!(entry.state, ExpertCacheEntryState::Loaded(_)) {
                    return Ok((curr_layer, expert));
                }
            }
            curr_layer = neg_mod(curr_layer as i64 - 1, self.n_layers as i64) as usize;
        }

        Err(ExpertCacheError::NoEvictionCandidate)
    }

    /// Evicts an entry and decrements the occupancy count.
    fn evict(&mut self, idx_layer: usize, idx_expert: usize) -> Result<(), ExpertCacheError> {
        let entry = self.get_entry_mut(idx_layer, idx_expert)?;
        entry.state = ExpertCacheEntryState::Empty;
        self.occupancy_current -= 1;
        println!("Evicted layer {} expert {}", idx_layer, idx_expert);
        Ok(())
    }

    /// Ensures that there is enough occupancy for one more entry.
    fn ensure_occupancy(&mut self) -> Result<(), ExpertCacheError> {
        while self.occupancy_current >= self.occupancy_limit {
            let (idx_layer, idx_expert) = self.get_eviction_candidate()?;
            self.evict(idx_layer, idx_expert)?;
        }

        Ok(())
    }

    /// Gets an entry by layer and expert index.
    fn get_entry(
        &self,
        idx_layer: usize,
        idx_expert: usize,
    ) -> Result<&ExpertCacheEntry<T>, ExpertCacheError> {
        Ok(self
            .entries
            .get(idx_layer * self.n_experts_per_layer + idx_expert)
            .ok_or(ExpertCacheError::ModuleNotFound(format!(
                "Layer {} expert {} not found",
                idx_layer, idx_expert
            )))?)
    }

    /// Gets a mutable entry by layer and expert index.
    fn get_entry_mut(
        &mut self,
        idx_layer: usize,
        idx_expert: usize,
    ) -> Result<&mut ExpertCacheEntry<T>, ExpertCacheError> {
        Ok(self
            .entries
            .get_mut(idx_layer * self.n_experts_per_layer + idx_expert)
            .ok_or(ExpertCacheError::ModuleNotFound(format!(
                "Layer {} expert {} not found",
                idx_layer, idx_expert
            )))?)
    }

    /// Request an expert be fetched asynchronously.
    pub fn prefetch(
        &mut self,
        idx_layer: usize,
        idx_expert: usize,
    ) -> Result<(), ExpertCacheError> {
        let entry = self.get_entry(idx_layer, idx_expert)?;

        if matches!(entry.state, ExpertCacheEntryState::Empty) {
            self.ensure_occupancy()?;
            self.get_entry_mut(idx_layer, idx_expert)?.state = ExpertCacheEntryState::Pending;
            self.occupancy_current += 1;
            println!("Prefetching layer {} expert {}", idx_layer, idx_expert);
            let tensors = (self.expert_spec)(idx_layer, idx_expert);
            self.prefetcher.enqueue((idx_layer, idx_expert), tensors);
        }

        Ok(())
    }

    /// Drains all prefetch results back into the cache. Non-blocking. Returns the set of experts that were drained.
    pub fn drain(&mut self) -> Result<Vec<(usize, usize)>, ExpertCacheError> {
        let mut drained_experts = Vec::new();
        while let Some(response) = self.prefetcher.try_recv() {
            let ((idx_layer, idx_expert), module) = self.collect_prefetch_response(response)?;
            self.set(idx_layer, idx_expert, module)?;
            drained_experts.push((idx_layer, idx_expert));
        }
        Ok(drained_experts)
    }

    fn collect_prefetch_response(
        &self,
        response: crate::prefetcher::PrefetchResponse<(usize, usize)>,
    ) -> Result<((usize, usize), T), ExpertCacheError> {
        let (idx_layer, idx_expert) = response.id;
        let tensors = response
            .tensors
            .map_err(|error| ExpertCacheError::FetchingError(error))?;
        let spec = (self.expert_spec)(idx_layer, idx_expert);
        let spec_len = spec.len();

        // map order of tensors back to the order of the spec
        let collected_tensors: HashMap<String, QTensor> = spec.into_iter().zip(tensors).collect();

        assert!(
            collected_tensors.len() == spec_len,
            "Mismatch between number of tensors and number of spec entries"
        );

        let module = (self.expert_factory)(ExpertPackage {
            tensors: collected_tensors,
            expert: (idx_layer, idx_expert),
        })
        .map_err(|error| ExpertCacheError::ExpertFactoryError(error))?;

        Ok(((idx_layer, idx_expert), module))
    }

    fn drain_blocking_one(&mut self) -> Result<(usize, usize), ExpertCacheError> {
        let response = self.prefetcher.recv_blocking();
        let ((idx_layer, idx_expert), module) = self.collect_prefetch_response(response)?;
        self.set(idx_layer, idx_expert, module)?;
        Ok((idx_layer, idx_expert))
    }

    /// Blocks until all specified experts are `Loaded`. Requests will be sent to load experts if they are not already loaded and are not already in the process of being loaded.
    /// Will error with `TooManyBlockingPrefetchRequests` if the number of desired experts is greater than the cache limit, as this would cause a deadlock.
    /// After this call, all requested experts are guaranteed to be `Loaded`.
    pub fn wait_for_experts(&mut self, experts: &[(usize, usize)]) -> Result<(), ExpertCacheError> {
        if experts.len() > self.occupancy_limit {
            return Err(ExpertCacheError::TooManyBlockingPrefetchRequests);
        }

        let mut pending_experts = HashSet::new();

        for (idx_layer, idx_expert) in experts {
            let entry = self.get_entry_mut(*idx_layer, *idx_expert)?;

            match entry.state {
                ExpertCacheEntryState::Empty => {
                    self.prefetch(*idx_layer, *idx_expert)?;
                    pending_experts.insert((*idx_layer, *idx_expert));
                }
                ExpertCacheEntryState::Pending => {
                    pending_experts.insert((*idx_layer, *idx_expert));
                }
                ExpertCacheEntryState::Loaded(_) => {
                    // already loaded
                }
            }
        }

        // invariant: all requested entries are either Pending or Loaded

        while !pending_experts.is_empty() {
            let (idx_layer, idx_expert) = self.drain_blocking_one()?;
            pending_experts.remove(&(idx_layer, idx_expert));
        }

        // invariant: all requested entries are Loaded

        Ok(())
    }

    /// Gets an expert by layer and expert index. If the expert is not loaded, a request will be sent to load it, the thread will block until the expert is loaded.
    pub fn get_blocking(
        &mut self,
        idx_layer: usize,
        idx_expert: usize,
    ) -> Result<&T, ExpertCacheError> {
        self.wait_for_experts(&[(idx_layer, idx_expert)])?;

        if let ExpertCacheEntryState::Loaded(module) =
            &self.get_entry_mut(idx_layer, idx_expert)?.state
        {
            Ok(module)
        } else {
            unreachable!()
        }
    }

    /// Sets the entry to Loaded and increments the occupancy count. Evicts if necessary.
    fn set(
        &mut self,
        idx_layer: usize,
        idx_expert: usize,
        module: T,
    ) -> Result<(), ExpertCacheError> {
        let entry = self.get_entry_mut(idx_layer, idx_expert)?;

        if matches!(entry.state, ExpertCacheEntryState::Empty) {
            self.ensure_occupancy()?;
            self.occupancy_current += 1;
        }

        self.get_entry_mut(idx_layer, idx_expert)?.state = ExpertCacheEntryState::Loaded(module);
        Ok(())
    }

    /// Configure the evictability of a given expert.
    pub fn set_evictable(
        &mut self,
        idx_layer: usize,
        idx_expert: usize,
        evictable: bool,
    ) -> Result<(), ExpertCacheError> {
        let entry = self.get_entry_mut(idx_layer, idx_expert)?;
        entry.evictable = evictable;
        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ExpertCacheError {
    #[error("Module not found: {0}")]
    ModuleNotFound(String),

    #[error("No eviction candidate")]
    NoEvictionCandidate,

    #[error("Too many blocking prefetch requests")]
    TooManyBlockingPrefetchRequests,

    #[error("Tensor fetching error: {0}")]
    FetchingError(candle_core::Error),

    #[error("Expert factory error: {0}")]
    ExpertFactoryError(Box<dyn std::error::Error + Send + Sync>),
}

impl From<ExpertCacheError> for candle_core::Error {
    fn from(error: ExpertCacheError) -> Self {
        candle_core::Error::Wrapped(Box::new(error))
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for ExpertCacheError {
    fn from(error: Box<dyn std::error::Error + Send + Sync>) -> Self {
        ExpertCacheError::ExpertFactoryError(error)
    }
}
