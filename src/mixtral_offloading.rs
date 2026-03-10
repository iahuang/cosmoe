//! Mixtral Model with GGUF quantized weights and dynamic expert offloading.
//!
//! Uses QMatMul for all linear layers (quantized matmul with on-the-fly
//! dequantization in Metal kernels). Expert weights are loaded on demand
//! from a GGUF file via pread().

use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::Activation;
use std::collections::HashMap;
use std::sync::Arc;

use crate::expert_cache::{ExpertCache, ExpertCacheConfig, ExpertFactoryResult};
use crate::pread_loader::PreadTensorLoader;

/// Mixtral configuration. Same as before but read from GGUF metadata at runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: usize,
    pub num_experts_per_tok: usize,
    pub num_local_experts: usize,
    pub use_flash_attn: bool,
}

impl Config {
    /// https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
    pub fn v0_1_8x7b(use_flash_attn: bool) -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            hidden_act: Activation::Silu,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: 1e6,
            sliding_window: 4096,
            num_experts_per_tok: 2,
            num_local_experts: 8,
            use_flash_attn,
        }
    }
}

// ── Quantized Linear wrapper ──────────────────────────────────────────────

/// A linear layer backed by QMatMul (quantized weights, no bias).
#[derive(Debug, Clone)]
struct Linear {
    weight: QMatMul,
}

impl Linear {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        Ok(Self {
            weight: QMatMul::from_qtensor(qtensor)?,
        })
    }

    fn from_arc(qtensor: Arc<QTensor>) -> Result<Self> {
        Ok(Self {
            weight: QMatMul::from_arc(qtensor)?,
        })
    }
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.weight)
    }
}

// ── RmsNorm (dequantized at load time) ────────────────────────────────────

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn from_qtensor(qtensor: &QTensor, eps: f64) -> Result<Self> {
        let weight = qtensor.dequantize(&qtensor.device())?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}

// ── Rotary Embeddings ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

impl RotaryEmbedding {
    fn new(cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (cfg.rope_theta as f32).powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin))?;
        let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin))?;
        Ok((q_embed, k_embed))
    }
}

// ── Attention ─────────────────────────────────────────────────────────────

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
    use_flash_attn: bool,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        tensors: &HashMap<String, Arc<QTensor>>,
        prefix: &str,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;

        let q_proj = Linear::from_arc(tensors[&format!("{prefix}.attn_q.weight")].clone())?;
        let k_proj = Linear::from_arc(tensors[&format!("{prefix}.attn_k.weight")].clone())?;
        let v_proj = Linear::from_arc(tensors[&format!("{prefix}.attn_v.weight")].clone())?;
        let o_proj = Linear::from_arc(tensors[&format!("{prefix}.attn_output.weight")].clone())?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            kv_cache: None,
            use_flash_attn: cfg.use_flash_attn,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = candle_transformers::utils::repeat_kv(key_states, self.num_kv_groups)?;
        let value_states = candle_transformers::utils::repeat_kv(value_states, self.num_kv_groups)?;

        let attn_output = if self.use_flash_attn {
            let q = query_states.transpose(1, 2)?;
            let k = key_states.transpose(1, 2)?;
            let v = value_states.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, q_len > 1)?.transpose(1, 2)?
        } else {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&value_states)?
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

// ── Expert MLP (quantized) ────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct BlockSparseTop2MLP {
    w1: Linear,
    w2: Linear,
    w3: Linear,
    act_fn: Activation,
}

impl BlockSparseTop2MLP {
    /// Construct an expert from 3 QTensors loaded from the GGUF file.
    fn from_qtensors(w1: QTensor, w2: QTensor, w3: QTensor, act_fn: Activation) -> Result<Self> {
        Ok(Self {
            w1: Linear::from_qtensor(w1)?,
            w2: Linear::from_qtensor(w2)?,
            w3: Linear::from_qtensor(w3)?,
            act_fn,
        })
    }
}

impl Module for BlockSparseTop2MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.w1)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.w3)?;
        (lhs * rhs)?.apply(&self.w2)
    }
}

// ── Sparse MoE Block ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SparseMoeBlock {
    gate: Linear,
    num_experts_per_tok: usize,
    num_local_experts: usize,
    this_layer: usize,
}

impl SparseMoeBlock {
    fn new(cfg: &Config, this_layer: usize, gate_tensor: Arc<QTensor>) -> Result<Self> {
        let gate = Linear::from_arc(gate_tensor)?;
        Ok(SparseMoeBlock {
            gate,
            num_local_experts: cfg.num_local_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            this_layer,
        })
    }
}

impl SparseMoeBlock {
    fn forward(
        &self,
        xs: &Tensor,
        expert_cache: &mut ExpertCache<BlockSparseTop2MLP>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let router_logits = xs.apply(&self.gate)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

        let mut top_x = vec![vec![]; self.num_local_experts];
        let mut selected_rws = vec![vec![]; self.num_local_experts];

        for (row_idx, rw) in routing_weights.iter().enumerate() {
            let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
            dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
            let mut sum_routing_weights = 0f32;
            for &expert_idx in dst.iter().take(self.num_experts_per_tok) {
                let expert_idx = expert_idx as usize;
                let routing_weight = rw[expert_idx];
                sum_routing_weights += routing_weight;
                top_x[expert_idx].push(row_idx as u32);
            }
            for &expert_idx in dst.iter().take(self.num_experts_per_tok) {
                let expert_idx = expert_idx as usize;
                let routing_weight = rw[expert_idx];
                selected_rws[expert_idx].push(routing_weight / sum_routing_weights)
            }
        }

        let mut ys = xs.zeros_like()?;
        for expert_idx in 0..self.num_local_experts {
            let top_x = &top_x[expert_idx];
            if top_x.is_empty() {
                continue;
            }
            let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
            let selected_rws =
                Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?.reshape(((), 1))?;
            let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
            let expert_layer = expert_cache.get_blocking(self.this_layer, expert_idx)?;
            let current_hidden_states = expert_layer.forward(&current_state)?;
            let current_hidden_states = current_hidden_states.broadcast_mul(&selected_rws)?;
            ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
        }

        let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
        Ok(ys)
    }
}

// ── Decoder Layer ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    block_sparse_moe: SparseMoeBlock,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        tensors: &HashMap<String, Arc<QTensor>>,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let self_attn = Attention::new(rotary_emb, cfg, tensors, &prefix)?;

        let gate_tensor = tensors[&format!("{prefix}.ffn_gate_inp.weight")].clone();
        let block_sparse_moe = SparseMoeBlock::new(cfg, layer_idx, gate_tensor)?;

        let input_layernorm = RmsNorm::from_qtensor(
            &tensors[&format!("{prefix}.attn_norm.weight")],
            cfg.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::from_qtensor(
            &tensors[&format!("{prefix}.ffn_norm.weight")],
            cfg.rms_norm_eps,
        )?;

        Ok(Self {
            self_attn,
            block_sparse_moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        expert_cache: &mut ExpertCache<BlockSparseTop2MLP>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = self.block_sparse_moe.forward(&xs, expert_cache)?;
        residual + xs
    }
}

// ── Model ─────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    sliding_window: usize,
    device: Device,
    expert_cache: ExpertCache<BlockSparseTop2MLP>,
}

type ExpertFactory<T> = dyn Fn(usize, usize) -> ExpertFactoryResult<T> + Send + Sync + 'static;

impl Model {
    pub fn new(
        cfg: &Config,
        expert_cache_occupancy_limit: usize,
        loader: PreadTensorLoader,
        device: &Device,
    ) -> Result<Self> {
        // Load non-expert tensors explicitly.
        let t = |name: &str| -> Result<Arc<QTensor>> {
            Ok(Arc::new(loader.load_qtensor(name, device)?))
        };

        // Embedding: dequantize to F32 at load time
        let embed_weight_qt = t("token_embd.weight")?;
        let embed_weight = embed_weight_qt.dequantize(device)?;
        let embed_tokens = candle_nn::Embedding::new(embed_weight, cfg.hidden_size);

        let rotary_emb = Arc::new(RotaryEmbedding::new(cfg, device)?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let prefix = format!("blk.{layer_idx}");
            let tensors: HashMap<String, Arc<QTensor>> = [
                format!("{prefix}.attn_q.weight"),
                format!("{prefix}.attn_k.weight"),
                format!("{prefix}.attn_v.weight"),
                format!("{prefix}.attn_output.weight"),
                format!("{prefix}.attn_norm.weight"),
                format!("{prefix}.ffn_norm.weight"),
                format!("{prefix}.ffn_gate_inp.weight"),
            ]
            .into_iter()
            .map(|name| Ok((name.clone(), t(&name)?)))
            .collect::<Result<_>>()?;

            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, &tensors, layer_idx)?;
            layers.push(layer);
        }

        let output_norm_qt = t("output_norm.weight")?;
        let norm = RmsNorm::from_qtensor(&output_norm_qt, cfg.rms_norm_eps)?;

        // lm_head: may share weights with token_embd in some models
        let lm_head_tensor = loader
            .load_qtensor("output.weight", device)
            .map(Arc::new)
            .unwrap_or(embed_weight_qt);
        let lm_head = Linear::from_arc(lm_head_tensor)?;

        let cfg_f = cfg.clone();
        let device_f = device.clone();
        let load_count = std::sync::atomic::AtomicUsize::new(0);

        let expert_cache = ExpertCache::new(ExpertCacheConfig {
            n_layers: cfg.num_hidden_layers,
            n_experts_per_layer: cfg.num_local_experts,
            expert_factory: Box::new(move |mut expert_package| {
                let (layer, expert) = expert_package.expert;
                let w1 = expert_package
                    .tensors
                    .remove(&format!("blk.{layer}.ffn_gate.{expert}.weight"))
                    .unwrap();
                let w2 = expert_package
                    .tensors
                    .remove(&format!("blk.{layer}.ffn_down.{expert}.weight"))
                    .unwrap();
                let w3 = expert_package
                    .tensors
                    .remove(&format!("blk.{layer}.ffn_up.{expert}.weight"))
                    .unwrap();
                Ok(BlockSparseTop2MLP::from_qtensors(
                    w1,
                    w2,
                    w3,
                    cfg_f.hidden_act,
                )?)
            }),
            expert_spec: Box::new(|layer, expert| {
                vec![
                    format!("blk.{layer}.ffn_gate.{expert}.weight"),
                    format!("blk.{layer}.ffn_down.{expert}.weight"),
                    format!("blk.{layer}.ffn_up.{expert}.weight"),
                ]
            }),
            occupancy_limit: expert_cache_occupancy_limit,
            device: device.clone(),
            loader: loader,
        });

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: cfg.sliding_window,
            device: device.clone(),
            expert_cache,
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + self.sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(DType::F32)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        println!(
            "Forwarding {:?} ids with seqlen_offset: {}",
            input_ids.shape(),
            seqlen_offset
        );
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset)?;
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            self.expert_cache.set_current_layer(layer_idx);
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                &mut self.expert_cache,
            )?
        }
        xs.narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }
}
