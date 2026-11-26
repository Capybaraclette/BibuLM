use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use thiserror::Error;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const MAX_STRING_LEN: usize = 16 * 1024 * 1024; // 16 MiB safety cap per string
const MAX_ARRAY_SKIP: u64 = 50_000_000;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("I/O error reading GGUF: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid GGUF magic header")]
    InvalidMagic,
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    #[error("Unsupported GGUF value type: {0}")]
    UnsupportedValueType(u32),
    #[error("Corrupt or oversized GGUF field: {0}")]
    Malformed(String),
}

/// GGML tensor quantization / data type IDs from ggml.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    I8,
    I16,
    I32,
    I64,
    F64,
    Iq1M,
    Bf16,
    Tq1_0,
    Tq2_0,
    Mxfp4,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::Iq1M,
            30 => Self::Bf16,
            34 => Self::Tq1_0,
            35 => Self::Tq2_0,
            39 => Self::Mxfp4,
            other => Self::Unknown(other),
        }
    }

    /// Returns `(block_size_elements, type_size_bytes)` per GGML tensor layout.
    pub fn block_layout(self) -> (u64, u64) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::Bf16 | Self::I16 => (1, 2),
            Self::F64 | Self::I64 => (1, 8),
            Self::I8 => (1, 1),
            Self::I32 => (1, 4),
            Self::Q4_0 | Self::Iq4Nl => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 292),
            Self::Iq2Xxs => (256, 66),
            Self::Iq2Xs => (256, 74),
            Self::Iq3Xxs => (256, 98),
            Self::Iq1S => (256, 50),
            Self::Iq3S => (256, 110),
            Self::Iq2S => (256, 82),
            Self::Iq4Xs => (256, 136),
            Self::Iq1M => (256, 56),
            Self::Tq1_0 => (256, 54),
            Self::Tq2_0 => (256, 66),
            Self::Mxfp4 => (32, 17),
            Self::Unknown(_) => (32, 18), // conservative ~4.5 bpw fallback
        }
    }

    /// Computes exact byte size of a tensor with `n_elements` elements.
    pub fn tensor_bytes(self, n_elements: u64) -> u64 {
        let (blck, sz) = self.block_layout();
        if n_elements == 0 {
            return 0;
        }
        let blocks = n_elements.div_ceil(blck);
        blocks.saturating_mul(sz)
    }
}

/// Exact per-category byte breakdown computed directly from GGUF tensor entries.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TensorMemorySummary {
    /// Embeddings, output head, final norm (always kept hot on VRAM when possible).
    pub non_layer_hot_bytes: u64,
    /// Per-layer dense/core weights: attention + norms + dense FFN + shared experts + router gates (layer index -> bytes).
    pub per_layer_hot_bytes: BTreeMap<u32, u64>,
    /// Per-layer routed MoE expert tensors (`blk.X.ffn_*_exps*`) (layer index -> bytes).
    pub per_layer_routed_expert_bytes: BTreeMap<u32, u64>,
    /// Per-layer Multi-Token Prediction (MTP / NextN) head blocks (layer index -> bytes).
    pub per_layer_mtp_bytes: BTreeMap<u32, u64>,
    /// Standalone non-block MTP tensors (e.g., `mtp.*` / `nextn.*` without `blk.X` prefix).
    pub non_layer_mtp_bytes: u64,
    /// Total tensor bytes across all parsed shards.
    pub total_tensor_bytes: u64,
    /// Total parameter element count summed across tensors.
    pub total_parameters: u64,
}

impl TensorMemorySummary {
    pub fn total_hot_bytes(&self) -> u64 {
        self.non_layer_hot_bytes
            + self.per_layer_hot_bytes.values().copied().sum::<u64>()
    }

    pub fn total_routed_expert_bytes(&self) -> u64 {
        self.per_layer_routed_expert_bytes.values().copied().sum()
    }

    pub fn routed_expert_layer_count(&self) -> u32 {
        self.per_layer_routed_expert_bytes.len() as u32
    }

    pub fn total_mtp_bytes(&self) -> u64 {
        self.non_layer_mtp_bytes
            + self.per_layer_mtp_bytes.values().copied().sum::<u64>()
    }

    pub fn mtp_layer_count(&self) -> u32 {
        self.per_layer_mtp_bytes.len() as u32
    }

    /// Bytes of routed MoE experts that move to CPU when `--n-cpu-moe N` is set
    /// (first `N` MoE layers in ascending block order).
    pub fn cpu_routed_expert_bytes_for_n(&self, n_cpu_moe: u32) -> u64 {
        self.per_layer_routed_expert_bytes
            .values()
            .take(n_cpu_moe as usize)
            .copied()
            .sum()
    }
}

/// Complete metadata parsed natively from a GGUF file (or split GGUF shard set) without launching llama.cpp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgufMetadata {
    pub path: PathBuf,
    pub shard_paths: Vec<PathBuf>,
    pub gguf_version: u32,
    pub total_size_bytes: u64,
    pub architecture: Option<String>,
    pub name: Option<String>,
    pub size_label: Option<String>,
    pub file_type: Option<u32>,
    pub quantization: Option<String>,
    pub parameter_count: Option<u64>,
    pub block_count: Option<u32>,
    /// Effective logical layer count seen by `llama.cpp` for `-ngl` and KV cache allocation.
    /// Equals `block_count` for standard models; for looped transformers (e.g. `inkling2_loop`),
    /// includes repeated loop passes: `block_count + (loop_count - 1) * (loop_end - loop_start)`.
    #[serde(default)]
    pub effective_block_count: Option<u32>,
    pub context_length: Option<u32>,
    pub embedding_length: Option<u32>,
    pub head_count: Option<u32>,
    pub head_count_kv: Option<u32>,
    /// Per-layer KV head counts when `{arch}.attention.head_count_kv` is stored as a GGUF array
    /// (e.g. hybrid SSM/conv models with `0` on non-attention layers, or Gemma 4 with differing SWA/full KV heads).
    #[serde(default)]
    pub head_count_kv_per_layer: Vec<u32>,
    pub key_length: Option<u32>,
    pub value_length: Option<u32>,
    #[serde(default)]
    pub key_length_swa: Option<u32>,
    #[serde(default)]
    pub value_length_swa: Option<u32>,
    #[serde(default)]
    pub sliding_window: Option<u32>,
    #[serde(default)]
    pub sliding_window_pattern_interval: Option<u32>,
    #[serde(default)]
    pub sliding_window_pattern_layers: Vec<bool>,
    #[serde(default)]
    pub shared_kv_layers: Option<u32>,
    #[serde(default)]
    pub kv_lora_rank: Option<u32>,
    #[serde(default)]
    pub key_length_mla: Option<u32>,
    #[serde(default)]
    pub value_length_mla: Option<u32>,
    #[serde(default)]
    pub rope_dimension_count: Option<u32>,
    pub rope_freq_base: Option<f32>,
    pub expert_count: Option<u32>,
    pub expert_used_count: Option<u32>,
    pub is_moe: bool,
    #[serde(default)]
    pub full_attention_interval: Option<u32>,
    #[serde(default)]
    pub ssm_state_size: Option<u32>,
    #[serde(default)]
    pub ssm_conv_kernel: Option<u32>,
    #[serde(default)]
    pub ssm_inner_size: Option<u32>,
    #[serde(default)]
    pub ssm_group_count: Option<u32>,
    #[serde(default)]
    pub ssm_time_step_rank: Option<u32>,
    #[serde(default)]
    pub nextn_predict_layers: Option<u32>,
    #[serde(default)]
    pub has_mtp: bool,
    pub chat_template: Option<String>,
    pub supports_reasoning: bool,
    pub supports_reasoning_effort: bool,
    pub supports_enable_thinking: bool,
    pub supports_tools: bool,
    pub is_embedding: bool,
    #[serde(default)]
    pub tokenizer_model: Option<String>,
    #[serde(default)]
    pub tokenizer_pre: Option<String>,
    #[serde(default)]
    pub vocab_size: Option<u32>,
    pub tensor_count: u64,
    pub tensor_summary: TensorMemorySummary,
}

impl GgufMetadata {
    /// Returns the effective logical layer count seen by `llama.cpp` (`n_layer`),
    /// falling back to physical `block_count` for legacy/non-looped metadata.
    pub fn effective_layer_count(&self) -> Option<u32> {
        self.effective_block_count.or(self.block_count)
    }
}

#[derive(Debug, Clone)]
enum ScalarValue {
    U64(u64),
    I64(i64),
    F64(f64),
    #[allow(dead_code)]
    Bool(bool),
    String(String),
    U32Array(Vec<u32>),
    BoolArray(Vec<bool>),
}

impl ScalarValue {
    fn as_u32(&self) -> Option<u32> {
        match *self {
            Self::U64(v) => u32::try_from(v).ok(),
            Self::I64(v) => u32::try_from(v).ok(),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U64(v) => Some(v),
            Self::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F64(v) => Some(v as f32),
            Self::U64(v) => Some(v as f32),
            Self::I64(v) => Some(v as f32),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    fn as_u32_vec(&self) -> Option<Vec<u32>> {
        match self {
            Self::U32Array(v) => Some(v.clone()),
            _ => None,
        }
    }

    fn as_bool_vec(&self) -> Option<Vec<bool>> {
        match self {
            Self::BoolArray(v) => Some(v.clone()),
            Self::U32Array(v) => Some(v.iter().map(|&x| x != 0).collect()),
            _ => None,
        }
    }
}

/// Parses a GGUF file (automatically resolving sibling split shards `*-00001-of-NNNNN.gguf`).
pub fn parse_gguf_file(path: &Path) -> Result<GgufMetadata, GgufError> {
    let shard_paths = discover_split_shards(path);
    let primary_path = shard_paths.first().map(PathBuf::as_path).unwrap_or(path);

    let (version, kv, mut tensor_count, mut summary) = parse_single_shard(primary_path, true)?;
    let mut total_size_bytes = fs::metadata(primary_path).map(|m| m.len()).unwrap_or(0);

    // If split GGUF has multiple shards, parse remaining shards' tensor tables for exact byte accounting.
    for shard in shard_paths.iter().skip(1) {
        total_size_bytes = total_size_bytes.saturating_add(fs::metadata(shard).map(|m| m.len()).unwrap_or(0));
        if let Ok((_, _, scount, ssum)) = parse_single_shard(shard, false) {
            tensor_count = tensor_count.saturating_add(scount);
            merge_tensor_summary(&mut summary, ssum);
        }
    }

    let architecture = kv
        .get("general.architecture")
        .and_then(ScalarValue::as_str)
        .map(ToOwned::to_owned);
    let name = kv
        .get("general.name")
        .and_then(ScalarValue::as_str)
        .map(ToOwned::to_owned);
    let size_label = kv
        .get("general.size_label")
        .and_then(ScalarValue::as_str)
        .map(ToOwned::to_owned);
    let file_type = kv.get("general.file_type").and_then(ScalarValue::as_u32);

    let arch_key = |suffix: &str| -> Option<&ScalarValue> {
        let arch = architecture.as_deref()?;
        kv.get(&format!("{arch}.{suffix}"))
    };

    let block_count = arch_key("block_count").and_then(ScalarValue::as_u32);
    let loop_count = arch_key("loop.count").and_then(ScalarValue::as_u32);
    let loop_start = arch_key("loop.start").and_then(ScalarValue::as_u32);
    let loop_end = arch_key("loop.end").and_then(ScalarValue::as_u32);
    let effective_block_count = match (block_count, loop_count, loop_start, loop_end) {
        (Some(bc), Some(lc), Some(ls), Some(le)) if lc > 1 && le > ls => {
            Some(bc.saturating_add((lc - 1).saturating_mul(le - ls)))
        }
        _ => block_count,
    };
    let context_length = arch_key("context_length").and_then(ScalarValue::as_u32);
    let embedding_length = arch_key("embedding_length").and_then(ScalarValue::as_u32);
    let head_count = arch_key("attention.head_count").and_then(ScalarValue::as_u32);
    let head_count_kv_val = arch_key("attention.head_count_kv");
    let head_count_kv_per_layer = head_count_kv_val
        .and_then(ScalarValue::as_u32_vec)
        .unwrap_or_default();
    let head_count_kv = head_count_kv_val
        .and_then(ScalarValue::as_u32)
        .or_else(|| {
            head_count_kv_per_layer
                .iter()
                .copied()
                .filter(|&h| h > 0)
                .max()
        })
        .or(head_count);
    let key_length = arch_key("attention.key_length").and_then(ScalarValue::as_u32);
    let value_length = arch_key("attention.value_length").and_then(ScalarValue::as_u32);
    let key_length_swa = arch_key("attention.key_length_swa").and_then(ScalarValue::as_u32);
    let value_length_swa = arch_key("attention.value_length_swa").and_then(ScalarValue::as_u32);
    let sliding_window = arch_key("attention.sliding_window").and_then(ScalarValue::as_u32);
    let swa_pattern_val = arch_key("attention.sliding_window_pattern");
    let sliding_window_pattern_interval = swa_pattern_val.and_then(ScalarValue::as_u32);
    let sliding_window_pattern_layers = swa_pattern_val
        .and_then(ScalarValue::as_bool_vec)
        .unwrap_or_default();
    let shared_kv_layers = arch_key("attention.shared_kv_layers").and_then(ScalarValue::as_u32);
    let kv_lora_rank = arch_key("attention.kv_lora_rank").and_then(ScalarValue::as_u32);
    let key_length_mla = arch_key("attention.key_length_mla").and_then(ScalarValue::as_u32);
    let value_length_mla = arch_key("attention.value_length_mla").and_then(ScalarValue::as_u32);
    let rope_dimension_count = arch_key("rope.dimension_count").and_then(ScalarValue::as_u32);
    let rope_freq_base = arch_key("rope.freq_base").and_then(ScalarValue::as_f32);
    let expert_count = arch_key("expert_count").and_then(ScalarValue::as_u32);
    let expert_used_count = arch_key("expert_used_count").and_then(ScalarValue::as_u32);
    let full_attention_interval = arch_key("full_attention_interval").and_then(ScalarValue::as_u32);
    let ssm_state_size = arch_key("ssm.state_size").and_then(ScalarValue::as_u32);
    let ssm_conv_kernel = arch_key("ssm.conv_kernel").and_then(ScalarValue::as_u32);
    let ssm_inner_size = arch_key("ssm.inner_size").and_then(ScalarValue::as_u32);
    let ssm_group_count = arch_key("ssm.group_count").and_then(ScalarValue::as_u32);
    let ssm_time_step_rank = arch_key("ssm.time_step_rank").and_then(ScalarValue::as_u32);
    let nextn_predict_layers = arch_key("nextn_predict_layers")
        .or_else(|| arch_key("mtp_num_hidden_layers"))
        .or_else(|| arch_key("num_nextn_predict_layers"))
        .and_then(ScalarValue::as_u32);

    // Consolidate any whole MTP/NextN transformer blocks (e.g. blk.64.* when blk.64.nextn.* exists
    // or blk.{idx >= block_count} when nextn_predict_layers > 0) so their attention/FFN weights
    // are accounted under MTP heads rather than double-counted in dense/MoE layers.
    consolidate_mtp_blocks(&mut summary, block_count, nextn_predict_layers);

    let has_mtp = summary.total_mtp_bytes() > 0 || nextn_predict_layers.is_some_and(|n| n > 0);

    let is_moe = expert_count.is_some_and(|n| n > 1)
        || !summary.per_layer_routed_expert_bytes.is_empty()
        || architecture
            .as_deref()
            .is_some_and(|a| a.to_ascii_lowercase().contains("moe") || a == "mixtral" || a == "deepseek2");

    let chat_template = kv
        .get("tokenizer.chat_template")
        .and_then(ScalarValue::as_str)
        .map(ToOwned::to_owned);
    let tokenizer_model = kv
        .get("tokenizer.ggml.model")
        .and_then(ScalarValue::as_str)
        .map(ToOwned::to_owned);
    let tokenizer_pre = kv
        .get("tokenizer.ggml.pre")
        .and_then(ScalarValue::as_str)
        .map(ToOwned::to_owned);
    let vocab_size = kv
        .get("tokenizer.ggml.tokens.__len")
        .and_then(ScalarValue::as_u32)
        .or_else(|| arch_key("vocab_size").and_then(ScalarValue::as_u32));

    let (supports_reasoning, supports_reasoning_effort, supports_enable_thinking, supports_tools) =
        inspect_chat_template(
            chat_template.as_deref(),
            name.as_deref(),
            primary_path.file_name().and_then(|s| s.to_str()),
        );

    let is_embedding = architecture.as_deref().is_some_and(|a| {
        let low = a.to_ascii_lowercase();
        low.contains("bert") || low.contains("nomic") || low.contains("bge") || low.contains("e5")
    }) || primary_path
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|f| f.to_ascii_lowercase().contains("embed"));

    let quantization = file_type
        .and_then(file_type_to_label)
        .map(ToOwned::to_owned)
        .or_else(|| infer_quant_from_filename(primary_path));

    let parameter_count = if summary.total_parameters > 0 {
        Some(summary.total_parameters)
    } else {
        kv.get("general.parameter_count")
            .and_then(ScalarValue::as_u64)
    };

    Ok(GgufMetadata {
        path: primary_path.to_path_buf(),
        shard_paths,
        gguf_version: version,
        total_size_bytes,
        architecture,
        name,
        size_label,
        file_type,
        quantization,
        parameter_count,
        block_count,
        effective_block_count,
        context_length,
        embedding_length,
        head_count,
        head_count_kv,
        head_count_kv_per_layer,
        key_length,
        value_length,
        key_length_swa,
        value_length_swa,
        sliding_window,
        sliding_window_pattern_interval,
        sliding_window_pattern_layers,
        shared_kv_layers,
        kv_lora_rank,
        key_length_mla,
        value_length_mla,
        rope_dimension_count,
        rope_freq_base,
        expert_count,
        expert_used_count,
        is_moe,
        full_attention_interval,
        ssm_state_size,
        ssm_conv_kernel,
        ssm_inner_size,
        ssm_group_count,
        ssm_time_step_rank,
        nextn_predict_layers,
        has_mtp,
        chat_template,
        supports_reasoning,
        supports_reasoning_effort,
        supports_enable_thinking,
        supports_tools,
        is_embedding,
        tokenizer_model,
        tokenizer_pre,
        vocab_size,
        tensor_count,
        tensor_summary: summary,
    })
}

fn parse_single_shard(
    path: &Path,
    collect_kv: bool,
) -> Result<(u32, BTreeMap<String, ScalarValue>, u64, TensorMemorySummary), GgufError> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic);
    }

    let version = read_u32(&mut reader)?;
    if version != 2 && version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = read_u64(&mut reader)?;
    let kv_count = read_u64(&mut reader)?;
    if kv_count > 1_000_000 || tensor_count > 2_000_000 {
        return Err(GgufError::Malformed("unreasonable header counts".into()));
    }

    let mut kv = BTreeMap::new();
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut reader)?;
        let vtype = read_u32(&mut reader)?;
        if collect_kv && key == "tokenizer.ggml.tokens" && vtype == 9 {
            let elem_type = read_u32(&mut reader)?;
            let count = read_u64(&mut reader)?;
            if count > MAX_ARRAY_SKIP {
                return Err(GgufError::Malformed(format!("array count {count} too large")));
            }
            kv.insert(
                "tokenizer.ggml.tokens.__len".to_string(),
                ScalarValue::U64(count),
            );
            if let Some(elem_sz) = fixed_type_size(elem_type) {
                skip_bytes(&mut reader, count.saturating_mul(elem_sz))?;
            } else {
                for _ in 0..count {
                    read_or_skip_value(&mut reader, elem_type, false)?;
                }
            }
            continue;
        }
        let keep = collect_kv && should_keep_kv_key(&key);
        if let Some(val) = read_or_skip_value(&mut reader, vtype, keep)?
            && keep
        {
            kv.insert(key, val);
        }
    }

    let mut summary = TensorMemorySummary::default();
    for _ in 0..tensor_count {
        let tname = read_gguf_string(&mut reader)?;
        let n_dims = read_u32(&mut reader)?;
        if n_dims > 16 {
            return Err(GgufError::Malformed(format!("tensor {tname} n_dims={n_dims}")));
        }
        let mut elems: u64 = 1;
        for _ in 0..n_dims {
            let dim = read_u64(&mut reader)?;
            elems = elems.saturating_mul(dim);
        }
        let ggml_type = GgmlType::from_u32(read_u32(&mut reader)?);
        let _offset = read_u64(&mut reader)?;

        let nbytes = ggml_type.tensor_bytes(elems);
        summary.total_tensor_bytes = summary.total_tensor_bytes.saturating_add(nbytes);
        summary.total_parameters = summary.total_parameters.saturating_add(elems);
        classify_tensor(&tname, nbytes, &mut summary);
    }

    Ok((version, kv, tensor_count, summary))
}

fn should_keep_kv_key(key: &str) -> bool {
    // Keep all scalar model metadata & chat template; skip massive tokenizer vocab arrays.
    !key.starts_with("tokenizer.ggml.tokens")
        && !key.starts_with("tokenizer.ggml.scores")
        && !key.starts_with("tokenizer.ggml.token_type")
        && !key.starts_with("tokenizer.ggml.merges")
}

fn classify_tensor(name: &str, nbytes: u64, summary: &mut TensorMemorySummary) {
    if let Some(rest) = name.strip_prefix("blk.")
        && let Some((layer_str, tensor_part)) = rest.split_once('.')
        && let Ok(layer_idx) = layer_str.parse::<u32>()
    {
        if is_mtp_tensor_name(tensor_part) {
            *summary.per_layer_mtp_bytes.entry(layer_idx).or_insert(0) += nbytes;
        } else if is_routed_moe_expert_tensor(tensor_part) {
            *summary
                .per_layer_routed_expert_bytes
                .entry(layer_idx)
                .or_insert(0) += nbytes;
        } else {
            *summary
                .per_layer_hot_bytes
                .entry(layer_idx)
                .or_insert(0) += nbytes;
        }
    } else if is_mtp_tensor_name(name) {
        summary.non_layer_mtp_bytes = summary.non_layer_mtp_bytes.saturating_add(nbytes);
    } else {
        summary.non_layer_hot_bytes = summary.non_layer_hot_bytes.saturating_add(nbytes);
    }
}

/// Identifies Multi-Token Prediction (MTP / NextN) tensor names.
pub fn is_mtp_tensor_name(name: &str) -> bool {
    let low = name.to_ascii_lowercase();
    low.contains("nextn")
        || low.starts_with("mtp.")
        || low.starts_with("mtp_")
        || low.contains(".mtp.")
        || low.contains("_mtp.")
}

fn consolidate_mtp_blocks(
    summary: &mut TensorMemorySummary,
    block_count: Option<u32>,
    nextn_predict_layers: Option<u32>,
) {
    let mut mtp_indices: BTreeSet<u32> = summary.per_layer_mtp_bytes.keys().copied().collect();

    if let Some(k) = nextn_predict_layers
        && k > 0
        && let Some(bc) = block_count
    {
        // Case A: MTP blocks live at indices >= block_count (e.g., blk.64.* when block_count = 64).
        let extra_blocks: Vec<u32> = summary
            .per_layer_hot_bytes
            .keys()
            .chain(summary.per_layer_routed_expert_bytes.keys())
            .copied()
            .filter(|&idx| idx >= bc)
            .collect();
        if !extra_blocks.is_empty() {
            mtp_indices.extend(extra_blocks);
        } else if mtp_indices.is_empty() && bc > k {
            // Case B: block_count includes the appended `k` NextN blocks at the tail.
            for idx in (bc - k)..bc {
                if summary.per_layer_hot_bytes.contains_key(&idx)
                    || summary.per_layer_routed_expert_bytes.contains_key(&idx)
                {
                    mtp_indices.insert(idx);
                }
            }
        }
    }

    for idx in mtp_indices {
        let mut moved = 0u64;
        if let Some(b) = summary.per_layer_hot_bytes.remove(&idx) {
            moved = moved.saturating_add(b);
        }
        if let Some(b) = summary.per_layer_routed_expert_bytes.remove(&idx) {
            moved = moved.saturating_add(b);
        }
        if moved > 0 {
            *summary.per_layer_mtp_bytes.entry(idx).or_insert(0) += moved;
        }
    }
}

/// Identifies routed MoE expert weight tensors eligible for `--cpu-moe` / `--n-cpu-moe` RAM offload
/// (explicitly excluding shared experts `*_shexp*` and router gates `ffn_gate_inp*`).
pub fn is_routed_moe_expert_tensor(tensor_part: &str) -> bool {
    if tensor_part.contains("shexp") || tensor_part.contains("gate_inp") {
        return false;
    }
    tensor_part.contains("_exps")
        || tensor_part.contains(".experts.")
        || (tensor_part.starts_with("ffn_") && tensor_part.contains("_exp."))
}

fn merge_tensor_summary(dst: &mut TensorMemorySummary, src: TensorMemorySummary) {
    dst.non_layer_hot_bytes = dst.non_layer_hot_bytes.saturating_add(src.non_layer_hot_bytes);
    dst.non_layer_mtp_bytes = dst.non_layer_mtp_bytes.saturating_add(src.non_layer_mtp_bytes);
    dst.total_tensor_bytes = dst.total_tensor_bytes.saturating_add(src.total_tensor_bytes);
    dst.total_parameters = dst.total_parameters.saturating_add(src.total_parameters);
    for (k, v) in src.per_layer_hot_bytes {
        *dst.per_layer_hot_bytes.entry(k).or_insert(0) += v;
    }
    for (k, v) in src.per_layer_routed_expert_bytes {
        *dst.per_layer_routed_expert_bytes.entry(k).or_insert(0) += v;
    }
    for (k, v) in src.per_layer_mtp_bytes {
        *dst.per_layer_mtp_bytes.entry(k).or_insert(0) += v;
    }
}

fn inspect_chat_template(
    template: Option<&str>,
    model_name: Option<&str>,
    filename: Option<&str>,
) -> (bool, bool, bool, bool) {
    let tmpl = template.unwrap_or("");
    let tmpl_low = tmpl.to_ascii_lowercase();
    let supports_reasoning_effort = tmpl_low.contains("reasoning_effort");
    let supports_enable_thinking = tmpl_low.contains("enable_thinking");
    let has_think_tags = tmpl_low.contains("<think>")
        || tmpl_low.contains("reasoning_content")
        || tmpl_low.contains("<|thought|>")
        || tmpl_low.contains("<|channel|>analysis");

    let name_hint = format!(
        "{} {}",
        model_name.unwrap_or("").to_ascii_lowercase(),
        filename.unwrap_or("").to_ascii_lowercase()
    );
    let name_suggests_reasoning = name_hint.contains("deepseek-r1")
        || name_hint.contains("qwq")
        || name_hint.contains("qwen3")
        || name_hint.contains("thinking")
        || name_hint.contains("reasoning")
        || name_hint.contains("gemma-4")
        || name_hint.contains("gemma4")
        || name_hint.contains("gpt-oss");

    let supports_reasoning = supports_reasoning_effort
        || supports_enable_thinking
        || has_think_tags
        || name_suggests_reasoning;

    let supports_tools = tmpl_low.contains("tools")
        || tmpl_low.contains("tool_calls")
        || tmpl_low.contains("<tool_call>");

    (
        supports_reasoning,
        supports_reasoning_effort,
        supports_enable_thinking,
        supports_tools,
    )
}

/// Compatibility status between a main model and a speculative-decoding draft model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DraftCompatibilityLevel {
    Compatible,
    Warning,
    Incompatible,
}

/// Detailed tokenizer/architecture compatibility verdict for speculative decoding (`-md`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DraftCompatibilityReport {
    pub level: DraftCompatibilityLevel,
    pub compatible: bool,
    pub reasons: Vec<String>,
}

/// Evaluates whether a draft GGUF model is tokenizer- and size-compatible with the main GGUF model
/// for speculative decoding (`llama-server -md`).
pub fn check_draft_compatibility(
    main: &GgufMetadata,
    draft: &GgufMetadata,
) -> DraftCompatibilityReport {
    let mut reasons = Vec::new();
    let mut has_incompatible = false;
    let mut has_warning = false;

    if main.path == draft.path {
        has_warning = true;
        reasons.push("Draft model is the same file as the main model.".to_string());
    }

    match (
        main.tokenizer_model.as_deref(),
        draft.tokenizer_model.as_deref(),
    ) {
        (Some(m), Some(d)) if !m.eq_ignore_ascii_case(d) => {
            has_incompatible = true;
            reasons.push(format!(
                "Tokenizer family mismatch: main uses '{m}', draft uses '{d}'."
            ));
        }
        (None, _) | (_, None) => {
            has_warning = true;
            reasons.push("Tokenizer family metadata missing in one or both GGUF headers.".to_string());
        }
        _ => {}
    }

    match (main.vocab_size, draft.vocab_size) {
        (Some(mv), Some(dv)) if mv != dv => {
            let diff = mv.abs_diff(dv);
            if diff <= 64 {
                has_warning = true;
                reasons.push(format!(
                    "Vocabulary size differs slightly (main {mv} vs draft {dv}; special-token offset)."
                ));
            } else {
                has_incompatible = true;
                reasons.push(format!(
                    "Vocabulary size mismatch: main has {mv} tokens, draft has {dv} tokens."
                ));
            }
        }
        (None, _) | (_, None) => {
            has_warning = true;
            reasons.push("Vocabulary size could not be verified from GGUF metadata.".to_string());
        }
        _ => {}
    }

    if let (Some(mp), Some(dp)) = (
        main.tokenizer_pre.as_deref(),
        draft.tokenizer_pre.as_deref(),
    ) && !mp.eq_ignore_ascii_case(dp)
    {
        has_warning = true;
        reasons.push(format!(
            "Pre-tokenizer mismatch: main '{mp}' vs draft '{dp}'."
        ));
    }

    if let (Some(ma), Some(da)) = (main.architecture.as_deref(), draft.architecture.as_deref())
        && !ma.eq_ignore_ascii_case(da)
    {
        has_warning = true;
        reasons.push(format!(
            "Architecture differs (main '{ma}' vs draft '{da}') — verify shared tokenizer."
        ));
    }

    if draft.total_size_bytes > 0 && main.total_size_bytes > 0 && draft.total_size_bytes >= main.total_size_bytes {
        has_warning = true;
        reasons.push("Draft model file is not smaller than the main model; speculative decoding may slow generation.".to_string());
    }

    let level = if has_incompatible {
        DraftCompatibilityLevel::Incompatible
    } else if has_warning {
        DraftCompatibilityLevel::Warning
    } else {
        reasons.push("Tokenizer family, vocabulary size, and architecture match.".to_string());
        DraftCompatibilityLevel::Compatible
    };

    DraftCompatibilityReport {
        level,
        compatible: !has_incompatible,
        reasons,
    }
}

/// Discovers sibling split shards (e.g. `model-00001-of-00003.gguf`, `model-00002-of-00003.gguf`, ...).
pub fn discover_split_shards(path: &Path) -> Vec<PathBuf> {
    let Some(file_name) = path.file_name().and_then(|s| s.to_str()) else {
        return vec![path.to_path_buf()];
    };
    let Some((prefix, total)) = parse_split_shard_pattern(file_name) else {
        return vec![path.to_path_buf()];
    };
    let Some(parent) = path.parent() else {
        return vec![path.to_path_buf()];
    };

    let mut shards = Vec::with_capacity(total as usize);
    for i in 1..=total {
        let candidate = parent.join(format!("{prefix}-{i:05}-of-{total:05}.gguf"));
        if candidate.exists() {
            shards.push(candidate);
        }
    }
    if shards.is_empty() {
        vec![path.to_path_buf()]
    } else {
        shards
    }
}

/// Matches `<prefix>-NNNNN-of-MMMMM.gguf` (case-insensitive extension) and returns `(prefix, total)`.
pub fn parse_split_shard_pattern(file_name: &str) -> Option<(String, u32)> {
    let stem = file_name
        .strip_suffix(".gguf")
        .or_else(|| file_name.strip_suffix(".GGUF"))?;
    let (before_of, total_str) = stem.rsplit_once("-of-")?;
    if total_str.len() != 5 || !total_str.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let (prefix, index_str) = before_of.rsplit_once('-')?;
    if index_str.len() != 5 || !index_str.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let total: u32 = total_str.parse().ok()?;
    if total == 0 || total > 999 {
        return None;
    }
    Some((prefix.to_string(), total))
}

/// Returns `true` if `file_name` is a non-primary split shard (`00002-of-NNNNN`, etc.).
pub fn is_secondary_split_shard(file_name: &str) -> bool {
    let Some(stem) = file_name
        .strip_suffix(".gguf")
        .or_else(|| file_name.strip_suffix(".GGUF"))
    else {
        return false;
    };
    let Some((before_of, total_str)) = stem.rsplit_once("-of-") else {
        return false;
    };
    if total_str.len() != 5 || !total_str.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let Some((_, index_str)) = before_of.rsplit_once('-') else {
        return false;
    };
    index_str.len() == 5
        && index_str.chars().all(|c| c.is_ascii_digit())
        && index_str != "00001"
}

fn file_type_to_label(ft: u32) -> Option<&'static str> {
    Some(match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4",
        _ => return None,
    })
}

pub fn infer_quant_from_filename(path: &Path) -> Option<String> {
    let upper = path.file_stem()?.to_str()?.to_ascii_uppercase();
    const KNOWN: &[&str] = &[
        "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S",
        "IQ3_M", "IQ4_NL", "IQ4_XS", "Q2_K_XL", "Q2_K_S", "Q2_K", "Q3_K_XL", "Q3_K_L", "Q3_K_M",
        "Q3_K_S", "Q4_K_XL", "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1", "Q5_K_XL", "Q5_K_M", "Q5_K_S",
        "Q5_0", "Q5_1", "Q6_K_XL", "Q6_K", "Q8_K_XL", "Q8_0", "MXFP4", "BF16", "F16", "F32",
    ];
    for q in KNOWN {
        if upper.contains(q) {
            return Some((*q).to_string());
        }
    }
    None
}

// --- Binary helpers ---

fn read_u8<R: Read>(r: &mut R) -> Result<u8, GgufError> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_i8<R: Read>(r: &mut R) -> Result<i8, GgufError> {
    Ok(read_u8(r)? as i8)
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16, GgufError> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_i16<R: Read>(r: &mut R) -> Result<i16, GgufError> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(i16::from_le_bytes(b))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, GgufError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32, GgufError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, GgufError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64, GgufError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(i64::from_le_bytes(b))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32, GgufError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn read_f64<R: Read>(r: &mut R) -> Result<f64, GgufError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

fn read_gguf_string<R: Read>(r: &mut R) -> Result<String, GgufError> {
    let len = read_u64(r)? as usize;
    if len > MAX_STRING_LEN {
        return Err(GgufError::Malformed(format!("string length {len} exceeds cap")));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn skip_bytes<R: Read>(r: &mut R, mut n: u64) -> Result<(), GgufError> {
    let mut scratch = [0u8; 4096];
    while n > 0 {
        let chunk = (n as usize).min(scratch.len());
        r.read_exact(&mut scratch[..chunk])?;
        n -= chunk as u64;
    }
    Ok(())
}

fn fixed_type_size(vtype: u32) -> Option<u64> {
    match vtype {
        0 | 1 | 7 => Some(1), // UINT8, INT8, BOOL
        2 | 3 => Some(2),     // UINT16, INT16
        4..=6 => Some(4),     // UINT32, INT32, FLOAT32
        10..=12 => Some(8),   // UINT64, INT64, FLOAT64
        _ => None,
    }
}

fn read_or_skip_value<R: Read>(
    r: &mut R,
    vtype: u32,
    keep: bool,
) -> Result<Option<ScalarValue>, GgufError> {
    match vtype {
        0 => {
            let v = read_u8(r)?;
            Ok(keep.then_some(ScalarValue::U64(v as u64)))
        }
        1 => {
            let v = read_i8(r)?;
            Ok(keep.then_some(ScalarValue::I64(v as i64)))
        }
        2 => {
            let v = read_u16(r)?;
            Ok(keep.then_some(ScalarValue::U64(v as u64)))
        }
        3 => {
            let v = read_i16(r)?;
            Ok(keep.then_some(ScalarValue::I64(v as i64)))
        }
        4 => {
            let v = read_u32(r)?;
            Ok(keep.then_some(ScalarValue::U64(v as u64)))
        }
        5 => {
            let v = read_i32(r)?;
            Ok(keep.then_some(ScalarValue::I64(v as i64)))
        }
        6 => {
            let v = read_f32(r)?;
            Ok(keep.then_some(ScalarValue::F64(v as f64)))
        }
        7 => {
            let v = read_u8(r)? != 0;
            Ok(keep.then_some(ScalarValue::Bool(v)))
        }
        8 => {
            let len = read_u64(r)?;
            if !keep {
                skip_bytes(r, len)?;
                Ok(None)
            } else {
                if len as usize > MAX_STRING_LEN {
                    return Err(GgufError::Malformed(format!("string length {len} exceeds cap")));
                }
                let mut buf = vec![0u8; len as usize];
                r.read_exact(&mut buf)?;
                Ok(Some(ScalarValue::String(
                    String::from_utf8_lossy(&buf).into_owned(),
                )))
            }
        }
        9 => {
            // Array: elem_type (u32), len (u64), elements...
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            if count > MAX_ARRAY_SKIP {
                return Err(GgufError::Malformed(format!("array count {count} too large")));
            }
            // Retain small per-layer numeric/bool metadata arrays (e.g. head_count_kv, sliding_window_pattern).
            const MAX_KEEP_ARRAY_LEN: u64 = 1024;
            if keep && count <= MAX_KEEP_ARRAY_LEN {
                match elem_type {
                    7 => {
                        let mut vals = Vec::with_capacity(count as usize);
                        for _ in 0..count {
                            vals.push(read_u8(r)? != 0);
                        }
                        return Ok(Some(ScalarValue::BoolArray(vals)));
                    }
                    0 | 1 | 2 | 3 | 4 | 5 | 10 | 11 => {
                        let mut vals = Vec::with_capacity(count as usize);
                        for _ in 0..count {
                            let v = match elem_type {
                                0 => read_u8(r)? as u32,
                                1 => read_i8(r)?.max(0) as u32,
                                2 => read_u16(r)? as u32,
                                3 => read_i16(r)?.max(0) as u32,
                                4 => read_u32(r)?,
                                5 => read_i32(r)?.max(0) as u32,
                                10 => read_u64(r)?.min(u32::MAX as u64) as u32,
                                11 => read_i64(r)?.clamp(0, u32::MAX as i64) as u32,
                                _ => unreachable!(),
                            };
                            vals.push(v);
                        }
                        return Ok(Some(ScalarValue::U32Array(vals)));
                    }
                    _ => {}
                }
            }
            if let Some(elem_sz) = fixed_type_size(elem_type) {
                skip_bytes(r, count.saturating_mul(elem_sz))?;
            } else {
                for _ in 0..count {
                    read_or_skip_value(r, elem_type, false)?;
                }
            }
            Ok(None)
        }
        10 => {
            let v = read_u64(r)?;
            Ok(keep.then_some(ScalarValue::U64(v)))
        }
        11 => {
            let v = read_i64(r)?;
            Ok(keep.then_some(ScalarValue::I64(v)))
        }
        12 => {
            let v = read_f64(r)?;
            Ok(keep.then_some(ScalarValue::F64(v)))
        }
        other => Err(GgufError::UnsupportedValueType(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn write_kv_str(buf: &mut Vec<u8>, k: &str, v: &str) {
        write_str(buf, k);
        buf.extend_from_slice(&8u32.to_le_bytes());
        write_str(buf, v);
    }

    fn write_kv_u32(buf: &mut Vec<u8>, k: &str, v: u32) {
        write_str(buf, k);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_tensor(buf: &mut Vec<u8>, name: &str, dims: &[u64], ggml_type: u32) {
        write_str(buf, name);
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&ggml_type.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset
    }

    #[test]
    fn parses_synthetic_moe_gguf_header_and_tensors() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes()); // version 3
        data.extend_from_slice(&4u64.to_le_bytes()); // 4 tensors
        data.extend_from_slice(&9u64.to_le_bytes()); // 9 kv pairs

        write_kv_str(&mut data, "general.architecture", "qwen3moe");
        write_kv_str(&mut data, "general.name", "Qwen3-30B-A3B");
        write_kv_u32(&mut data, "general.file_type", 15); // Q4_K_M
        write_kv_u32(&mut data, "qwen3moe.block_count", 48);
        write_kv_u32(&mut data, "qwen3moe.context_length", 131072);
        write_kv_u32(&mut data, "qwen3moe.attention.head_count", 32);
        write_kv_u32(&mut data, "qwen3moe.attention.head_count_kv", 4);
        write_kv_u32(&mut data, "qwen3moe.expert_count", 128);
        write_kv_str(
            &mut data,
            "tokenizer.chat_template",
            "{% if enable_thinking %}<think>{% endif %}{% if tools %}<tool_call>{% endif %}",
        );

        // Tensors: embedding, attn (hot), shared expert (hot), routed experts (offloadable)
        write_tensor(&mut data, "token_embd.weight", &[2048, 151936], 12); // Q4_K
        write_tensor(&mut data, "blk.0.attn_q.weight", &[2048, 4096], 12);
        write_tensor(&mut data, "blk.0.ffn_up_shexp.weight", &[2048, 2048], 12);
        write_tensor(&mut data, "blk.0.ffn_gate_exps.weight", &[2048, 768, 128], 12);

        let tmp = std::env::temp_dir().join(format!("test_moe_{}.gguf", uuid::Uuid::new_v4()));
        {
            let mut f = File::create(&tmp).unwrap();
            f.write_all(&data).unwrap();
        }

        let meta = parse_gguf_file(&tmp).unwrap();
        let _ = fs::remove_file(&tmp);

        assert_eq!(meta.architecture.as_deref(), Some("qwen3moe"));
        assert_eq!(meta.quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(meta.block_count, Some(48));
        assert_eq!(meta.context_length, Some(131072));
        assert_eq!(meta.expert_count, Some(128));
        assert!(meta.is_moe);
        assert!(meta.supports_reasoning);
        assert!(meta.supports_enable_thinking);
        assert!(meta.supports_tools);
        assert!(meta.tensor_summary.non_layer_hot_bytes > 0);
        assert!(meta.tensor_summary.per_layer_hot_bytes.contains_key(&0));
        assert!(meta.tensor_summary.per_layer_routed_expert_bytes.contains_key(&0));
        assert!(
            meta.tensor_summary.total_routed_expert_bytes()
                > meta.tensor_summary.per_layer_hot_bytes[&0] * 10
        );
    }

    #[test]
    fn recognizes_split_shard_names() {
        assert_eq!(
            parse_split_shard_pattern("DeepSeek-R1-Q4_K_M-00001-of-00005.gguf"),
            Some(("DeepSeek-R1-Q4_K_M".to_string(), 5))
        );
        assert!(!is_secondary_split_shard("DeepSeek-R1-Q4_K_M-00001-of-00005.gguf"));
        assert!(is_secondary_split_shard("DeepSeek-R1-Q4_K_M-00002-of-00005.gguf"));
    }

    #[test]
    fn parses_hybrid_qwen35_with_mtp_heads() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&5u64.to_le_bytes()); // 5 tensors
        data.extend_from_slice(&9u64.to_le_bytes()); // 9 kv pairs

        write_kv_str(&mut data, "general.architecture", "qwen35");
        write_kv_str(&mut data, "general.name", "Qwen3.5-27B");
        write_kv_u32(&mut data, "qwen35.block_count", 64);
        write_kv_u32(&mut data, "qwen35.context_length", 262144);
        write_kv_u32(&mut data, "qwen35.attention.head_count", 24);
        write_kv_u32(&mut data, "qwen35.attention.head_count_kv", 4);
        write_kv_u32(&mut data, "qwen35.full_attention_interval", 4);
        write_kv_u32(&mut data, "qwen35.ssm.state_size", 128);
        write_kv_u32(&mut data, "qwen35.nextn_predict_layers", 1);

        // Dense layer 0, plus MTP block 64 (contains nextn.eh_proj + attn_q + ffn_gate)
        write_tensor(&mut data, "token_embd.weight", &[5120, 151936], 12);
        write_tensor(&mut data, "blk.0.attn_q.weight", &[5120, 5120], 12);
        write_tensor(&mut data, "blk.64.nextn.eh_proj.weight", &[10240, 5120], 12);
        write_tensor(&mut data, "blk.64.attn_q.weight", &[5120, 5120], 12);
        write_tensor(&mut data, "blk.64.ffn_gate.weight", &[5120, 17408], 12);

        let tmp = std::env::temp_dir().join(format!("test_qwen35_mtp_{}.gguf", uuid::Uuid::new_v4()));
        {
            let mut f = File::create(&tmp).unwrap();
            f.write_all(&data).unwrap();
        }

        let meta = parse_gguf_file(&tmp).unwrap();
        let _ = fs::remove_file(&tmp);

        assert_eq!(meta.architecture.as_deref(), Some("qwen35"));
        assert_eq!(meta.full_attention_interval, Some(4));
        assert_eq!(meta.ssm_state_size, Some(128));
        assert_eq!(meta.nextn_predict_layers, Some(1));
        assert!(meta.has_mtp);
        assert_eq!(meta.tensor_summary.mtp_layer_count(), 1);
        assert!(!meta.tensor_summary.per_layer_hot_bytes.contains_key(&64));
        assert!(meta.tensor_summary.per_layer_hot_bytes.contains_key(&0));
        assert!(meta.tensor_summary.total_mtp_bytes() > meta.tensor_summary.per_layer_hot_bytes[&0]);
    }

    #[test]
    fn extracts_tokenizer_metadata_and_checks_draft_compatibility() {
        let write_tok_model = |tok_model: &str, pre: &str, vocab: u32, arch: &str, bytes: u64| {
            let mut m = GgufMetadata {
                path: PathBuf::from(format!("/{arch}-{vocab}-{bytes}.gguf")),
                shard_paths: vec![],
                gguf_version: 3,
                total_size_bytes: bytes,
                architecture: Some(arch.into()),
                name: None,
                size_label: None,
                file_type: None,
                quantization: None,
                parameter_count: None,
                block_count: None,
                effective_block_count: None,
                context_length: None,
                embedding_length: None,
                head_count: None,
                head_count_kv: None,
                head_count_kv_per_layer: Vec::new(),
                key_length: None,
                value_length: None,
                key_length_swa: None,
                value_length_swa: None,
                sliding_window: None,
                sliding_window_pattern_interval: None,
                sliding_window_pattern_layers: Vec::new(),
                shared_kv_layers: None,
                kv_lora_rank: None,
                key_length_mla: None,
                value_length_mla: None,
                rope_dimension_count: None,
                rope_freq_base: None,
                expert_count: None,
                expert_used_count: None,
                is_moe: false,
                full_attention_interval: None,
                ssm_state_size: None,
                ssm_conv_kernel: None,
                ssm_inner_size: None,
                ssm_group_count: None,
                ssm_time_step_rank: None,
                nextn_predict_layers: None,
                has_mtp: false,
                chat_template: None,
                supports_reasoning: false,
                supports_reasoning_effort: false,
                supports_enable_thinking: false,
                supports_tools: false,
                is_embedding: false,
                tokenizer_model: Some(tok_model.into()),
                tokenizer_pre: Some(pre.into()),
                vocab_size: Some(vocab),
                tensor_count: 0,
                tensor_summary: TensorMemorySummary::default(),
            };
            let _ = &mut m;
            m
        };

        let main = write_tok_model("gpt2", "qwen2", 151936, "qwen2", 18_000_000_000);
        let good_draft = write_tok_model("gpt2", "qwen2", 151936, "qwen2", 500_000_000);
        let rep_ok = check_draft_compatibility(&main, &good_draft);
        assert_eq!(rep_ok.level, DraftCompatibilityLevel::Compatible);
        assert!(rep_ok.compatible);

        let minor_vocab = write_tok_model("gpt2", "qwen2", 151940, "qwen2", 500_000_000);
        let rep_warn = check_draft_compatibility(&main, &minor_vocab);
        assert_eq!(rep_warn.level, DraftCompatibilityLevel::Warning);
        assert!(rep_warn.compatible);

        let bad_draft = write_tok_model("llama", "llama3", 128256, "llama", 1_000_000_000);
        let rep_bad = check_draft_compatibility(&main, &bad_draft);
        assert_eq!(rep_bad.level, DraftCompatibilityLevel::Incompatible);
        assert!(!rep_bad.compatible);

        // Verify binary GGUF token array length extraction without loading strings
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&3u64.to_le_bytes());
        write_kv_str(&mut data, "tokenizer.ggml.model", "gpt2");
        write_kv_str(&mut data, "tokenizer.ggml.pre", "qwen2");
        write_str(&mut data, "tokenizer.ggml.tokens");
        data.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
        data.extend_from_slice(&8u32.to_le_bytes()); // STRING elements
        data.extend_from_slice(&3u64.to_le_bytes()); // 3 tokens
        write_str(&mut data, "a");
        write_str(&mut data, "b");
        write_str(&mut data, "c");

        let tmp = std::env::temp_dir().join(format!("test_tok_{}.gguf", uuid::Uuid::new_v4()));
        File::create(&tmp).unwrap().write_all(&data).unwrap();
        let parsed = parse_gguf_file(&tmp).unwrap();
        let _ = fs::remove_file(&tmp);
        assert_eq!(parsed.tokenizer_model.as_deref(), Some("gpt2"));
        assert_eq!(parsed.tokenizer_pre.as_deref(), Some("qwen2"));
        assert_eq!(parsed.vocab_size, Some(3));
    }

    #[test]
    fn parses_looped_transformer_effective_block_count() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&5u64.to_le_bytes());
        write_kv_str(&mut data, "general.architecture", "inkling2_loop");
        write_kv_u32(&mut data, "inkling2_loop.block_count", 24);
        write_kv_u32(&mut data, "inkling2_loop.loop.count", 3);
        write_kv_u32(&mut data, "inkling2_loop.loop.start", 6);
        write_kv_u32(&mut data, "inkling2_loop.loop.end", 18);

        let tmp = std::env::temp_dir().join(format!("test_loop_{}.gguf", uuid::Uuid::new_v4()));
        File::create(&tmp).unwrap().write_all(&data).unwrap();
        let parsed = parse_gguf_file(&tmp).unwrap();
        let _ = fs::remove_file(&tmp);

        assert_eq!(parsed.block_count, Some(24));
        assert_eq!(parsed.effective_block_count, Some(48));
        assert_eq!(parsed.effective_layer_count(), Some(48));
    }

    #[test]
    fn parses_per_layer_kv_and_swa_metadata_arrays() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&10u64.to_le_bytes());

        write_kv_str(&mut data, "general.architecture", "gemma4");
        write_kv_u32(&mut data, "gemma4.block_count", 6);
        write_kv_u32(&mut data, "gemma4.attention.head_count", 16);
        // Per-layer u32 array for head_count_kv: [8, 8, 8, 8, 8, 2]
        write_str(&mut data, "gemma4.attention.head_count_kv");
        data.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
        data.extend_from_slice(&4u32.to_le_bytes()); // UINT32
        data.extend_from_slice(&6u64.to_le_bytes());
        for h in [8u32, 8, 8, 8, 8, 2] {
            data.extend_from_slice(&h.to_le_bytes());
        }
        write_kv_u32(&mut data, "gemma4.attention.sliding_window", 1024);
        // Per-layer bool array for sliding_window_pattern: [T, T, T, T, T, F]
        write_str(&mut data, "gemma4.attention.sliding_window_pattern");
        data.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
        data.extend_from_slice(&7u32.to_le_bytes()); // BOOL
        data.extend_from_slice(&6u64.to_le_bytes());
        for b in [1u8, 1, 1, 1, 1, 0] {
            data.push(b);
        }
        write_kv_u32(&mut data, "gemma4.attention.key_length", 512);
        write_kv_u32(&mut data, "gemma4.attention.key_length_swa", 256);
        write_kv_u32(&mut data, "gemma4.attention.value_length_swa", 256);
        write_kv_u32(&mut data, "gemma4.attention.shared_kv_layers", 2);

        let tmp = std::env::temp_dir().join(format!("test_gemma4_arr_{}.gguf", uuid::Uuid::new_v4()));
        File::create(&tmp).unwrap().write_all(&data).unwrap();
        let parsed = parse_gguf_file(&tmp).unwrap();
        let _ = fs::remove_file(&tmp);

        assert_eq!(parsed.head_count_kv, Some(8));
        assert_eq!(parsed.head_count_kv_per_layer, vec![8, 8, 8, 8, 8, 2]);
        assert_eq!(parsed.sliding_window, Some(1024));
        assert_eq!(
            parsed.sliding_window_pattern_layers,
            vec![true, true, true, true, true, false]
        );
        assert_eq!(parsed.key_length, Some(512));
        assert_eq!(parsed.key_length_swa, Some(256));
        assert_eq!(parsed.value_length_swa, Some(256));
        assert_eq!(parsed.shared_kv_layers, Some(2));
    }
}
