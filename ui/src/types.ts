export type NavTab =
  | 'benchmark'
  | 'server'
  | 'link'
  | 'library'
  | 'search'
  | 'runtime'
  | 'hardware'
  | 'settings';

export type PlacementStatus =
  | 'gpu_only'
  | 'hybrid_gpu_cpu'
  | 'gpu_spillover_warning'
  | 'cpu_only'
  | 'unknown';

export type FitVerdict =
  | 'full_gpu'
  | 'moe_hybrid'
  | 'partial_offload'
  | 'cpu_only'
  | 'too_large';

export type GuardrailLevel = 'off' | 'relaxed' | 'balanced' | 'strict';

export type ReasoningEffort =
  | 'default'
  | 'off'
  | 'minimal'
  | 'low'
  | 'medium'
  | 'high'
  | 'xhigh'
  | 'max';

export type ReasoningFormat = 'auto' | 'deepseek' | 'deepseek_legacy' | 'none';
export type FlashAttentionMode = 'auto' | 'on' | 'off';
export type KvQuantType =
  | 'f32'
  | 'f16'
  | 'bf16'
  | 'f8_e4m3'
  | 'q8_0'
  | 'q5_1'
  | 'q5_0'
  | 'q4_1'
  | 'q4_0'
  | 'iq4_nl';
export type RopeScalingType = 'none' | 'linear' | 'yarn';
export type SplitMode = 'none' | 'layer' | 'row' | 'tensor';

export type CliFlagValue =
  | { kind: 'flag'; value: boolean }
  | { kind: 'value'; value: string }
  | { kind: 'multi_arg'; value: string[] }
  | { kind: 'repeatable'; value: string[] };

export interface TensorOverride {
  pattern: string;
  buffer: string;
}

export type MoeOffloadMode =
  | { mode: 'full_gpu' }
  | { mode: 'all_experts_cpu' }
  | { mode: 'first_n_layers_experts_cpu'; value: number }
  | { mode: 'custom_overrides'; value: TensorOverride[] };

export interface ReasoningConfig {
  effort: ReasoningEffort;
  format: ReasoningFormat;
  budget: number | null;
  budget_message: string | null;
  enable_thinking: boolean | null;
}

export interface SamplingParams {
  system_prompt: string;
  temperature: number | null;
  top_k: number | null;
  top_p: number | null;
  min_p: number | null;
  typical_p: number | null;
  repeat_penalty: number | null;
  repeat_last_n: number | null;
  presence_penalty: number | null;
  frequency_penalty: number | null;
  dry_multiplier: number | null;
  dry_base: number | null;
  dry_allowed_length: number | null;
  xtc_probability: number | null;
  xtc_threshold: number | null;
  dynatemp_range: number | null;
  mirostat: number | null;
  mirostat_tau: number | null;
  mirostat_eta: number | null;
  max_tokens: number | null;
  stop_strings: string[];
  structured_output_enabled: boolean;
  json_schema: string | null;
  grammar: string | null;
}

export interface LoadConfig {
  model_path: string;
  model_alias: string | null;
  runtime_id: string | null;
  enable_vision: boolean;
  mmproj_path: string | null;
  offload_mmproj: boolean;
  ctx_size: number;
  batch_size: number;
  ubatch_size: number;
  parallel_slots: number;
  unified_kv_cache: boolean;
  cache_ram_mib: number | null;
  context_shift: boolean | null;
  gpu_layers: number | null;
  moe_offload: MoeOffloadMode;
  auto_fit: boolean | null;
  fit_ctx: number | null;
  fit_target_mib: number | null;
  threads: number | null;
  threads_batch: number | null;
  devices: string[];
  split_mode: SplitMode | null;
  main_gpu: number | null;
  tensor_split: string | null;
  offload_kv_cache: boolean;
  flash_attention: FlashAttentionMode;
  cache_type_k: KvQuantType | null;
  cache_type_v: KvQuantType | null;
  use_mmap: boolean;
  use_mlock: boolean;
  direct_io: boolean;
  rope_scaling: RopeScalingType | null;
  rope_freq_base: number | null;
  rope_freq_scale: number | null;
  yarn_ext_factor: number | null;
  yarn_attn_factor: number | null;
  yarn_beta_fast: number | null;
  yarn_beta_slow: number | null;
  jinja: boolean;
  chat_template_override: string | null;
  chat_template_file: string | null;
  reasoning: ReasoningConfig;
  mtp_enabled: boolean;
  mtp_spec_type: string | null;
  draft_model_path: string | null;
  draft_gpu_layers: number | null;
  draft_max: number | null;
  draft_min: number | null;
  draft_p_min: number | null;
  seed: number | null;
  embeddings: boolean;
  cli_flags: Record<string, CliFlagValue>;
  extra_args: string[];
  custom_cli_args: string | null;
  sampling_defaults: SamplingParams | null;
}

export interface ModelAliasRule {
  pattern: string;
  target: string;
  enabled: boolean;
}

export interface ServerConfig {
  port: number;
  serve_on_lan: boolean;
  require_auth: boolean;
  enable_cors: boolean;
  jit_loading: boolean;
  auto_unload_idle: boolean;
  max_idle_ttl_minutes: number;
  only_keep_last_jit_model: boolean;
  model_aliases?: ModelAliasRule[];
  fallback_to_first_loaded?: boolean;
}

export type LinkAccessPolicy = 'self_only' | 'same_tailnet';

export interface LinkConfig {
  enabled: boolean;
  auto_connect: boolean;
  device_name: string | null;
  expose_local_models: boolean;
  allow_remote_load: boolean;
  allow_remote_manage: boolean;
  access_policy: LinkAccessPolicy;
  control_url: string | null;
  expose_remote_in_openai_api: boolean;
  mesh_port: number;
}

export type LinkConnectionState =
  | 'disconnected'
  | 'starting'
  | 'needs_login'
  | 'connecting'
  | 'connected'
  | 'error';

export interface LinkStatus {
  state: LinkConnectionState;
  auth_url: string | null;
  hostname: string;
  node_id: string | null;
  tailscale_ip: string | null;
  magic_dns: string | null;
  tailnet_name: string | null;
  account_login: string | null;
  account_name: string | null;
  access_policy: string;
  expose_local_models: boolean;
  sidecar_available: boolean;
  sidecar_path: string | null;
  error: string | null;
}

export interface RemoteModelSummary {
  id: string;
  display_name: string;
  loaded: boolean;
  quantization: string | null;
  architecture: string | null;
  size_bytes: number;
  max_context_length: number | null;
  supports_vision: boolean;
  supports_reasoning: boolean;
  placement_status: string | null;
}

export interface RemotePeer {
  node_id: string;
  slug: string;
  hostname: string;
  device_name: string;
  tailscale_ip: string;
  magic_dns: string;
  os: string;
  online: boolean;
  connection_type: 'direct' | 'derp' | 'unknown' | string;
  relay: string | null;
  latency_ms: number | null;
  app_version: string;
  expose_local_models: boolean;
  allow_remote_load: boolean;
  allow_remote_manage: boolean;
  is_headless: boolean;
  platform: string;
  runtimes_count: number;
  active_runtime_tag: string | null;
  active_downloads_count: number;
  gpu_summary: string;
  vram_used_mib: number;
  vram_total_mib: number;
  ram_used_mib: number;
  ram_total_mib: number;
  loaded_models: RemoteModelSummary[];
  available_models: RemoteModelSummary[];
}

export interface RemoteRuntimesResponse {
  installed: InstalledRuntimeRecord[];
  active_runtime_id: string | null;
  platform: string;
  gpu_summary: string;
}

export interface RemoteInstallRuntimeResponse {
  ok: boolean;
  job_id: string;
  variant_id: string;
  message: string | null;
}

export interface RemoteActionResponse {
  ok: boolean;
  message: string | null;
}

export interface RemoteLoadResponse {
  ok: boolean;
  model_id: string;
  message: string | null;
}

export interface LiveMemoryContext {
  hardware_vram_total_mib: number;
  current_vram_used_mib: number;
  available_vram_mib: number;
  hardware_ram_total_mib: number;
  current_ram_used_mib: number;
  available_ram_mib: number;
  vram_sample_count: number;
}

export interface RemoteRuntimeSummary {
  id: string;
  tag: string;
  backend: string;
  display_name: string;
}

export interface RemoteDraftCandidate {
  id: string;
  display_name: string;
  path: string;
  quantization: string | null;
  size_bytes: number;
}

export interface RemoteModelInspectResponse {
  model: LocalModelEntry;
  saved_preset: LoadConfig | null;
  active_load_config: LoadConfig | null;
  runtimes: RemoteRuntimeSummary[];
  active_runtime_id: string | null;
  default_offload_kv_cache: boolean;
  guardrail_level: GuardrailLevel;
  cpu_logical_cores: number;
  live_memory: LiveMemoryContext;
  draft_models: RemoteDraftCandidate[];
}

export interface ExternalModelSource {
  app_name: string;
  path: string;
  kind: 'gguf_dir' | 'ollama' | string;
}

export interface ActiveRemoteModel {
  peer: RemotePeer;
  model: RemoteModelSummary;
  qualifiedId: string;
}

export interface AppSettings {
  models_dirs: string[];
  auto_discover_external_models: boolean;
  runtimes_dir: string;
  logs_dir: string;
  hf_token: string | null;
  github_token: string | null;
  active_runtime_id: string | null;
  auto_start_server: boolean;
  minimize_to_tray: boolean;
  start_minimized: boolean;
  auto_check_runtime_updates: boolean;
  old_runtimes_to_keep: number;
  auto_check_app_updates: boolean;
  log_retention_days: number;
  guardrail_level: GuardrailLevel;
  limit_offload_to_dedicated_vram: boolean;
  default_offload_kv_cache: boolean;
  disabled_gpu_ids: number[];
  server: ServerConfig;
  link: LinkConfig;
  default_sampling: SamplingParams;
}

export interface BufferAllocation {
  backend: string;
  is_gpu: boolean;
  size_mib: number;
}

export interface MemoryComponentBreakdown {
  dense_vram_mib: number;
  dense_ram_mib: number;
  moe_experts_vram_mib: number;
  moe_experts_ram_mib: number;
  mtp_vram_mib: number;
  mtp_ram_mib: number;
  vision_vram_mib: number;
  vision_ram_mib: number;
  kv_cache_vram_mib: number;
  kv_cache_ram_mib: number;
  compute_vram_mib: number;
  compute_ram_mib: number;
  source: string;
}

export interface MemoryPlacementReport {
  offloaded_layers: number | null;
  total_layers: number | null;
  cpu_moe_layers: number | null;
  model_buffers: BufferAllocation[];
  vision_buffers: BufferAllocation[];
  mtp_buffers: BufferAllocation[];
  kv_buffers: BufferAllocation[];
  compute_buffers: BufferAllocation[];
  tensor_overrides_applied: string[];
  wddm_shared_spillover_detected: boolean;
  shared_gpu_mem_mib: number | null;
  process_vram_mib: number | null;
  process_ram_mib: number | null;
  estimated_breakdown: MemoryComponentBreakdown | null;
}

export interface TensorMemorySummary {
  non_layer_hot_bytes: number;
  per_layer_hot_bytes: Record<string, number>;
  per_layer_routed_expert_bytes: Record<string, number>;
  per_layer_mtp_bytes: Record<string, number>;
  non_layer_mtp_bytes: number;
  total_tensor_bytes: number;
  total_parameters: number;
}

export interface GgufMetadata {
  path: string;
  shard_paths: string[];
  gguf_version: number;
  total_size_bytes: number;
  architecture: string | null;
  name: string | null;
  size_label: string | null;
  file_type: number | null;
  quantization: string | null;
  parameter_count: number | null;
  block_count: number | null;
  context_length: number | null;
  embedding_length: number | null;
  head_count: number | null;
  head_count_kv: number | null;
  head_count_kv_per_layer?: number[];
  key_length: number | null;
  value_length: number | null;
  key_length_swa?: number | null;
  value_length_swa?: number | null;
  sliding_window?: number | null;
  sliding_window_pattern_interval?: number | null;
  sliding_window_pattern_layers?: boolean[];
  shared_kv_layers?: number | null;
  kv_lora_rank?: number | null;
  key_length_mla?: number | null;
  value_length_mla?: number | null;
  rope_dimension_count?: number | null;
  rope_freq_base: number | null;
  expert_count: number | null;
  expert_used_count: number | null;
  is_moe: boolean;
  full_attention_interval: number | null;
  ssm_state_size: number | null;
  ssm_conv_kernel: number | null;
  ssm_inner_size: number | null;
  ssm_group_count: number | null;
  ssm_time_step_rank: number | null;
  nextn_predict_layers: number | null;
  has_mtp: boolean;
  chat_template: string | null;
  supports_reasoning: boolean;
  supports_reasoning_effort: boolean;
  supports_enable_thinking: boolean;
  supports_tools: boolean;
  is_embedding: boolean;
  tokenizer_model?: string | null;
  tokenizer_pre?: string | null;
  vocab_size?: number | null;
  tensor_count: number;
  tensor_summary: TensorMemorySummary;
}

export type DraftCompatibilityLevel = 'compatible' | 'warning' | 'incompatible';

export interface DraftCompatibilityReport {
  level: DraftCompatibilityLevel;
  compatible: boolean;
  reasons: string[];
}

export interface DiscoveredCliFlag {
  flag: string;
  aliases: string[];
  value_hint: string | null;
  description: string;
}

export interface LlamaSlotInfo {
  id: number;
  is_processing: boolean;
  n_ctx: number;
  n_past: number | null;
  n_decoded: number | null;
  id_task: number | null;
}

export interface LlamaServerMetrics {
  port: number;
  slots_total: number;
  slots_processing: number;
  kv_cache_usage_ratio: number | null;
  kv_cache_tokens: number | null;
  prompt_tokens_total: number | null;
  predicted_tokens_total: number | null;
  prompt_tok_s: number | null;
  predicted_tok_s: number | null;
  requests_processing: number | null;
  requests_deferred: number | null;
  slots: LlamaSlotInfo[];
}

export interface MemoryEstimate {
  weights_vram_mib: number;
  weights_ram_mib: number;
  kv_cache_vram_mib: number;
  kv_cache_ram_mib: number;
  compute_vram_mib: number;
  compute_ram_mib: number;
  total_vram_mib: number;
  total_ram_mib: number;
  vram_utilization_pct: number;
  ram_utilization_pct: number;
  fit_verdict: FitVerdict;
  recommended_n_cpu_moe: number | null;
  guardrail_warning: string | null;
  breakdown: MemoryComponentBreakdown;
  vision_available_mib: number;
  mtp_available_mib: number;
  hardware_vram_total_mib: number;
  current_vram_used_mib: number;
  available_vram_mib: number;
  hardware_ram_total_mib: number;
  current_ram_used_mib: number;
  available_ram_mib: number;
  vram_sample_count: number;
}

export interface LiveMemoryBudget {
  vram_total_mib: number;
  vram_used_avg_mib: number;
  vram_used_current_mib: number;
  vram_available_mib: number;
  ram_total_mib: number;
  ram_used_avg_mib: number;
  ram_available_mib: number;
  sample_count: number;
}

export interface AppUpdateInfo {
  current_version: string;
  latest_version: string;
  update_available: boolean;
  release_name: string;
  release_notes: string;
  published_at: string;
  html_url: string;
  asset_name: string | null;
  asset_url: string | null;
  asset_size_bytes: number;
  sha256: string | null;
}

export interface AppUpdateProgress {
  stage: 'downloading' | 'verifying' | 'installing' | 'error' | string;
  downloaded_bytes: number;
  total_bytes: number;
  percent: number;
  message: string;
}

export interface LocalModelEntry {
  id: string;
  display_name: string;
  path: string;
  source: string | null;
  shard_count: number;
  mmproj_path: string | null;
  mmproj_size_bytes: number | null;
  size_bytes: number;
  architecture: string | null;
  quantization: string | null;
  parameter_count: number | null;
  is_moe: boolean;
  expert_count: number | null;
  expert_used_count: number | null;
  routed_expert_layers: number;
  has_mtp: boolean;
  mtp_layer_count: number;
  mtp_bytes: number;
  layer_count: number | null;
  head_count: number | null;
  head_count_kv: number | null;
  max_context_length: number | null;
  supports_reasoning: boolean;
  supports_reasoning_effort: boolean;
  supports_enable_thinking: boolean;
  supports_tools: boolean;
  supports_vision: boolean;
  is_embedding: boolean;
  has_chat_template: boolean;
  chat_template_preview: string | null;
  gguf: GgufMetadata | null;
}

export interface LoadedModelInfo {
  model_id: string;
  display_name: string;
  alias: string | null;
  runtime_id: string;
  runtime_tag: string;
  port: number;
  pid: number | null;
  is_jit: boolean;
  loaded_at: string;
  last_used_at: string;
  idle_seconds: number;
  active_requests: number;
  placement: MemoryPlacementReport;
  placement_status: PlacementStatus;
  breakdown: MemoryComponentBreakdown;
  load_config: LoadConfig;
  cli_args: string[];
}

export interface CpuInfo {
  name: string;
  arch: string;
  physical_cores: number;
  logical_cores: number;
  has_avx: boolean;
  has_avx2: boolean;
  has_avx512: boolean;
  has_fma: boolean;
}

export interface GpuDeviceInfo {
  device_id: number;
  name: string;
  vendor: string;
  backend: string;
  vram_total_bytes: number;
  shared_mem_total_bytes: number;
  driver_version: string | null;
  cuda_version: string | null;
  cuda_version_major: number | null;
  cuda_version_minor: number | null;
}

export interface SystemHardwareProfile {
  cpu: CpuInfo;
  ram_total_bytes: number;
  gpus: GpuDeviceInfo[];
  total_vram_bytes: number;
  recommended_backend: string;
  os_name: string;
}

export interface TelemetrySnapshot {
  ram_used_bytes: number;
  ram_total_bytes: number;
  vram_used_bytes: number;
  vram_total_bytes: number;
  shared_gpu_mem_used_bytes: number;
  cpu_utilization_pct: number;
  gpu_utilization_pct: number | null;
  gpu_temp_celsius: number | null;
  wddm_spillover_warning: boolean;
}

export interface GatewayStatus {
  running: boolean;
  bind_address: string;
  reachable_url: string;
  config: ServerConfig;
}

export interface ServerLogEntry {
  timestamp: string;
  level: string;
  source: string;
  message: string;
}

export interface ApiTokenRecord {
  id: string;
  name: string;
  token: string;
  created_at: string;
  last_used_at: string | null;
  enabled: boolean;
}

export interface NamedPresetRecord {
  id: string;
  name: string;
  kind: string;
  payload_json: string;
  updated_at: string;
}

export interface InstalledRuntimeRecord {
  id: string;
  tag: string;
  backend: string;
  display_name: string;
  install_dir: string;
  server_exe: string;
  bench_exe: string | null;
  is_custom: boolean;
  installed_at: string;
  release_notes: string | null;
  size_bytes: number;
}

export interface ReleaseAssetInfo {
  name: string;
  download_url: string;
  size_bytes: number;
}

export interface RuntimeBackendVariant {
  id: string;
  tag: string;
  backend: string;
  display_name: string;
  description: string;
  cuda_version: string | null;
  assets: ReleaseAssetInfo[];
  total_size_bytes: number;
  compatible: boolean;
  recommended: boolean;
}

export interface LlamaRelease {
  tag_name: string;
  name: string;
  published_at: string;
  html_url: string;
  release_notes: string;
  variants: RuntimeBackendVariant[];
}

export type HfSortOrder =
  | 'best_match'
  | 'most_downloads'
  | 'most_likes'
  | 'recently_updated';

export interface ModelCapabilities {
  vision: boolean;
  tool_use: boolean;
  reasoning: boolean;
  moe: boolean;
  embedding: boolean;
}

export interface HfModelSummary {
  repo_id: string;
  author: string;
  name: string;
  downloads: number;
  likes: number;
  last_modified: string;
  tags: string[];
  gated: boolean;
  capabilities: ModelCapabilities;
  architecture: string | null;
  param_count_label: string | null;
  context_length: number | null;
}

export interface HfDownloadFile {
  path: string;
  file_name: string;
  url: string;
  size_bytes: number;
  sha256: string | null;
}

export interface HfQuantOption {
  id: string;
  label: string;
  quantization: string;
  total_size_bytes: number;
  shard_count: number;
  files: HfDownloadFile[];
  fit_verdict: FitVerdict;
}

export interface HfModelDetail {
  summary: HfModelSummary;
  readme_markdown: string;
  quant_options: HfQuantOption[];
  mmproj_files: HfDownloadFile[];
}

export interface DownloadJobState {
  id: string;
  kind: string;
  title: string;
  repo_id: string | null;
  dest_dir: string;
  files: HfDownloadFile[];
  current_file: string | null;
  total_bytes: number;
  downloaded_bytes: number;
  speed_bytes_per_sec: number;
  eta_seconds: number | null;
  status: string;
  error: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface BenchmarkRunRecord {
  id: string;
  created_at: string;
  model_id: string;
  runtime_tag: string;
  kind: string;
  prompt_preset: string;
  prompt_tokens: number;
  reasoning_tokens: number;
  output_tokens: number;
  ttft_ms: number;
  prompt_eval_tok_s: number;
  generation_tok_s: number;
  peak_vram_mib: number;
  peak_ram_mib: number;
  placement: MemoryPlacementReport;
  load_config: LoadConfig;
  notes: string | null;
}

export interface QuickTestRequest {
  prompt_preset: string;
  custom_prompt: string | null;
  max_tokens: number;
  reasoning_effort: string | null;
  sampling: SamplingParams | null;
}

export interface QuickTestStreamChunk {
  run_id: string;
  reasoning_delta: string;
  content_delta: string;
  elapsed_ms: number;
  approx_tokens: number;
}

export interface QuickTestResult {
  record: BenchmarkRunRecord;
  reasoning_text: string;
  output_text: string;
  total_duration_ms: number;
  prompt_eval_ms: number;
  generation_ms: number;
}

export interface PlaygroundMessage {
  role: 'system' | 'user' | 'assistant' | 'tool' | string;
  content: unknown;
  tool_call_id?: string | null;
  tool_calls?: Record<string, unknown>[] | null;
}

export interface PlaygroundTurnRequest {
  messages: PlaygroundMessage[];
  max_tokens: number;
  reasoning_effort: string | null;
  sampling: SamplingParams | null;
  tools_json: string | null;
}

export interface PlaygroundStreamChunk {
  turn_id: string;
  reasoning_delta: string;
  content_delta: string;
  elapsed_ms: number;
  approx_tokens: number;
}

export interface PlaygroundTurnResult {
  turn_id: string;
  reasoning_text: string;
  output_text: string;
  tool_calls: Record<string, unknown>[];
  finish_reason: string | null;
  prompt_tokens: number;
  cached_tokens: number;
  completion_tokens: number;
  ttft_ms: number;
  prompt_eval_tok_s: number;
  generation_tok_s: number;
  total_duration_ms: number;
}

export interface LlamaBenchRequest {
  model_path: string;
  model_id: string;
  runtime_tag: string;
  prompt_sizes: number[];
  gen_sizes: number[];
  n_gpu_layers: number[];
  flash_attn: boolean[];
  cache_types_k: string[];
  cache_types_v: string[];
  threads: number | null;
  batch_size: number | null;
  ubatch_size: number | null;
  repetitions: number;
  extra_args: string[];
}

export interface LlamaBenchRow {
  model_type: string;
  model_size: number;
  n_prompt: number;
  n_gen: number;
  n_gpu_layers: number;
  flash_attn: boolean;
  type_k: string;
  type_v: string;
  avg_ts: number;
  stddev_ts: number;
  avg_ns: number;
  test_label: string;
}
