import { invoke } from '@tauri-apps/api/core';
import type {
  ApiTokenRecord,
  AppSettings,
  AppUpdateInfo,
  BenchmarkRunRecord,
  DiscoveredCliFlag,
  DownloadJobState,
  DraftCompatibilityReport,
  ExternalModelSource,
  GatewayStatus,
  GgufMetadata,
  GuardrailLevel,
  HfDownloadFile,
  HfModelDetail,
  HfModelSummary,
  HfSortOrder,
  InstalledRuntimeRecord,
  LinkStatus,
  LiveMemoryBudget,
  LiveMemoryContext,
  LlamaBenchRequest,
  LlamaBenchRow,
  LlamaRelease,
  LlamaServerMetrics,
  LoadConfig,
  LoadedModelInfo,
  LocalModelEntry,
  MemoryComponentBreakdown,
  MemoryEstimate,
  MemoryPlacementReport,
  MoeOffloadMode,
  NamedPresetRecord,
  PlaygroundTurnRequest,
  PlaygroundTurnResult,
  QuickTestRequest,
  QuickTestResult,
  RemoteActionResponse,
  RemoteInstallRuntimeResponse,
  RemoteLoadResponse,
  RemoteModelInspectResponse,
  RemotePeer,
  RemoteRuntimesResponse,
  RuntimeBackendVariant,
  ServerLogEntry,
  SystemHardwareProfile,
  TelemetrySnapshot,
} from './types';

export const api = {
  // Hardware & Telemetry
  getHardwareProfile: () => invoke<SystemHardwareProfile>('get_hardware_profile'),
  refreshHardwareProfile: () => invoke<SystemHardwareProfile>('refresh_hardware_profile'),
  getTelemetrySnapshot: () => invoke<TelemetrySnapshot>('get_telemetry_snapshot'),
  formatHardwareSummary: () => invoke<string>('format_hardware_summary'),
  resetVramUsageAverage: () => invoke<LiveMemoryBudget>('reset_vram_usage_average'),

  // App Auto-Updater
  getAppVersion: () => invoke<string>('get_app_version'),
  checkAppUpdate: () => invoke<AppUpdateInfo>('check_app_update'),
  installAppUpdate: (updateInfo: AppUpdateInfo) =>
    invoke<void>('install_app_update', { updateInfo }),

  // Settings & Presets
  getSettings: () => invoke<AppSettings>('get_settings'),
  saveSettings: (settings: AppSettings) => invoke<AppSettings>('save_settings', { settings }),
  getModelPreset: (modelId: string) => invoke<LoadConfig | null>('get_model_preset', { modelId }),
  listNamedPresets: (kind?: string) =>
    invoke<NamedPresetRecord[]>('list_named_presets', { kind: kind ?? null }),
  saveNamedPreset: (id: string | null, name: string, kind: string, payloadJson: string) =>
    invoke<NamedPresetRecord>('save_named_preset', { id, name, kind, payloadJson }),
  deleteNamedPreset: (id: string) => invoke<void>('delete_named_preset', { id }),

  // Local Library & Estimator
  scanLocalLibrary: () => invoke<LocalModelEntry[]>('scan_local_library'),
  listDiscoveredModelSources: () => invoke<ExternalModelSource[]>('list_discovered_model_sources'),
  checkDraftModelCompatibility: (mainPath: string, draftPath: string) =>
    invoke<DraftCompatibilityReport>('check_draft_model_compatibility', {
      mainPath,
      draftPath,
    }),
  estimateModelMemory: (gguf: GgufMetadata, config: LoadConfig, replacingModelId?: string) =>
    invoke<MemoryEstimate>('estimate_model_memory', {
      gguf,
      config,
      replacingModelId: replacingModelId ?? null,
    }),
  estimateModelMemoryWithContext: (
    gguf: GgufMetadata,
    config: LoadConfig,
    liveCtx: LiveMemoryContext,
    guardrail: GuardrailLevel,
    mmprojSizeBytes?: number | null,
    draftSizeBytes?: number | null,
  ) =>
    invoke<MemoryEstimate>('estimate_model_memory_with_context', {
      gguf,
      config,
      liveCtx,
      guardrail,
      mmprojSizeBytes: mmprojSizeBytes ?? null,
      draftSizeBytes: draftSizeBytes ?? null,
    }),
  computeAutoFitMoe: (gguf: GgufMetadata, config: LoadConfig, replacingModelId?: string) =>
    invoke<MoeOffloadMode>('compute_auto_fit_moe', {
      gguf,
      config,
      replacingModelId: replacingModelId ?? null,
    }),
  computeAutoFitMoeWithContext: (
    gguf: GgufMetadata,
    config: LoadConfig,
    liveCtx: LiveMemoryContext,
    mmprojSizeBytes?: number | null,
  ) =>
    invoke<MoeOffloadMode>('compute_auto_fit_moe_with_context', {
      gguf,
      config,
      liveCtx,
      mmprojSizeBytes: mmprojSizeBytes ?? null,
    }),
  previewLlamaCli: (config: LoadConfig) => invoke<string>('preview_llama_cli', { config }),
  formatLlamaCliArgs: (config: LoadConfig) => invoke<string>('format_llama_cli_args', { config }),
  deleteLocalModelFiles: (path: string) => invoke<number>('delete_local_model_files', { path }),
  revealInExplorer: (path: string) => invoke<void>('reveal_in_explorer', { path }),

  // HuggingFace Search & Downloads
  searchHfModels: (query: string, sort: HfSortOrder, extraFilter?: string) =>
    invoke<HfModelSummary[]>('search_hf_models', {
      query,
      sort,
      extraFilter: extraFilter ?? null,
    }),
  getHfModelDetail: (
    repoId: string,
    targetVramBytes?: number | null,
    targetRamBytes?: number | null,
  ) =>
    invoke<HfModelDetail>('get_hf_model_detail_cmd', {
      repoId,
      targetVramBytes: targetVramBytes ?? null,
      targetRamBytes: targetRamBytes ?? null,
    }),
  enqueueModelDownload: (title: string, repoId: string, files: HfDownloadFile[]) =>
    invoke<DownloadJobState>('enqueue_model_download', { title, repoId, files }),
  listDownloads: () => invoke<DownloadJobState[]>('list_downloads'),
  pauseDownload: (id: string) => invoke<boolean>('pause_download', { id }),
  resumeDownload: (id: string) => invoke<boolean>('resume_download', { id }),
  cancelDownload: (id: string) => invoke<boolean>('cancel_download', { id }),
  clearCompletedDownloads: () => invoke<void>('clear_completed_downloads'),

  // Runtimes
  fetchGithubReleases: () => invoke<LlamaRelease[]>('fetch_github_releases'),
  listInstalledRuntimes: () => invoke<InstalledRuntimeRecord[]>('list_installed_runtimes'),
  installRuntime: (variant: RuntimeBackendVariant, releaseNotes: string | null) =>
    invoke<InstalledRuntimeRecord>('install_runtime', { variant, releaseNotes }),
  registerCustomRuntime: (displayName: string, path: string) =>
    invoke<InstalledRuntimeRecord>('register_custom_runtime_cmd', { displayName, path }),
  uninstallRuntime: (id: string) => invoke<void>('uninstall_runtime_cmd', { id }),
  setActiveRuntime: (runtimeId: string) =>
    invoke<AppSettings>('set_active_runtime', { runtimeId }),
  discoverRuntimeFlags: (runtimeId?: string | null) =>
    invoke<DiscoveredCliFlag[]>('discover_runtime_flags', { runtimeId: runtimeId ?? null }),
  getLoadedModelMetrics: (modelId: string) =>
    invoke<LlamaServerMetrics>('get_loaded_model_metrics', { modelId }),

  // Model Loading & Supervisor
  loadModel: (
    modelId: string,
    displayName: string,
    config: LoadConfig,
    rememberPreset: boolean,
    forceOverride?: boolean,
  ) =>
    invoke<LoadedModelInfo>('load_model_cmd', {
      modelId,
      displayName,
      config,
      rememberPreset,
      forceOverride: forceOverride ?? false,
    }),
  readTextFilePreview: (path: string) => invoke<string>('read_text_file_preview', { path }),
  unloadModel: (modelId: string) => invoke<boolean>('unload_model_cmd', { modelId }),
  listLoadedModels: () => invoke<LoadedModelInfo[]>('list_loaded_models'),
  getServerLogs: () => invoke<ServerLogEntry[]>('get_server_logs'),
  clearServerLogs: () => invoke<void>('clear_server_logs'),

  // Server & Tokens
  getServerStatus: () => invoke<GatewayStatus>('get_server_status'),
  startServer: () => invoke<GatewayStatus>('start_server'),
  stopServer: () => invoke<GatewayStatus>('stop_server'),
  listApiTokens: () => invoke<ApiTokenRecord[]>('list_api_tokens'),
  createApiToken: (name: string) => invoke<ApiTokenRecord>('create_api_token', { name }),
  deleteApiToken: (id: string) => invoke<void>('delete_api_token', { id }),
  setApiTokenEnabled: (id: string, enabled: boolean) =>
    invoke<void>('set_api_token_enabled', { id, enabled }),

  // Benchmarks
  runQuickBenchmark: (modelId: string, request: QuickTestRequest) =>
    invoke<QuickTestResult>('run_quick_benchmark', { modelId, request }),
  runPlaygroundTurn: (modelId: string, request: PlaygroundTurnRequest) =>
    invoke<PlaygroundTurnResult>('run_playground_turn_cmd', { modelId, request }),
  runLlamaBenchMatrix: (runtimeId: string | null, request: LlamaBenchRequest) =>
    invoke<LlamaBenchRow[]>('run_llama_bench_matrix', { runtimeId, request }),
  listBenchmarkHistory: (limit = 50) =>
    invoke<BenchmarkRunRecord[]>('list_benchmark_history', { limit }),
  deleteBenchmarkRun: (id: string) => invoke<void>('delete_benchmark_run', { id }),
  exportBenchmarksCsv: () => invoke<string>('export_benchmarks_csv'),

  // BibuLM Link (BYO Tailscale Mesh)
  getLinkStatus: () => invoke<LinkStatus>('get_link_status'),
  linkConnect: () => invoke<LinkStatus>('link_connect'),
  linkDisconnect: () => invoke<LinkStatus>('link_disconnect'),
  linkLogout: () => invoke<LinkStatus>('link_logout'),
  listLinkPeers: () => invoke<RemotePeer[]>('list_link_peers'),
  refreshLinkPeers: () => invoke<RemotePeer[]>('refresh_link_peers'),
  remoteInspectModel: (peerId: string, modelId: string) =>
    invoke<RemoteModelInspectResponse>('remote_inspect_model', { peerId, modelId }),
  remoteLoadModel: (
    peerId: string,
    modelId: string,
    config?: LoadConfig | null,
    rememberPreset?: boolean,
  ) =>
    invoke<RemoteLoadResponse>('remote_load_model', {
      peerId,
      modelId,
      config: config ?? null,
      rememberPreset: rememberPreset ?? false,
    }),
  remoteUnloadModel: (peerId: string, modelId: string) =>
    invoke<RemoteLoadResponse>('remote_unload_model', { peerId, modelId }),
  remoteListRuntimes: (peerId: string) =>
    invoke<RemoteRuntimesResponse>('remote_list_runtimes', { peerId }),
  remoteFetchRuntimeReleases: (peerId: string) =>
    invoke<LlamaRelease[]>('remote_fetch_runtime_releases', { peerId }),
  remoteInstallRuntime: (
    peerId: string,
    variant: RuntimeBackendVariant,
    releaseNotes: string | null,
  ) =>
    invoke<RemoteInstallRuntimeResponse>('remote_install_runtime', {
      peerId,
      variant,
      releaseNotes,
    }),
  remoteSetActiveRuntime: (peerId: string, runtimeId: string) =>
    invoke<RemoteActionResponse>('remote_set_active_runtime', { peerId, runtimeId }),
  remoteUninstallRuntime: (peerId: string, runtimeId: string) =>
    invoke<RemoteActionResponse>('remote_uninstall_runtime', { peerId, runtimeId }),
  remoteEnqueueModelDownload: (
    peerId: string,
    title: string,
    repoId: string,
    files: HfDownloadFile[],
  ) =>
    invoke<DownloadJobState>('remote_enqueue_model_download', {
      peerId,
      title,
      repoId,
      files,
    }),
  remoteListDownloads: (peerId: string) =>
    invoke<DownloadJobState[]>('remote_list_downloads', { peerId }),
  remotePauseDownload: (peerId: string, id: string) =>
    invoke<boolean>('remote_pause_download', { peerId, id }),
  remoteResumeDownload: (peerId: string, id: string) =>
    invoke<boolean>('remote_resume_download', { peerId, id }),
  remoteCancelDownload: (peerId: string, id: string) =>
    invoke<boolean>('remote_cancel_download', { peerId, id }),
  remoteClearCompletedDownloads: (peerId: string) =>
    invoke<void>('remote_clear_completed_downloads', { peerId }),
  remoteDeleteModel: (peerId: string, modelId: string) =>
    invoke<RemoteActionResponse>('remote_delete_model', { peerId, modelId }),
  remoteGetLogs: (peerId: string) =>
    invoke<ServerLogEntry[]>('remote_get_logs', { peerId }),
};

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const idx = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const val = bytes / Math.pow(1024, idx);
  return `${val >= 100 ? val.toFixed(0) : val.toFixed(2)} ${units[idx]}`;
}

export function formatMiB(mib: number): string {
  if (!Number.isFinite(mib) || mib <= 0) return '0 MiB';
  if (mib >= 1024) return `${(mib / 1024).toFixed(2)} GB`;
  return `${mib.toFixed(0)} MiB`;
}

export function formatParams(n: number | null): string {
  if (!n || n <= 0) return '—';
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  return `${n}`;
}

export function formatEta(secs: number | null): string {
  if (secs == null || !Number.isFinite(secs)) return '—';
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function formatRelativeDate(iso: string): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const diffMs = Date.now() - d.getTime();
  const days = Math.floor(diffMs / 86400000);
  if (days <= 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 30) return `${days}d ago`;
  return d.toLocaleDateString();
}

export function sumBuffers(
  list: { is_gpu: boolean; size_mib: number }[],
  gpu: boolean,
): number {
  return list.filter((b) => b.is_gpu === gpu).reduce((acc, b) => acc + b.size_mib, 0);
}

export function breakdownTotals(b: MemoryComponentBreakdown) {
  const gpuWeights =
    b.dense_vram_mib + b.moe_experts_vram_mib + b.mtp_vram_mib + b.vision_vram_mib;
  const cpuWeights = b.dense_ram_mib + b.moe_experts_ram_mib + b.mtp_ram_mib + b.vision_ram_mib;
  return {
    gpuModel: gpuWeights,
    cpuModel: cpuWeights,
    gpuKv: b.kv_cache_vram_mib,
    cpuKv: b.kv_cache_ram_mib,
    gpuCompute: b.compute_vram_mib,
    cpuCompute: b.compute_ram_mib,
    totalGpu: gpuWeights + b.kv_cache_vram_mib + b.compute_vram_mib,
    totalCpu: cpuWeights + b.kv_cache_ram_mib + b.compute_ram_mib,
  };
}

export function placementTotals(
  p: MemoryPlacementReport,
  breakdown?: MemoryComponentBreakdown | null,
) {
  if (breakdown) {
    const bt = breakdownTotals(breakdown);
    if (bt.totalGpu > 0 || bt.totalCpu > 0) {
      return bt;
    }
  }
  const gpuModel =
    sumBuffers(p.model_buffers ?? [], true) +
    sumBuffers(p.vision_buffers ?? [], true) +
    sumBuffers(p.mtp_buffers ?? [], true);
  const cpuModel =
    sumBuffers(p.model_buffers ?? [], false) +
    sumBuffers(p.vision_buffers ?? [], false) +
    sumBuffers(p.mtp_buffers ?? [], false);
  const gpuKv = sumBuffers(p.kv_buffers ?? [], true);
  const cpuKv = sumBuffers(p.kv_buffers ?? [], false);
  const gpuCompute = sumBuffers(p.compute_buffers ?? [], true);
  const cpuCompute = sumBuffers(p.compute_buffers ?? [], false);
  const totalGpu = gpuModel + gpuKv + gpuCompute;
  const totalCpu = cpuModel + cpuKv + cpuCompute;
  if (totalGpu > 0 || totalCpu > 0) {
    return {
      gpuModel,
      cpuModel,
      gpuKv,
      cpuKv,
      gpuCompute,
      cpuCompute,
      totalGpu,
      totalCpu,
    };
  }
  if (p.estimated_breakdown) {
    return breakdownTotals(p.estimated_breakdown);
  }
  return {
    gpuModel: p.process_vram_mib ?? 0,
    cpuModel: p.process_ram_mib ?? 0,
    gpuKv: 0,
    cpuKv: 0,
    gpuCompute: 0,
    cpuCompute: 0,
    totalGpu: p.process_vram_mib ?? 0,
    totalCpu: p.process_ram_mib ?? 0,
  };
}

export function defaultLoadConfigForModel(
  model: LocalModelEntry,
  settings?: AppSettings | null,
): LoadConfig {
  const maxCtx = model.max_context_length ?? 32768;
  return {
    model_path: model.path,
    model_alias: model.id,
    runtime_id: null,
    enable_vision: Boolean(model.mmproj_path),
    mmproj_path: model.mmproj_path,
    offload_mmproj: true,
    ctx_size: Math.min(16384, maxCtx),
    batch_size: 2048,
    ubatch_size: 512,
    parallel_slots: 1,
    unified_kv_cache: true,
    cache_ram_mib: null,
    context_shift: null,
    gpu_layers: 999,
    moe_offload: { mode: 'full_gpu' },
    auto_fit: null,
    fit_ctx: null,
    fit_target_mib: null,
    threads: null,
    threads_batch: null,
    devices: [],
    split_mode: null,
    main_gpu: null,
    tensor_split: null,
    offload_kv_cache: settings?.default_offload_kv_cache ?? true,
    flash_attention: 'on',
    cache_type_k: 'q8_0',
    cache_type_v: 'q8_0',
    use_mmap: true,
    use_mlock: false,
    direct_io: false,
    rope_scaling: null,
    rope_freq_base: null,
    rope_freq_scale: null,
    yarn_ext_factor: null,
    yarn_attn_factor: null,
    yarn_beta_fast: null,
    yarn_beta_slow: null,
    jinja: true,
    chat_template_override: null,
    chat_template_file: null,
    reasoning: {
      effort: 'default',
      format: 'auto',
      budget: null,
      budget_message: null,
      enable_thinking: null,
    },
    mtp_enabled: false,
    mtp_spec_type: null,
    draft_model_path: null,
    draft_gpu_layers: null,
    draft_max: model.has_mtp ? 3 : null,
    draft_min: null,
    draft_p_min: null,
    seed: null,
    embeddings: model.is_embedding,
    cli_flags: {},
    extra_args: [],
    custom_cli_args: null,
    sampling_defaults: null,
  };
}
