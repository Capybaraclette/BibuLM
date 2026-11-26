import React, { useEffect, useState } from 'react';
import { open } from '@tauri-apps/plugin-dialog';
import {
  Check,
  ChevronLeft,
  Copy,
  Cpu,
  Eye,
  EyeOff,
  FolderOpen,
  Link2,
  Pencil,
  RotateCcw,
  Terminal,
} from 'lucide-react';
import type {
  AppSettings,
  DiscoveredCliFlag,
  FlashAttentionMode,
  InstalledRuntimeRecord,
  KvQuantType,
  LoadConfig,
  LocalModelEntry,
  MemoryEstimate,
  RemoteModelInspectResponse,
  RemotePeer,
  RopeScalingType,
  SplitMode,
} from '../types';
import { api, defaultLoadConfigForModel, formatBytes, formatMiB } from '../api';
import { Toggle } from './Toggle';
import { LlamaFlagsCatalogPanel } from './LlamaFlagsCatalogPanel';
import {
  FloatField,
  KV_QUANT_OPTIONS,
  NumberField,
  fileNameOnly,
} from './load-config/FieldPrimitives';
import { MemoryPreviewCard } from './load-config/MemoryPreviewCard';
import { MoeOffloadSection, TensorOverridesEditor } from './load-config/MoeOffloadSection';
import { SpeculativeSection } from './load-config/SpeculativeSection';
import { ReasoningSection } from './load-config/ReasoningSection';

interface LoadConfigModalProps {
  model: LocalModelEntry;
  localModels: LocalModelEntry[];
  runtimes: InstalledRuntimeRecord[];
  settings: AppSettings;
  cpuLogicalCores: number;
  remoteTarget?: {
    peer: RemotePeer;
    inspect: RemoteModelInspectResponse;
    onRefreshInspect: () => Promise<RemoteModelInspectResponse>;
  };
  onCancel: () => void;
  onConfirmLoad: (config: LoadConfig, rememberPreset: boolean, forceOverride?: boolean) => void;
}

export const LoadConfigModal: React.FC<LoadConfigModalProps> = ({
  model,
  localModels,
  runtimes,
  settings,
  cpuLogicalCores,
  remoteTarget,
  onCancel,
  onConfirmLoad,
}) => {
  const [remoteInspect, setRemoteInspect] = useState<RemoteModelInspectResponse | null>(
    remoteTarget?.inspect ?? null,
  );

  const [config, setConfig] = useState<LoadConfig>(() => {
    if (remoteTarget) {
      const defaults = defaultLoadConfigForModel(model, {
        ...settings,
        default_offload_kv_cache: remoteTarget.inspect.default_offload_kv_cache,
      });
      const base =
        remoteTarget.inspect.active_load_config ?? remoteTarget.inspect.saved_preset;
      if (base) {
        return {
          ...defaults,
          ...base,
          model_path: model.path,
          mmproj_path: base.mmproj_path ?? defaults.mmproj_path,
          enable_vision:
            typeof base.enable_vision === 'boolean'
              ? base.enable_vision
              : Boolean(base.mmproj_path ?? defaults.mmproj_path),
          cli_flags: { ...(base.cli_flags ?? {}) },
        };
      }
      return defaults;
    }
    return defaultLoadConfigForModel(model, settings);
  });
  const [rememberPreset, setRememberPreset] = useState(true);
  const [forceOverride, setForceOverride] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [estimate, setEstimate] = useState<MemoryEstimate | null>(null);
  const [vramRefreshTick, setVramRefreshTick] = useState(0);
  const [resettingVramAvg, setResettingVramAvg] = useState(false);
  const [cliPreview, setCliPreview] = useState('');
  const [copiedCli, setCopiedCli] = useState(false);
  const [discoveredFlags, setDiscoveredFlags] = useState<DiscoveredCliFlag[]>([]);
  const [extraArgsText, setExtraArgsText] = useState(() =>
    (
      (remoteTarget?.inspect.active_load_config ?? remoteTarget?.inspect.saved_preset)
        ?.extra_args ?? []
    ).join(' '),
  );

  useEffect(() => {
    if (remoteTarget?.inspect) {
      setRemoteInspect(remoteTarget.inspect);
    }
  }, [remoteTarget?.inspect]);

  // Load any previously saved per-model preset from SQLite (local models only)
  useEffect(() => {
    if (remoteTarget) return;
    let active = true;
    api.getModelPreset(model.id).then((saved) => {
      if (!active || !saved) return;
      const defaults = defaultLoadConfigForModel(model, settings);
      const merged: LoadConfig = {
        ...defaults,
        ...saved,
        model_path: model.path,
        mmproj_path: saved.mmproj_path ?? defaults.mmproj_path,
        enable_vision:
          typeof saved.enable_vision === 'boolean'
            ? saved.enable_vision
            : Boolean(saved.mmproj_path ?? defaults.mmproj_path),
        cli_flags: { ...(saved.cli_flags ?? {}) },
      };
      setConfig(merged);
      setExtraArgsText((merged.extra_args ?? []).join(' '));
    });
    return () => {
      active = false;
    };
  }, [model, settings, remoteTarget]);

  // Discover supported flags from selected runtime when advanced panel is expanded
  useEffect(() => {
    if (!showAdvanced || remoteTarget) return;
    let cancelled = false;
    api
      .discoverRuntimeFlags(config.runtime_id)
      .then((flags) => {
        if (!cancelled) setDiscoveredFlags(flags);
      })
      .catch(() => {
        if (!cancelled) setDiscoveredFlags([]);
      });
    return () => {
      cancelled = true;
    };
  }, [showAdvanced, config.runtime_id, remoteTarget]);

  // Live Memory Estimation + Live CLI Preview on every config change
  useEffect(() => {
    let cancelled = false;
    if (model.gguf) {
      if (remoteInspect) {
        const draftSize =
          remoteInspect.draft_models.find((d) => d.path === config.draft_model_path)
            ?.size_bytes ?? null;
        api
          .estimateModelMemoryWithContext(
            model.gguf,
            config,
            remoteInspect.live_memory,
            remoteInspect.guardrail_level,
            model.mmproj_size_bytes,
            draftSize,
          )
          .then((est) => {
            if (!cancelled) setEstimate(est);
          });
      } else {
        api.estimateModelMemory(model.gguf, config, model.id).then((est) => {
          if (!cancelled) setEstimate(est);
        });
      }
    }
    api.previewLlamaCli(config).then((cmd) => {
      if (!cancelled) setCliPreview(cmd);
    });
    return () => {
      cancelled = true;
    };
  }, [model.gguf, model.id, model.mmproj_size_bytes, config, vramRefreshTick, remoteInspect]);

  const maxCtx = model.max_context_length ?? 131072;
  const maxLayers = (model.layer_count ?? 64) + 1;
  const maxMoeLayers = Math.max(model.routed_expert_layers, model.layer_count ?? 48);

  const effectiveGuardrail = remoteInspect?.guardrail_level ?? settings.guardrail_level;
  const vramExceedsAvail = Boolean(
    settings.limit_offload_to_dedicated_vram &&
      (effectiveGuardrail === 'balanced' || effectiveGuardrail === 'strict') &&
      estimate &&
      estimate.available_vram_mib > 0 &&
      estimate.total_vram_mib > estimate.available_vram_mib * 1.02,
  );
  const blockedByGuardrail = Boolean(
    !remoteTarget &&
      !forceOverride &&
      estimate &&
      ((effectiveGuardrail === 'strict' && estimate.guardrail_warning) || vramExceedsAvail),
  );

  // Keyboard shortcuts: Esc to cancel, Ctrl+Enter to load
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel();
      if (e.ctrlKey && e.key === 'Enter' && !blockedByGuardrail) {
        e.preventDefault();
        onConfirmLoad(config, rememberPreset, forceOverride);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [config, rememberPreset, forceOverride, blockedByGuardrail, onCancel, onConfirmLoad]);

  const patch = (p: Partial<LoadConfig>) => setConfig((prev) => ({ ...prev, ...p }));

  const effectiveCpuCores = remoteInspect?.cpu_logical_cores ?? cpuLogicalCores;
  const effectiveRuntimes = remoteInspect ? remoteInspect.runtimes : runtimes;
  const effectiveDraftModels = remoteInspect
    ? remoteInspect.draft_models
    : localModels
        .filter((m) => m.path !== model.path)
        .map((m) => ({
          id: m.id,
          display_name: m.display_name,
          path: m.path,
          quantization: m.quantization,
          size_bytes: m.size_bytes,
        }));

  const handleAutoFitMoe = async () => {
    if (!model.gguf) return;
    const mode = remoteInspect
      ? await api.computeAutoFitMoeWithContext(
          model.gguf,
          config,
          remoteInspect.live_memory,
          model.mmproj_size_bytes,
        )
      : await api.computeAutoFitMoe(model.gguf, config, model.id);
    patch({
      gpu_layers: 999,
      moe_offload: mode,
      auto_fit: mode.mode === 'full_gpu' ? null : false,
    });
  };

  const handleResetVramAvg = async () => {
    setResettingVramAvg(true);
    try {
      if (remoteTarget) {
        const refreshed = await remoteTarget.onRefreshInspect();
        setRemoteInspect(refreshed);
      } else {
        await api.resetVramUsageAverage();
      }
      setVramRefreshTick((t) => t + 1);
    } finally {
      setResettingVramAvg(false);
    }
  };

  const handleBrowseMmproj = async () => {
    const selected = await open({
      multiple: false,
      title: 'Select Vision Projector (mmproj*.gguf)',
      filters: [{ name: 'GGUF Projector', extensions: ['gguf'] }],
    });
    if (typeof selected === 'string' && selected.trim()) {
      patch({
        mmproj_path: selected,
        enable_vision: true,
      });
    }
  };

  const copyCli = () => {
    navigator.clipboard.writeText(cliPreview);
    setCopiedCli(true);
    setTimeout(() => setCopiedCli(false), 1500);
  };

  const ctxPresets = [4096, 8192, 16384, 32768, 65536, 131072].filter((c) => c <= maxCtx);
  const hasMmproj = Boolean(config.mmproj_path || model.mmproj_path);

  return (
    <div className="apple-modal-backdrop">
      <div className="w-[840px] max-w-full max-h-[92vh] apple-sheet flex flex-col">
        {/* Top Title */}
        <div className="px-5 py-3.5 apple-glass border-b flex items-center justify-between">
          <button
            type="button"
            onClick={onCancel}
            className="apple-btn-ghost w-8 h-8"
            aria-label="Close configuration"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <div className="text-center min-w-0 px-3">
            <div className="flex items-center justify-center gap-2 flex-wrap">
              <h2 className="text-[14px] font-semibold text-[var(--label-primary)] truncate tracking-tight">
                {model.display_name}
              </h2>
              {remoteTarget && (
                <span className="apple-badge apple-badge-accent text-[10px]">
                  <Link2 className="w-3 h-3" />
                  Remote · {remoteTarget.peer.device_name}
                </span>
              )}
            </div>
            <p className="text-[11px] text-[var(--label-secondary)] mt-0.5">
              {model.architecture ?? 'GGUF'} • {model.quantization ?? 'Unknown Quant'}
              {model.is_moe &&
                ` • MoE (${model.expert_count ?? '?'}E / ${model.expert_used_count ?? '?'}A)`}
              {model.has_mtp && ` • MTP (${model.mtp_layer_count || 1}L)`}
              {model.supports_vision && ' • Vision'}
              {remoteTarget?.peer.gpu_summary ? ` • ${remoteTarget.peer.gpu_summary}` : ''}
            </p>
          </div>
          <div className="w-8" />
        </div>

        {/* Sticky Live Estimated Memory Breakdown Header */}
        <MemoryPreviewCard
          estimate={estimate}
          config={config}
          guardrailLevel={effectiveGuardrail}
          limitDedicatedVram={settings.limit_offload_to_dedicated_vram}
          forceOverride={forceOverride}
          onForceOverrideChange={setForceOverride}
          resettingVramAvg={resettingVramAvg}
          isRemote={Boolean(remoteTarget)}
          onResetVramAvg={handleResetVramAvg}
        />

        {/* Scrollable Config Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4 text-[12px] bg-[var(--bg-surface)]">
          {model.is_moe && (
            <MoeOffloadSection
              config={config}
              maxLayers={maxLayers}
              maxMoeLayers={maxMoeLayers}
              recommendedNCpuMoe={estimate?.recommended_n_cpu_moe ?? null}
              onPatch={patch}
              onAutoFitMoe={handleAutoFitMoe}
            />
          )}

          {/* Context Length & GPU Offload */}
          <div className="space-y-4">
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <div>
                  <span className="font-semibold text-[var(--label-primary)]">Context Length</span>
                  <span className="ml-2 text-[11px] text-[var(--label-tertiary)]">
                    Model supports up to{' '}
                    <code className="text-[var(--label-secondary)]">{maxCtx}</code> tokens
                  </span>
                </div>
                <input
                  type="number"
                  min={512}
                  max={maxCtx}
                  step={512}
                  value={config.ctx_size}
                  onChange={(e) =>
                    patch({ ctx_size: Math.max(512, parseInt(e.target.value || '8192', 10)) })
                  }
                  className="w-28 bibulm-input h-7 text-right font-mono tabular-nums"
                />
              </div>
              <input
                type="range"
                min={512}
                max={maxCtx}
                step={512}
                value={config.ctx_size}
                onChange={(e) => patch({ ctx_size: parseInt(e.target.value, 10) })}
                className="w-full"
              />
              <div className="flex flex-wrap gap-1.5">
                {ctxPresets.map((c) => (
                  <button
                    key={c}
                    type="button"
                    onClick={() => patch({ ctx_size: c })}
                    className={`px-2 py-0.5 rounded-md text-[11px] font-mono tabular-nums transition-colors cursor-pointer ${
                      config.ctx_size === c
                        ? 'bg-[var(--accent)] text-white font-medium'
                        : 'bg-[var(--bg-control)] text-[var(--label-secondary)] hover:text-[var(--label-primary)]'
                    }`}
                  >
                    {c >= 1024 ? `${c / 1024}K` : c}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-[var(--label-primary)]">
                  GPU Offload Layers (-ngl)
                </span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => patch({ gpu_layers: 999 })}
                    className="bibulm-btn-secondary h-7 px-2.5 text-[11px]"
                  >
                    All ({maxLayers})
                  </button>
                  <input
                    type="number"
                    min={0}
                    max={999}
                    value={config.gpu_layers ?? 999}
                    onChange={(e) =>
                      patch({ gpu_layers: Math.max(0, parseInt(e.target.value || '0', 10)) })
                    }
                    className="w-20 bibulm-input h-7 text-right font-mono tabular-nums"
                  />
                </div>
              </div>
              <input
                type="range"
                min={0}
                max={maxLayers}
                value={Math.min(config.gpu_layers ?? maxLayers, maxLayers)}
                onChange={(e) => {
                  const val = parseInt(e.target.value, 10);
                  patch({ gpu_layers: val >= maxLayers ? 999 : val });
                }}
                className="w-full"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <NumberField
                label="CPU Thread Pool Size (-t)"
                value={config.threads}
                placeholder={`Auto (${Math.max(1, Math.floor(effectiveCpuCores / 2))})`}
                onChange={(v) => patch({ threads: v })}
              />
              <NumberField
                label="Batch Threads (-tb)"
                value={config.threads_batch}
                placeholder={`Auto (${effectiveCpuCores})`}
                onChange={(v) => patch({ threads_batch: v })}
              />
              <NumberField
                label="Evaluation Batch Size (-b)"
                value={config.batch_size}
                onChange={(v) => patch({ batch_size: v ?? 2048 })}
              />
              <NumberField
                label="Physical Micro-Batch (-ub)"
                value={config.ubatch_size}
                onChange={(v) => patch({ ubatch_size: v ?? 512 })}
              />
              <NumberField
                label="Max Concurrent Slots (-np)"
                value={config.parallel_slots}
                onChange={(v) => patch({ parallel_slots: Math.max(1, v ?? 1) })}
              />
              <NumberField
                label="Prompt Cache RAM MiB (--cache-ram)"
                value={config.cache_ram_mib}
                placeholder="Default (8192)"
                onChange={(v) => patch({ cache_ram_mib: v })}
              />
            </div>
          </div>

          {/* KV Cache & Flash Attention */}
          <div className="apple-card-inset p-4 space-y-3">
            <div className="font-semibold text-[var(--label-primary)]">
              KV Cache &amp; Flash Attention
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="flex items-center justify-between">
                <span className="text-[var(--label-secondary)]">Offload KV Cache to GPU</span>
                <Toggle
                  checked={config.offload_kv_cache}
                  onChange={(v) => patch({ offload_kv_cache: v })}
                />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[var(--label-secondary)]">Unified KV Cache (-kvu)</span>
                <Toggle
                  checked={config.unified_kv_cache}
                  onChange={(v) => patch({ unified_kv_cache: v })}
                />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[var(--label-secondary)]">Flash Attention (-fa)</span>
                <select
                  value={config.flash_attention}
                  onChange={(e) =>
                    patch({ flash_attention: e.target.value as FlashAttentionMode })
                  }
                  className="apple-select h-7"
                >
                  <option value="auto">Auto</option>
                  <option value="on">On</option>
                  <option value="off">Off</option>
                </select>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[var(--label-secondary)]">
                  Context Shift (--context-shift)
                </span>
                <select
                  value={
                    config.context_shift === null ? 'auto' : config.context_shift ? 'on' : 'off'
                  }
                  onChange={(e) =>
                    patch({
                      context_shift: e.target.value === 'auto' ? null : e.target.value === 'on',
                    })
                  }
                  className="apple-select h-7"
                >
                  <option value="auto">Default</option>
                  <option value="on">On (--context-shift)</option>
                  <option value="off">Off (--no-context-shift)</option>
                </select>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[var(--label-secondary)]">Auto-Fit (--fit)</span>
                <select
                  value={config.auto_fit === null ? 'auto' : config.auto_fit ? 'on' : 'off'}
                  onChange={(e) =>
                    patch({
                      auto_fit: e.target.value === 'auto' ? null : e.target.value === 'on',
                    })
                  }
                  className="apple-select h-7"
                >
                  <option value="auto">Smart Default</option>
                  <option value="on">Force On</option>
                  <option value="off">Force Off</option>
                </select>
              </div>
              {config.auto_fit === true && (
                <>
                  <NumberField
                    label="Fit Min Context (--fit-ctx)"
                    value={config.fit_ctx}
                    placeholder="4096"
                    onChange={(v) => patch({ fit_ctx: v })}
                  />
                  <NumberField
                    label="Fit Target Margin MiB (--fit-target)"
                    value={config.fit_target_mib}
                    placeholder="1024"
                    onChange={(v) => patch({ fit_target_mib: v })}
                  />
                </>
              )}
              <div className="flex items-center justify-between">
                <span className="text-[var(--label-secondary)]">K Cache Quant (-ctk)</span>
                <select
                  value={config.cache_type_k ?? ''}
                  onChange={(e) =>
                    patch({
                      cache_type_k: (e.target.value || null) as KvQuantType | null,
                    })
                  }
                  className="apple-select h-7"
                >
                  {KV_QUANT_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[var(--label-secondary)]">V Cache Quant (-ctv)</span>
                <select
                  value={config.cache_type_v ?? ''}
                  onChange={(e) =>
                    patch({
                      cache_type_v: (e.target.value || null) as KvQuantType | null,
                    })
                  }
                  className="apple-select h-7"
                >
                  {KV_QUANT_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Vision Projector */}
          <div
            className={`rounded-xl border p-4 space-y-3 ${
              hasMmproj && config.enable_vision
                ? 'bg-[var(--signal-amber-soft)] border-[var(--signal-amber-border)]'
                : 'apple-card-inset'
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 min-w-0">
                {config.enable_vision && hasMmproj ? (
                  <Eye className="w-4 h-4 text-[var(--signal-amber)] shrink-0" />
                ) : (
                  <EyeOff className="w-4 h-4 text-[var(--label-tertiary)] shrink-0" />
                )}
                <div className="min-w-0">
                  <div className="font-semibold text-[var(--label-primary)] flex items-center gap-2 flex-wrap">
                    <span>Vision Projector (mmproj)</span>
                    {!config.enable_vision && (estimate?.vision_available_mib ?? 0) > 0.5 && (
                      <span className="apple-badge apple-badge-green font-mono text-[10px]">
                        Disabled · saves {formatMiB(estimate!.vision_available_mib)}
                      </span>
                    )}
                    {config.enable_vision && hasMmproj && model.mmproj_size_bytes && (
                      <span className="apple-badge apple-badge-amber font-mono text-[10px]">
                        {formatBytes(model.mmproj_size_bytes)}
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] text-[var(--label-secondary)] truncate">
                    {config.mmproj_path
                      ? fileNameOnly(config.mmproj_path)
                      : 'No paired mmproj file detected — disable to save VRAM or browse to attach one'}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2 shrink-0">
                {!remoteTarget && (
                  <button
                    type="button"
                    onClick={handleBrowseMmproj}
                    className="bibulm-btn-secondary h-7 px-2.5 text-[11px]"
                    title="Select custom mmproj GGUF file"
                  >
                    <FolderOpen className="w-3 h-3" />
                    <span>Browse</span>
                  </button>
                )}
                <Toggle
                  checked={config.enable_vision && Boolean(config.mmproj_path)}
                  onChange={(v) => {
                    if (v) {
                      patch({
                        enable_vision: true,
                        mmproj_path: config.mmproj_path ?? model.mmproj_path,
                      });
                    } else {
                      patch({ enable_vision: false });
                    }
                  }}
                />
              </div>
            </div>

            {config.enable_vision && config.mmproj_path && (
              <div className="flex items-center justify-between pt-2 border-t border-[var(--border-hairline)]">
                <span className="text-[var(--label-secondary)]">
                  Offload Vision Projector to GPU (off adds <code>--no-mmproj-offload</code>)
                </span>
                <Toggle
                  checked={config.offload_mmproj}
                  onChange={(v) => patch({ offload_mmproj: v })}
                />
              </div>
            )}
          </div>

          <SpeculativeSection
            model={model}
            config={config}
            draftModels={effectiveDraftModels}
            isRemote={Boolean(remoteTarget)}
            onPatch={patch}
          />

          <ReasoningSection
            model={model}
            config={config}
            isRemote={Boolean(remoteTarget)}
            onPatch={patch}
          />

          {/* Advanced Settings Section */}
          {showAdvanced && (
            <div className="apple-card-inset p-4 space-y-4">
              <div className="flex items-center gap-2 font-semibold text-[var(--label-primary)]">
                <Cpu className="w-4 h-4 text-[var(--accent-text)]" />
                <span>Advanced Engine, Memory, RoPE &amp; Raw CLI Controls</span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[var(--label-secondary)]">Runtime Override</span>
                  <select
                    value={config.runtime_id ?? ''}
                    onChange={(e) => patch({ runtime_id: e.target.value || null })}
                    className="w-44 apple-select h-7"
                  >
                    <option value="">Active Default Runtime</option>
                    {effectiveRuntimes.map((r) => (
                      <option key={r.id} value={r.id}>
                        {r.tag} ({r.backend})
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[var(--label-secondary)]">Model Alias (--alias)</span>
                  <input
                    value={config.model_alias ?? ''}
                    onChange={(e) => patch({ model_alias: e.target.value || null })}
                    placeholder={model.id}
                    className="w-44 bibulm-input h-7 font-mono"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[var(--label-secondary)]">Try mmap()</span>
                  <Toggle checked={config.use_mmap} onChange={(v) => patch({ use_mmap: v })} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[var(--label-secondary)]">
                    Keep Model in RAM (--mlock)
                  </span>
                  <Toggle checked={config.use_mlock} onChange={(v) => patch({ use_mlock: v })} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[var(--label-secondary)]">Direct I/O (--direct-io)</span>
                  <Toggle checked={config.direct_io} onChange={(v) => patch({ direct_io: v })} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[var(--label-secondary)]">
                    Embeddings Mode (--embeddings)
                  </span>
                  <Toggle checked={config.embeddings} onChange={(v) => patch({ embeddings: v })} />
                </div>
              </div>

              {/* RoPE Scaling & Seed */}
              <div className="grid grid-cols-2 gap-3 pt-2 border-t border-[var(--border-hairline)]">
                <div className="flex items-center justify-between">
                  <span className="text-[var(--label-secondary)]">RoPE Scaling</span>
                  <select
                    value={config.rope_scaling ?? ''}
                    onChange={(e) =>
                      patch({
                        rope_scaling: (e.target.value || null) as RopeScalingType | null,
                      })
                    }
                    className="apple-select h-7"
                  >
                    <option value="">Auto</option>
                    <option value="none">None</option>
                    <option value="linear">Linear</option>
                    <option value="yarn">YaRN</option>
                  </select>
                </div>
                <NumberField
                  label="Seed (-s)"
                  value={config.seed}
                  placeholder="Random"
                  onChange={(v) => patch({ seed: v })}
                />
                <FloatField
                  label="RoPE Freq Base (--rope-freq-base)"
                  value={config.rope_freq_base}
                  placeholder="Auto"
                  onChange={(v) => patch({ rope_freq_base: v })}
                />
                <FloatField
                  label="RoPE Freq Scale (--rope-freq-scale)"
                  value={config.rope_freq_scale}
                  placeholder="Auto"
                  onChange={(v) => patch({ rope_freq_scale: v })}
                />
                <FloatField
                  label="YaRN Ext Factor (--yarn-ext-factor)"
                  value={config.yarn_ext_factor}
                  placeholder="-1.0"
                  onChange={(v) => patch({ yarn_ext_factor: v })}
                />
                <FloatField
                  label="YaRN Attn Factor (--yarn-attn-factor)"
                  value={config.yarn_attn_factor}
                  placeholder="-1.0"
                  onChange={(v) => patch({ yarn_attn_factor: v })}
                />
                <FloatField
                  label="YaRN Beta Fast (--yarn-beta-fast)"
                  value={config.yarn_beta_fast}
                  placeholder="-1.0"
                  onChange={(v) => patch({ yarn_beta_fast: v })}
                />
                <FloatField
                  label="YaRN Beta Slow (--yarn-beta-slow)"
                  value={config.yarn_beta_slow}
                  placeholder="-1.0"
                  onChange={(v) => patch({ yarn_beta_slow: v })}
                />
              </div>

              {/* Multi-GPU & Devices */}
              <div className="grid grid-cols-2 gap-3 pt-2 border-t border-[var(--border-hairline)]">
                <div className="flex items-center justify-between">
                  <span className="text-[var(--label-secondary)]">Multi-GPU Split Mode (-sm)</span>
                  <select
                    value={config.split_mode ?? ''}
                    onChange={(e) =>
                      patch({ split_mode: (e.target.value || null) as SplitMode | null })
                    }
                    className="apple-select h-7"
                  >
                    <option value="">Default (layer)</option>
                    <option value="none">None (Single GPU)</option>
                    <option value="layer">Layer</option>
                    <option value="row">Row</option>
                    <option value="tensor">Tensor (Experimental)</option>
                  </select>
                </div>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[var(--label-secondary)]">Tensor Split (-ts)</span>
                  <input
                    value={config.tensor_split ?? ''}
                    onChange={(e) => patch({ tensor_split: e.target.value || null })}
                    placeholder="e.g. 3,1"
                    className="w-28 bibulm-input h-7 font-mono"
                  />
                </div>
                <NumberField
                  label="Main GPU Index (-mg)"
                  value={config.main_gpu}
                  placeholder="0"
                  onChange={(v) => patch({ main_gpu: v })}
                />
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[var(--label-secondary)]">Offload Devices (-dev)</span>
                  <input
                    value={(config.devices ?? []).join(',')}
                    onChange={(e) =>
                      patch({
                        devices: e.target.value
                          .split(',')
                          .map((s) => s.trim())
                          .filter(Boolean),
                      })
                    }
                    placeholder="e.g. CUDA0,CUDA1 or none"
                    className="w-40 bibulm-input h-7 font-mono"
                  />
                </div>
              </div>

              {!model.is_moe && (
                <div className="space-y-2 pt-2 border-t border-[var(--border-hairline)]">
                  <div className="flex items-center justify-between">
                    <span className="text-[var(--label-primary)] font-medium">
                      Custom Tensor Overrides (-ot)
                    </span>
                    {config.moe_offload.mode !== 'custom_overrides' ? (
                      <button
                        type="button"
                        onClick={() =>
                          patch({
                            moe_offload: {
                              mode: 'custom_overrides',
                              value: [{ pattern: 'blk\\.\\d+\\.ffn_.*_exps\\.weight', buffer: 'CPU' }],
                            },
                            gpu_layers: maxLayers,
                          })
                        }
                        className="text-[11px] font-medium text-[var(--accent-text)] hover:underline cursor-pointer"
                      >
                        + Add Override Rule
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={() => patch({ moe_offload: { mode: 'full_gpu' } })}
                        className="text-[11px] text-[var(--label-secondary)] hover:text-[var(--signal-red)] cursor-pointer"
                      >
                        Clear Overrides
                      </button>
                    )}
                  </div>
                  {config.moe_offload.mode === 'custom_overrides' && (
                    <TensorOverridesEditor
                      rules={config.moe_offload.value}
                      onChange={(next) =>
                        patch({ moe_offload: { mode: 'custom_overrides', value: next } })
                      }
                    />
                  )}
                </div>
              )}

              <LlamaFlagsCatalogPanel
                cliFlags={config.cli_flags ?? {}}
                discoveredFlags={discoveredFlags}
                onChange={(next) => patch({ cli_flags: next })}
              />

              <div className="space-y-1 pt-2 border-t border-[var(--border-hairline)]">
                <span className="text-[var(--label-secondary)]">
                  Extra Raw CLI Arguments (Day-1 Escape Hatch)
                </span>
                <input
                  value={extraArgsText}
                  onChange={(e) => {
                    setExtraArgsText(e.target.value);
                    patch({
                      extra_args: e.target.value.trim() ? [e.target.value.trim()] : [],
                    });
                  }}
                  placeholder="e.g. --swa-full --no-warmup --prio 2"
                  className="w-full bibulm-input h-7 font-mono"
                />
              </div>
            </div>
          )}

          {/* Live CLI Command Preview / Manual CLI Override */}
          <div className="rounded-xl bg-[var(--bg-terminal)] border border-[var(--border-hairline)] p-3 space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-1.5 text-[11px] font-medium text-[#a1a1a6]">
                <Terminal className="w-3.5 h-3.5 text-[#409cff]" />
                <span>
                  {config.custom_cli_args != null
                    ? 'Manual llama-server Launch Arguments'
                    : 'Generated llama-server Command Preview'}
                </span>
                {config.custom_cli_args != null && (
                  <span className="apple-badge apple-badge-amber text-[10px] py-0 px-1.5">
                    Manual Override
                  </span>
                )}
              </div>
              <div className="flex items-center gap-3">
                {!remoteTarget &&
                  (config.custom_cli_args != null ? (
                    <button
                      type="button"
                      onClick={() => patch({ custom_cli_args: null })}
                      className="inline-flex items-center gap-1 text-[11px] text-[#ff9f0a] hover:text-white cursor-pointer"
                      title="Discard manual edits and return to auto-generated arguments"
                    >
                      <RotateCcw className="w-3 h-3" />
                      <span>Reset to auto</span>
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={async () => {
                        try {
                          const rawArgs = await api.formatLlamaCliArgs(config);
                          patch({ custom_cli_args: rawArgs });
                        } catch {
                          patch({ custom_cli_args: '' });
                        }
                      }}
                      className="inline-flex items-center gap-1 text-[11px] text-[#409cff] hover:text-white cursor-pointer"
                      title="Manually edit all llama-server CLI arguments for this model"
                    >
                      <Pencil className="w-3 h-3" />
                      <span>Edit manually</span>
                    </button>
                  ))}
                <button
                  type="button"
                  onClick={copyCli}
                  className="inline-flex items-center gap-1 text-[11px] text-[#a1a1a6] hover:text-white cursor-pointer"
                >
                  {copiedCli ? (
                    <>
                      <Check className="w-3 h-3 text-[#30d158]" />
                      <span className="text-[#30d158]">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      <span>Copy CLI</span>
                    </>
                  )}
                </button>
              </div>
            </div>
            {config.custom_cli_args != null ? (
              <div className="space-y-1.5">
                <textarea
                  value={config.custom_cli_args}
                  onChange={(e) => patch({ custom_cli_args: e.target.value })}
                  rows={4}
                  spellCheck={false}
                  placeholder="-m path/to/model.gguf -c 16384 -ngl 999 -ctk q8_0 -ctv q8_0 --metrics"
                  className="w-full rounded-lg bg-black/30 border border-white/15 px-2.5 py-2 text-[11px] font-mono text-[#e5e5ea] placeholder-[#636366] focus:outline-none focus:border-[#0a84ff] resize-y"
                  aria-label="Manual llama-server launch arguments"
                />
                <div className="text-[10px] text-[#8e8e93]">
                  Manual arguments override the generated flags above (
                  <code className="text-[#a1a1a6]">--host</code> and{' '}
                  <code className="text-[#a1a1a6]">--port</code> remain managed by BibuLM).
                </div>
              </div>
            ) : (
              <pre className="text-[11px] font-mono text-[#e5e5ea] whitespace-pre-wrap break-all max-h-24 overflow-y-auto">
                {cliPreview}
              </pre>
            )}
          </div>
        </div>

        {/* Footer Actions */}
        <div className="px-5 py-3.5 bg-[var(--bg-surface-inset)] border-t border-[var(--border-hairline)] flex items-center justify-between gap-4">
          <div className="space-y-1.5">
            <label className="flex items-center gap-2 text-[12px] text-[var(--label-primary)] cursor-pointer">
              <input
                type="checkbox"
                checked={rememberPreset}
                onChange={(e) => setRememberPreset(e.target.checked)}
                className="rounded accent-[var(--accent)]"
              />
              <span>
                {remoteTarget
                  ? `Remember settings on ${remoteTarget.peer.device_name} for `
                  : 'Remember settings for '}
                <code className="apple-badge font-mono text-[11px] py-0 px-1.5">{model.id}</code>
              </span>
            </label>
            <div className="flex items-center gap-2 text-[12px] text-[var(--label-secondary)]">
              <Toggle size="sm" checked={showAdvanced} onChange={setShowAdvanced} />
              <span>Show advanced settings &amp; all llama.cpp flags</span>
              {Object.keys(config.cli_flags ?? {}).length > 0 && (
                <span className="apple-badge apple-badge-accent font-mono text-[10px]">
                  {Object.keys(config.cli_flags ?? {}).length} flag overrides
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button type="button" onClick={onCancel} className="bibulm-btn-secondary h-9 px-4">
              Cancel
            </button>
            <button
              type="button"
              disabled={blockedByGuardrail}
              onClick={() => onConfirmLoad(config, rememberPreset, forceOverride)}
              className="bibulm-btn-primary h-9 px-4 font-semibold"
            >
              <span>
                {remoteTarget
                  ? `${remoteInspect?.active_load_config ? 'Reload' : 'Load'} on ${remoteTarget.peer.device_name}`
                  : 'Load Model'}
              </span>
              <span className="text-[10px] opacity-85 font-mono">⌃↵</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
