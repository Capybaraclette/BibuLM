import React, { useState } from 'react';
import { AlertTriangle, Check, Copy, Cpu, RefreshCw, ShieldAlert } from 'lucide-react';
import type {
  AppSettings,
  GuardrailLevel,
  SystemHardwareProfile,
  TelemetrySnapshot,
} from '../types';
import { api, formatBytes } from '../api';
import { Toggle } from '../components/Toggle';
import { PageHeader } from '../components/PageHeader';

interface HardwareViewProps {
  profile: SystemHardwareProfile | null;
  telemetry: TelemetrySnapshot | null;
  settings: AppSettings;
  onSaveSettings: (next: AppSettings) => void;
  onRefreshProfile: () => void;
}

const GUARDRAILS: { id: GuardrailLevel; label: string; desc: string }[] = [
  {
    id: 'off',
    label: 'OFF (Not Recommended)',
    desc: 'No precautions against VRAM/RAM overload',
  },
  {
    id: 'relaxed',
    label: 'Relaxed',
    desc: 'Mild precautions — warns above 97% VRAM or 92% RAM',
  },
  {
    id: 'balanced',
    label: 'Balanced',
    desc: 'Moderate precautions — warns above 90% VRAM or 85% RAM',
  },
  {
    id: 'strict',
    label: 'Strict',
    desc: 'Strict safety headroom — warns above 82% VRAM or 75% RAM',
  },
];

export const HardwareView: React.FC<HardwareViewProps> = ({
  profile,
  telemetry,
  settings,
  onSaveSettings,
  onRefreshProfile,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopyInfo = async () => {
    const text = await api.formatHardwareSummary();
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const toggleGpu = (deviceId: number, enabled: boolean) => {
    const next = enabled
      ? settings.disabled_gpu_ids.filter((id) => id !== deviceId)
      : [...settings.disabled_gpu_ids, deviceId];
    onSaveSettings({ ...settings, disabled_gpu_ids: next });
  };

  const vramPct =
    telemetry && telemetry.vram_total_bytes > 0
      ? (telemetry.vram_used_bytes / telemetry.vram_total_bytes) * 100
      : 0;
  const ramPct =
    telemetry && telemetry.ram_total_bytes > 0
      ? (telemetry.ram_used_bytes / telemetry.ram_total_bytes) * 100
      : 0;

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <PageHeader
        icon={<Cpu className="w-4 h-4" />}
        title="Hardware & Telemetry"
        subtitle="Live NVIDIA NVML & Windows DXGI telemetry, GPU enablement, and memory guardrails."
        actions={
          <>
            <button onClick={onRefreshProfile} className="bibulm-btn-secondary">
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Refresh</span>
            </button>
            <button onClick={handleCopyInfo} className="bibulm-btn-secondary">
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-[var(--signal-green)]" />
                  <span className="text-[var(--signal-green)]">Copied Info</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy Info</span>
                </>
              )}
            </button>
          </>
        }
      />

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-5xl mx-auto space-y-5">
      {/* CPU & Memory Capacity Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bibulm-card p-4 space-y-3 text-[12px]">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-[var(--label-primary)] flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-[var(--accent-text)]" /> CPU
            </span>
            <span className="apple-badge apple-badge-green text-[11px] font-medium">
              ✓ Compatible
            </span>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between gap-2">
              <span className="text-[var(--label-secondary)]">Name</span>
              <span className="text-[var(--label-primary)] font-medium text-right">
                {profile?.cpu.name ?? 'Detecting…'}
              </span>
            </div>
            <div className="flex justify-between gap-2">
              <span className="text-[var(--label-secondary)]">Cores</span>
              <span className="text-[var(--label-primary)] font-mono tabular-nums">
                {profile
                  ? `${profile.cpu.physical_cores} physical / ${profile.cpu.logical_cores} logical`
                  : '—'}
              </span>
            </div>
            <div className="flex justify-between items-center gap-2">
              <span className="text-[var(--label-secondary)]">Architecture</span>
              <div className="flex flex-wrap gap-1 justify-end">
                {profile && (
                  <>
                    <IsaPill label={profile.cpu.arch} />
                    {profile.cpu.has_avx && <IsaPill label="AVX" />}
                    {profile.cpu.has_avx2 && <IsaPill label="AVX2" />}
                    {profile.cpu.has_avx512 && <IsaPill label="AVX-512" />}
                    {profile.cpu.has_fma && <IsaPill label="FMA" />}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="bibulm-card p-4 space-y-3 text-[12px]">
          <div className="font-semibold text-[var(--label-primary)]">Memory Capacity</div>
          <div className="space-y-2.5">
            <div className="flex justify-between items-center">
              <span className="text-[var(--label-secondary)]">System RAM</span>
              <span className="text-[14px] font-mono font-semibold text-[var(--label-primary)] tabular-nums">
                {formatBytes(profile?.ram_total_bytes ?? 0)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[var(--label-secondary)]">Dedicated VRAM</span>
              <span className="text-[14px] font-mono font-semibold text-[var(--signal-green)] tabular-nums">
                {formatBytes(profile?.total_vram_bytes ?? 0)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[var(--label-secondary)]">Recommended Backend</span>
              <span className="apple-badge apple-badge-accent font-mono uppercase text-[11px]">
                {profile?.recommended_backend ?? 'cpu'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* GPUs Section */}
      <div className="bibulm-card p-4 space-y-4 text-[12px]">
        <div className="font-semibold text-[var(--label-primary)]">
          GPUs ({profile?.gpus.length ?? 0} detected)
        </div>

        {!profile || profile.gpus.length === 0 ? (
          <div className="text-[var(--label-tertiary)] py-3">
            No discrete GPU detected — inference will use the CPU backend.
          </div>
        ) : (
          <div className="space-y-2.5">
            {profile.gpus.map((gpu) => {
              const enabled = !settings.disabled_gpu_ids.includes(gpu.device_id);
              return (
                <div
                  key={gpu.device_id}
                  className="apple-card-inset p-3.5 flex items-center justify-between gap-3"
                >
                  <div className="space-y-1">
                    <div className="text-[14px] font-semibold text-[var(--label-primary)]">
                      {gpu.name}
                    </div>
                    <div className="text-[11px] text-[var(--label-secondary)] font-mono tabular-nums">
                      VRAM: {formatBytes(gpu.vram_total_bytes)} • {gpu.backend}
                      {gpu.cuda_version && ` ${gpu.cuda_version}`}
                      {gpu.driver_version && ` • Driver ${gpu.driver_version}`} • deviceId:{' '}
                      {gpu.device_id}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] font-mono text-[var(--label-secondary)]">
                      {enabled ? 'ON' : 'OFF'}
                    </span>
                    <Toggle checked={enabled} onChange={(v) => toggleGpu(gpu.device_id, v)} />
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <div className="space-y-3 pt-3 border-t border-[var(--border-hairline)]">
          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="font-medium text-[var(--label-primary)]">
                Limit Model Offload to Dedicated GPU Memory
              </div>
              <div className="text-[11px] text-[var(--label-secondary)]">
                ON: Prevents silent Windows WDDM VRAM-to-RAM spillover warnings when estimating fit.
              </div>
            </div>
            <Toggle
              checked={settings.limit_offload_to_dedicated_vram}
              onChange={(v) =>
                onSaveSettings({ ...settings, limit_offload_to_dedicated_vram: v })
              }
            />
          </div>

          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="font-medium text-[var(--label-primary)]">
                Offload KV Cache to GPU Memory
              </div>
              <div className="text-[11px] text-[var(--label-secondary)]">
                Default KV cache placement for newly loaded models.
              </div>
            </div>
            <Toggle
              checked={settings.default_offload_kv_cache}
              onChange={(v) => onSaveSettings({ ...settings, default_offload_kv_cache: v })}
            />
          </div>
        </div>
      </div>

      {/* Live Resource Monitor */}
      <div className="bibulm-card p-4 space-y-4 text-[12px]">
        <div className="flex items-center justify-between">
          <span className="font-semibold text-[var(--label-primary)]">Live Resource Monitor</span>
          {telemetry?.wddm_spillover_warning && (
            <span className="apple-badge apple-badge-red text-[11px] font-medium">
              <AlertTriangle className="w-3.5 h-3.5" /> WDDM Shared Memory Spillover Detected
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <GaugeCard
            title="Dedicated VRAM"
            value={
              telemetry
                ? `${formatBytes(telemetry.vram_used_bytes)} / ${formatBytes(telemetry.vram_total_bytes)}`
                : '—'
            }
            pct={vramPct}
            barCls={vramPct > 92 ? 'bg-[var(--signal-red-fill)]' : 'bg-[var(--signal-green-fill)]'}
            sub={
              telemetry
                ? `GPU Util: ${telemetry.gpu_utilization_pct?.toFixed(0) ?? '—'}% · Temp: ${telemetry.gpu_temp_celsius ?? '—'}°C`
                : ''
            }
          />
          <GaugeCard
            title="Windows Shared GPU Memory"
            value={telemetry ? formatBytes(telemetry.shared_gpu_mem_used_bytes) : '—'}
            pct={
              telemetry && telemetry.ram_total_bytes > 0
                ? Math.min(100, (telemetry.shared_gpu_mem_used_bytes / (telemetry.ram_total_bytes / 2)) * 100)
                : 0
            }
            barCls={
              telemetry?.wddm_spillover_warning
                ? 'bg-[var(--signal-red-fill)]'
                : 'bg-[var(--signal-amber-fill)]'
            }
            sub="Spillover detector (should stay near 0 GB)"
          />
          <GaugeCard
            title="System RAM & CPU"
            value={
              telemetry
                ? `${formatBytes(telemetry.ram_used_bytes)} / ${formatBytes(telemetry.ram_total_bytes)}`
                : '—'
            }
            pct={ramPct}
            barCls="bg-[var(--accent)]"
            sub={telemetry ? `CPU Load: ${telemetry.cpu_utilization_pct.toFixed(1)}%` : ''}
          />
        </div>
      </div>

      {/* Model Loading Guardrails */}
      <div className="bibulm-card p-4 space-y-3 text-[12px]">
        <div className="font-semibold text-[var(--label-primary)] flex items-center gap-1.5">
          <ShieldAlert className="w-4 h-4 text-[var(--signal-amber)]" />
          <span>Model Loading Guardrails</span>
        </div>
        <div className="space-y-2">
          {GUARDRAILS.map((g) => (
            <label
              key={g.id}
              className={`flex items-start gap-3 p-3 rounded-xl border cursor-pointer transition-colors ${
                settings.guardrail_level === g.id
                  ? 'bg-[var(--accent-soft)] border-[var(--accent-border)]'
                  : 'bg-[var(--bg-surface-inset)] border-[var(--border-hairline)] hover:border-[var(--border-strong)]'
              }`}
            >
              <input
                type="radio"
                name="guardrail"
                checked={settings.guardrail_level === g.id}
                onChange={() => onSaveSettings({ ...settings, guardrail_level: g.id })}
                className="mt-0.5 accent-[var(--accent)]"
              />
              <div>
                <div className="font-medium text-[var(--label-primary)]">{g.label}</div>
                <div className="text-[11px] text-[var(--label-secondary)]">{g.desc}</div>
              </div>
            </label>
          ))}
        </div>
      </div>
        </div>
      </div>
    </div>
  );
};

const IsaPill: React.FC<{ label: string }> = ({ label }) => (
  <span className="apple-badge font-mono text-[10px] py-0 px-1.5">{label}</span>
);

const GaugeCard: React.FC<{
  title: string;
  value: string;
  pct: number;
  barCls: string;
  sub: string;
}> = ({ title, value, pct, barCls, sub }) => (
  <div className="apple-card-inset p-3.5 space-y-2 tabular-nums">
    <div className="text-[11px] text-[var(--label-secondary)] font-medium">{title}</div>
    <div className="text-[14px] font-mono font-semibold text-[var(--label-primary)]">{value}</div>
    <div className="h-1.5 rounded-full bg-[var(--bg-control)] overflow-hidden">
      <div
        className={`h-full transition-all duration-300 rounded-full ${barCls}`}
        style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
      />
    </div>
    <div className="text-[11px] text-[var(--label-tertiary)]">{sub}</div>
  </div>
);
