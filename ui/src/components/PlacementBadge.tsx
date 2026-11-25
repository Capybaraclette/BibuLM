import React from 'react';
import { AlertTriangle, Cpu, HelpCircle, Layers, Zap } from 'lucide-react';
import type {
  FitVerdict,
  MemoryComponentBreakdown,
  MemoryPlacementReport,
  PlacementStatus,
} from '../types';
import { breakdownTotals, formatMiB, placementTotals } from '../api';

export const PlacementBadge: React.FC<{ status: PlacementStatus; compact?: boolean }> = ({
  status,
  compact = false,
}) => {
  const cfg: Record<
    PlacementStatus,
    { label: string; short: string; cls: string; icon: React.ReactNode }
  > = {
    gpu_only: {
      label: '100% GPU (VRAM)',
      short: '100% GPU',
      cls: 'apple-badge-green',
      icon: <Zap className="w-3 h-3" />,
    },
    hybrid_gpu_cpu: {
      label: 'Hybrid (VRAM + RAM/CPU)',
      short: 'Hybrid GPU+CPU',
      cls: 'apple-badge-amber',
      icon: <Layers className="w-3 h-3" />,
    },
    gpu_spillover_warning: {
      label: 'WDDM Shared Memory Spillover',
      short: 'WDDM Spillover',
      cls: 'apple-badge-red',
      icon: <AlertTriangle className="w-3 h-3" />,
    },
    cpu_only: {
      label: 'CPU Only (System RAM)',
      short: 'CPU Only',
      cls: '',
      icon: <Cpu className="w-3 h-3" />,
    },
    unknown: {
      label: 'Placement Pending',
      short: 'Pending',
      cls: '',
      icon: <HelpCircle className="w-3 h-3" />,
    },
  };

  const item = cfg[status] ?? cfg.unknown;
  return (
    <span className={`apple-badge ${item.cls} font-medium`}>
      {item.icon}
      <span>{compact ? item.short : item.label}</span>
    </span>
  );
};

export const FitVerdictBadge: React.FC<{ verdict: FitVerdict }> = ({ verdict }) => {
  const map: Record<FitVerdict, { label: string; cls: string }> = {
    full_gpu: {
      label: 'Full GPU Fit',
      cls: 'apple-badge-green',
    },
    moe_hybrid: {
      label: 'MoE Hybrid (VRAM+RAM)',
      cls: 'apple-badge-teal',
    },
    partial_offload: {
      label: 'Partial GPU Offload',
      cls: 'apple-badge-amber',
    },
    cpu_only: {
      label: 'CPU / RAM Only',
      cls: '',
    },
    too_large: {
      label: 'Exceeds Memory',
      cls: 'apple-badge-red',
    },
  };
  const m = map[verdict] ?? map.too_large;
  return <span className={`apple-badge ${m.cls}`}>{m.label}</span>;
};

export const MemorySourceBadge: React.FC<{ source?: string | null }> = ({ source }) => {
  if (!source) return null;
  const map: Record<string, { label: string; cls: string; title: string }> = {
    verified_logs: {
      label: 'Verified (logs)',
      cls: 'apple-badge-green',
      title: 'Exact buffer sizes parsed from llama-server logs + GGUF component proportions',
    },
    telemetry_reconciled: {
      label: 'Live OS + GGUF',
      cls: 'apple-badge-teal',
      title: 'Reconciled from live per-process VRAM/RAM telemetry and native GGUF inspection',
    },
    estimated_gguf: {
      label: 'Estimated (GGUF)',
      cls: '',
      title: 'Estimated natively from GGUF tensor headers and load parameters',
    },
  };
  const item = map[source];
  if (!item) return null;
  return (
    <span title={item.title} className={`apple-badge ${item.cls} text-[10px]`}>
      {item.label}
    </span>
  );
};

interface BreakdownItem {
  key: string;
  label: string;
  vram: number;
  ram: number;
  color: string;
  mutedNote?: string;
}

export const ComponentBreakdownView: React.FC<{
  breakdown: MemoryComponentBreakdown;
  compact?: boolean;
  visionDisabledSavedMib?: number;
  mtpDisabledAvailableMib?: number;
}> = ({ breakdown, compact = false, visionDisabledSavedMib = 0, mtpDisabledAvailableMib = 0 }) => {
  const totals = breakdownTotals(breakdown);
  const grandTotal = Math.max(totals.totalGpu + totals.totalCpu, 1);

  const items: BreakdownItem[] = [
    {
      key: 'dense',
      label: 'Dense / Core',
      vram: breakdown.dense_vram_mib,
      ram: breakdown.dense_ram_mib,
      color: 'bg-[var(--signal-green-fill)]',
    },
  ];

  if (breakdown.moe_experts_vram_mib > 0.5 || breakdown.moe_experts_ram_mib > 0.5) {
    items.push({
      key: 'moe',
      label: 'MoE Experts',
      vram: breakdown.moe_experts_vram_mib,
      ram: breakdown.moe_experts_ram_mib,
      color: 'bg-[var(--signal-purple-fill)]',
    });
  }

  if (
    breakdown.mtp_vram_mib > 0.5 ||
    breakdown.mtp_ram_mib > 0.5 ||
    mtpDisabledAvailableMib > 0.5
  ) {
    items.push({
      key: 'mtp',
      label: 'MTP Heads',
      vram: breakdown.mtp_vram_mib,
      ram: breakdown.mtp_ram_mib,
      color: 'bg-[#d946ef]',
      mutedNote:
        breakdown.mtp_vram_mib <= 0.5 && breakdown.mtp_ram_mib <= 0.5 && mtpDisabledAvailableMib > 0.5
          ? `Off (${formatMiB(mtpDisabledAvailableMib)} in GGUF)`
          : undefined,
    });
  }

  if (
    breakdown.vision_vram_mib > 0.5 ||
    breakdown.vision_ram_mib > 0.5 ||
    visionDisabledSavedMib > 0.5
  ) {
    items.push({
      key: 'vision',
      label: 'Vision (mmproj)',
      vram: breakdown.vision_vram_mib,
      ram: breakdown.vision_ram_mib,
      color: 'bg-[var(--signal-amber-fill)]',
      mutedNote:
        breakdown.vision_vram_mib <= 0.5 &&
        breakdown.vision_ram_mib <= 0.5 &&
        visionDisabledSavedMib > 0.5
          ? `Off (saves ${formatMiB(visionDisabledSavedMib)})`
          : undefined,
    });
  }

  items.push(
    {
      key: 'kv',
      label: 'KV Cache',
      vram: breakdown.kv_cache_vram_mib,
      ram: breakdown.kv_cache_ram_mib,
      color: 'bg-[#06b6d4]',
    },
    {
      key: 'compute',
      label: 'Compute',
      vram: breakdown.compute_vram_mib,
      ram: breakdown.compute_ram_mib,
      color: 'bg-[var(--accent)]',
    },
  );

  // Build stacked bar segments grouped by VRAM first, then RAM
  const vramSegments = items
    .filter((i) => i.vram > 0.5)
    .map((i) => ({
      id: `vram-${i.key}`,
      label: `${i.label} (VRAM)`,
      value: i.vram,
      color: i.color,
    }));
  const ramSegments = items
    .filter((i) => i.ram > 0.5)
    .map((i) => ({
      id: `ram-${i.key}`,
      label: `${i.label} (RAM)`,
      value: i.ram,
      color: `${i.color} opacity-50`,
    }));
  const allSegments = [...vramSegments, ...ramSegments];

  return (
    <div className="space-y-2">
      {/* macOS Storage Settings-inspired Segmented Bar */}
      <div className="h-2.5 w-full rounded-full bg-[var(--bg-control)] overflow-hidden flex border border-[var(--border-hairline)] gap-px">
        {allSegments.map((seg) => (
          <div
            key={seg.id}
            style={{ width: `${(seg.value / grandTotal) * 100}%` }}
            className={`${seg.color} h-full transition-all duration-300`}
            title={`${seg.label}: ${formatMiB(seg.value)}`}
          />
        ))}
      </div>

      {/* Per-component VRAM & RAM Breakdown Grid */}
      <div
        className={`grid gap-1.5 ${
          compact
            ? 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-6'
            : 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-6'
        }`}
      >
        {items.map((item) => {
          const active = item.vram > 0.5 || item.ram > 0.5;
          return (
            <div
              key={item.key}
              className={`apple-card-inset px-2.5 py-1.5 text-[11px] ${
                active ? '' : 'opacity-65'
              }`}
            >
              <div className="flex items-center gap-1.5 text-[var(--label-primary)] font-medium truncate">
                <span className={`w-2 h-2 rounded-full shrink-0 ${item.color}`} />
                <span className="truncate">{item.label}</span>
              </div>
              {item.mutedNote ? (
                <div
                  className="text-[10px] font-mono text-[var(--label-tertiary)] mt-0.5 truncate tabular-nums"
                  title={item.mutedNote}
                >
                  {item.mutedNote}
                </div>
              ) : (
                <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 mt-0.5 font-mono text-[10px] tabular-nums">
                  {item.vram > 0.5 && (
                    <span className="text-[var(--signal-green)] font-medium" title="Dedicated GPU VRAM">
                      V {formatMiB(item.vram)}
                    </span>
                  )}
                  {item.ram > 0.5 && (
                    <span className="text-[var(--signal-amber)] font-medium" title="System RAM">
                      R {formatMiB(item.ram)}
                    </span>
                  )}
                  {!active && <span className="text-[var(--label-tertiary)]">0 MiB</span>}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const PlacementBar: React.FC<{
  placement: MemoryPlacementReport;
  breakdown?: MemoryComponentBreakdown | null;
}> = ({ placement, breakdown }) => {
  const effectiveBreakdown = breakdown ?? placement.estimated_breakdown;
  if (effectiveBreakdown) {
    const bt = breakdownTotals(effectiveBreakdown);
    if (bt.totalGpu > 0 || bt.totalCpu > 0) {
      return <ComponentBreakdownView breakdown={effectiveBreakdown} />;
    }
  }

  // Fallback for legacy saved benchmark rows without component breakdown
  const t = placementTotals(placement, breakdown);
  const grandTotal = Math.max(t.totalGpu + t.totalCpu, 1);

  const segments = [
    { label: 'VRAM Weights', value: t.gpuModel, color: 'bg-[var(--signal-green-fill)]' },
    { label: 'VRAM KV', value: t.gpuKv, color: 'bg-[#06b6d4]' },
    { label: 'VRAM Compute', value: t.gpuCompute, color: 'bg-[var(--accent)]' },
    { label: 'RAM Weights', value: t.cpuModel, color: 'bg-[var(--signal-amber-fill)]' },
    { label: 'RAM KV/Comp', value: t.cpuKv + t.cpuCompute, color: 'bg-[#fb923c]' },
  ].filter((s) => s.value > 0.5);

  return (
    <div className="space-y-2">
      <div className="h-2.5 w-full rounded-full bg-[var(--bg-control)] overflow-hidden flex border border-[var(--border-hairline)] gap-px">
        {segments.map((seg) => (
          <div
            key={seg.label}
            style={{ width: `${(seg.value / grandTotal) * 100}%` }}
            className={`${seg.color} h-full transition-all duration-300`}
            title={`${seg.label}: ${formatMiB(seg.value)}`}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-[var(--label-secondary)] tabular-nums">
        {segments.map((seg) => (
          <span key={seg.label} className="inline-flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${seg.color}`} />
            <span>{seg.label}:</span>
            <span className="text-[var(--label-primary)] font-mono font-medium">
              {formatMiB(seg.value)}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
};
