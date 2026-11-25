import React from 'react';
import { Layers, Plus, Sparkles, Trash2 } from 'lucide-react';
import type { LoadConfig, MoeOffloadMode, TensorOverride } from '../../types';

interface MoeOffloadSectionProps {
  config: LoadConfig;
  maxLayers: number;
  maxMoeLayers: number;
  recommendedNCpuMoe: number | null;
  onPatch: (p: Partial<LoadConfig>) => void;
  onAutoFitMoe: () => void;
}

export const MoeOffloadSection: React.FC<MoeOffloadSectionProps> = ({
  config,
  maxLayers,
  maxMoeLayers,
  recommendedNCpuMoe,
  onPatch,
  onAutoFitMoe,
}) => {
  const setMoeModeKind = (kind: MoeOffloadMode['mode']) => {
    if (kind === 'full_gpu') {
      onPatch({ moe_offload: { mode: 'full_gpu' } });
    } else if (kind === 'all_experts_cpu') {
      onPatch({ moe_offload: { mode: 'all_experts_cpu' }, gpu_layers: 999 });
    } else if (kind === 'first_n_layers_experts_cpu') {
      const initN = recommendedNCpuMoe ?? Math.floor(maxMoeLayers / 2);
      onPatch({
        moe_offload: { mode: 'first_n_layers_experts_cpu', value: Math.max(1, initN) },
        gpu_layers: 999,
      });
    } else {
      onPatch({
        moe_offload: {
          mode: 'custom_overrides',
          value: [{ pattern: 'blk\\.\\d+\\.ffn_.*_exps\\.weight', buffer: 'CPU' }],
        },
        gpu_layers: 999,
      });
    }
  };

  return (
    <div className="rounded-xl border p-4 space-y-3 bg-[var(--signal-purple-soft)] border-[var(--signal-purple-border)]">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2.5">
          <Layers className="w-4 h-4 text-[var(--signal-purple)] shrink-0" />
          <div>
            <div className="font-semibold text-[var(--label-primary)]">
              MoE Selective VRAM / RAM Offloading
            </div>
            <div className="text-[11px] text-[var(--label-secondary)]">
              Keep attention, shared experts &amp; KV cache on GPU VRAM; spill only routed MoE
              experts to System RAM.
            </div>
          </div>
        </div>
        <button
          type="button"
          onClick={onAutoFitMoe}
          className="h-8 px-3 rounded-lg bg-[var(--signal-purple)] hover:brightness-110 text-white font-medium flex items-center gap-1.5 shrink-0 shadow-sm transition-all cursor-pointer"
        >
          <Sparkles className="w-3.5 h-3.5" />
          <span>Auto-Fit MoE to VRAM</span>
        </button>
      </div>

      <div className="apple-segmented grid grid-cols-4 w-full">
        {(
          [
            { id: 'full_gpu', label: 'Full GPU' },
            { id: 'all_experts_cpu', label: 'All Experts RAM' },
            { id: 'first_n_layers_experts_cpu', label: 'Smart Hybrid' },
            { id: 'custom_overrides', label: 'Custom Rules' },
          ] as const
        ).map((opt) => (
          <button
            key={opt.id}
            type="button"
            onClick={() => setMoeModeKind(opt.id)}
            className={`apple-segmented-item py-1.5 text-[11px] ${
              config.moe_offload.mode === opt.id ? 'apple-segmented-item-active' : ''
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {config.moe_offload.mode === 'first_n_layers_experts_cpu' && (
        <div className="space-y-1.5 pt-1">
          <div className="flex items-center justify-between">
            <span className="text-[var(--label-primary)]">
              Routed MoE Expert Layers on CPU RAM (<code>--n-cpu-moe</code>)
            </span>
            <div className="flex items-center gap-2 tabular-nums">
              <span className="text-[11px] text-[var(--signal-green)] font-medium">
                {Math.max(0, maxMoeLayers - config.moe_offload.value)} layers on VRAM
              </span>
              <input
                type="number"
                min={0}
                max={maxMoeLayers}
                value={config.moe_offload.value}
                onChange={(e) =>
                  onPatch({
                    moe_offload: {
                      mode: 'first_n_layers_experts_cpu',
                      value: Math.max(
                        0,
                        Math.min(maxMoeLayers, parseInt(e.target.value || '0', 10)),
                      ),
                    },
                  })
                }
                className="w-16 bibulm-input h-7 text-right font-mono"
              />
            </div>
          </div>
          <input
            type="range"
            min={0}
            max={maxMoeLayers}
            value={config.moe_offload.value}
            onChange={(e) =>
              onPatch({
                moe_offload: {
                  mode: 'first_n_layers_experts_cpu',
                  value: parseInt(e.target.value, 10),
                },
              })
            }
            className="w-full"
          />
        </div>
      )}

      {config.moe_offload.mode === 'custom_overrides' && (
        <TensorOverridesEditor
          rules={config.moe_offload.value}
          onChange={(next) =>
            onPatch({ moe_offload: { mode: 'custom_overrides', value: next } })
          }
        />
      )}
    </div>
  );
};

export const TensorOverridesEditor: React.FC<{
  rules: TensorOverride[];
  onChange: (next: TensorOverride[]) => void;
}> = ({ rules, onChange }) => (
  <div className="space-y-2 pt-1">
    {rules.map((rule, idx) => (
      <div key={idx} className="flex items-center gap-2">
        <input
          value={rule.pattern}
          onChange={(e) => {
            const next = [...rules];
            next[idx] = { ...next[idx], pattern: e.target.value };
            onChange(next);
          }}
          placeholder="blk\.\d+\.ffn_.*_exps\.weight"
          className="flex-1 bibulm-input h-7 font-mono text-[11px]"
        />
        <span className="text-[var(--label-tertiary)]">=</span>
        <input
          value={rule.buffer}
          onChange={(e) => {
            const next = [...rules];
            next[idx] = { ...next[idx], buffer: e.target.value };
            onChange(next);
          }}
          placeholder="CPU"
          className="w-24 bibulm-input h-7 font-mono text-[11px]"
        />
        <button
          type="button"
          onClick={() => onChange(rules.filter((_, i) => i !== idx))}
          className="apple-btn-ghost w-7 h-7 hover:text-[var(--signal-red)]"
          aria-label="Remove override rule"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>
    ))}
    <button
      type="button"
      onClick={() =>
        onChange([...rules, { pattern: 'blk\\.(0|1|2)\\.ffn_.*_exps', buffer: 'CPU' }])
      }
      className="inline-flex items-center gap-1 text-[11px] font-medium text-[var(--accent-text)] hover:underline cursor-pointer"
    >
      <Plus className="w-3 h-3" /> Add Tensor Override Rule
    </button>
  </div>
);
