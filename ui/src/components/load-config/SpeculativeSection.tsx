import React, { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, XCircle, Zap } from 'lucide-react';
import type {
  DraftCompatibilityReport,
  LoadConfig,
  LocalModelEntry,
  RemoteDraftCandidate,
} from '../../types';
import { api, formatBytes } from '../../api';
import { Toggle } from '../Toggle';
import { FloatField, NumberField } from './FieldPrimitives';

interface SpeculativeSectionProps {
  model: LocalModelEntry;
  config: LoadConfig;
  draftModels: RemoteDraftCandidate[];
  isRemote: boolean;
  onPatch: (p: Partial<LoadConfig>) => void;
}

export const SpeculativeSection: React.FC<SpeculativeSectionProps> = ({
  model,
  config,
  draftModels,
  isRemote,
  onPatch,
}) => {
  const [compatReport, setCompatReport] = useState<DraftCompatibilityReport | null>(null);

  useEffect(() => {
    if (isRemote || config.mtp_enabled || !config.draft_model_path) {
      setCompatReport(null);
      return;
    }
    let cancelled = false;
    api
      .checkDraftModelCompatibility(model.path, config.draft_model_path)
      .then((rep) => {
        if (!cancelled) setCompatReport(rep);
      })
      .catch(() => {
        if (!cancelled) setCompatReport(null);
      });
    return () => {
      cancelled = true;
    };
  }, [model.path, config.draft_model_path, config.mtp_enabled, isRemote]);

  return (
    <div
      className={`rounded-xl border p-4 space-y-3 ${
        config.mtp_enabled || model.has_mtp
          ? 'bg-[var(--signal-purple-soft)] border-[var(--signal-purple-border)]'
          : 'apple-card-inset'
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <Zap className="w-4 h-4 text-[var(--signal-purple)] shrink-0" />
          <div className="min-w-0">
            <div className="font-semibold text-[var(--label-primary)] flex items-center gap-2 flex-wrap">
              <span>Multi-Token Prediction (MTP) &amp; Speculative Decoding</span>
              {model.has_mtp && (
                <span className="apple-badge apple-badge-purple font-mono text-[10px]">
                  MTP Heads Detected ({model.mtp_layer_count || 1}L
                  {model.mtp_bytes > 0 ? ` · ${formatBytes(model.mtp_bytes)}` : ''})
                </span>
              )}
            </div>
            <div className="text-[11px] text-[var(--label-secondary)]">
              Use native GGUF MTP heads (<code>--spec-type draft-mtp</code>) or an external draft
              model to accelerate token generation.
            </div>
          </div>
        </div>
        <Toggle
          checked={config.mtp_enabled}
          onChange={(v) =>
            onPatch({
              mtp_enabled: v,
              draft_max: v ? (config.draft_max ?? 3) : config.draft_max,
            })
          }
        />
      </div>

      {config.mtp_enabled ? (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-2 border-t border-[var(--border-hairline)]">
          <div className="flex flex-col gap-1">
            <span className="text-[11px] text-[var(--label-secondary)]">
              MTP Flag (--spec-type)
            </span>
            <select
              value={config.mtp_spec_type ?? 'draft-mtp'}
              onChange={(e) => onPatch({ mtp_spec_type: e.target.value })}
              className="apple-select h-7 font-mono"
            >
              <option value="draft-mtp">draft-mtp (Upstream MTP)</option>
              <option value="mtp">mtp (Legacy/Fork)</option>
              <option value="draft-simple">draft-simple</option>
              <option value="draft-eagle3">draft-eagle3</option>
              <option value="draft-dflash">draft-dflash</option>
              <option value="draft-dspark">draft-dspark</option>
              <option value="draft-dflash2">draft-dflash2</option>
              <option value="draft-lilicorr">draft-lilicorr</option>
              <option value="ngram-simple">ngram-simple</option>
              <option value="ngram-map-k">ngram-map-k</option>
              <option value="ngram-map-k4v">ngram-map-k4v</option>
              <option value="ngram-mod">ngram-mod</option>
              <option value="ngram-cache">ngram-cache</option>
              <option value="none">none</option>
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[11px] text-[var(--label-secondary)]">
              Draft Max (--spec-draft-n-max)
            </span>
            <input
              type="number"
              min={1}
              max={16}
              placeholder="3"
              value={config.draft_max ?? ''}
              onChange={(e) =>
                onPatch({
                  draft_max: e.target.value === '' ? null : parseInt(e.target.value, 10),
                })
              }
              className="bibulm-input h-7 text-right font-mono tabular-nums"
            />
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[11px] text-[var(--label-secondary)]">Draft Min</span>
            <input
              type="number"
              min={0}
              max={16}
              placeholder="Auto"
              value={config.draft_min ?? ''}
              onChange={(e) =>
                onPatch({
                  draft_min: e.target.value === '' ? null : parseInt(e.target.value, 10),
                })
              }
              className="bibulm-input h-7 text-right font-mono tabular-nums"
            />
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[11px] text-[var(--label-secondary)]">Draft P-Min</span>
            <input
              type="number"
              step="0.05"
              min={0}
              max={1}
              placeholder="Auto"
              value={config.draft_p_min ?? ''}
              onChange={(e) =>
                onPatch({
                  draft_p_min: e.target.value === '' ? null : parseFloat(e.target.value),
                })
              }
              className="bibulm-input h-7 text-right font-mono tabular-nums"
            />
          </div>
        </div>
      ) : (
        <div className="space-y-2 pt-2 border-t border-[var(--border-hairline)]">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[var(--label-secondary)] shrink-0">
              External Draft Model (-md)
            </span>
            <select
              value={config.draft_model_path ?? ''}
              onChange={(e) => onPatch({ draft_model_path: e.target.value || null })}
              className="w-72 max-w-full apple-select h-7"
            >
              <option value="">None (Disabled)</option>
              {draftModels.map((m) => (
                <option key={m.id} value={m.path}>
                  {m.display_name} ({m.quantization ?? 'GGUF'})
                </option>
              ))}
            </select>
          </div>

          {compatReport && (
            <div
              className={`rounded-lg border px-3 py-2 text-[11px] space-y-0.5 ${
                compatReport.level === 'compatible'
                  ? 'bg-[var(--signal-green-soft)] border-[var(--signal-green-border)] text-[var(--signal-green)]'
                  : compatReport.level === 'warning'
                    ? 'bg-[var(--signal-amber-soft)] border-[var(--signal-amber-border)] text-[var(--signal-amber)]'
                    : 'bg-[var(--signal-red-soft)] border-[var(--signal-red-border)] text-[var(--signal-red)]'
              }`}
            >
              <div className="flex items-center gap-1.5 font-semibold">
                {compatReport.level === 'compatible' ? (
                  <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                ) : compatReport.level === 'warning' ? (
                  <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                ) : (
                  <XCircle className="w-3.5 h-3.5 shrink-0" />
                )}
                <span>
                  {compatReport.level === 'compatible'
                    ? 'Draft Tokenizer Verified Compatible'
                    : compatReport.level === 'warning'
                      ? 'Draft Compatibility Warning'
                      : 'Incompatible Draft Tokenizer'}
                </span>
              </div>
              {compatReport.reasons.map((r, idx) => (
                <div key={idx} className="pl-5 opacity-90">
                  {r}
                </div>
              ))}
            </div>
          )}

          {config.draft_model_path && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <NumberField
                label="Draft GPU Layers (-ngld)"
                value={config.draft_gpu_layers}
                placeholder="Auto"
                onChange={(v) => onPatch({ draft_gpu_layers: v })}
              />
              <NumberField
                label="Draft Max (--spec-draft-n-max)"
                value={config.draft_max}
                placeholder="3"
                onChange={(v) => onPatch({ draft_max: v })}
              />
              <NumberField
                label="Draft Min (--spec-draft-n-min)"
                value={config.draft_min}
                placeholder="0"
                onChange={(v) => onPatch({ draft_min: v })}
              />
              <FloatField
                label="Draft P-Min (--spec-draft-p-min)"
                value={config.draft_p_min}
                placeholder="0.0"
                onChange={(v) => onPatch({ draft_p_min: v })}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};
