import React from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';
import type { GuardrailLevel, LoadConfig, MemoryEstimate } from '../../types';
import { formatMiB } from '../../api';
import { ComponentBreakdownView, FitVerdictBadge } from '../PlacementBadge';

interface MemoryPreviewCardProps {
  estimate: MemoryEstimate | null;
  config: LoadConfig;
  guardrailLevel: GuardrailLevel;
  limitDedicatedVram: boolean;
  forceOverride: boolean;
  onForceOverrideChange: (v: boolean) => void;
  resettingVramAvg: boolean;
  isRemote: boolean;
  onResetVramAvg: () => void;
}

export const MemoryPreviewCard: React.FC<MemoryPreviewCardProps> = ({
  estimate,
  config,
  guardrailLevel,
  limitDedicatedVram,
  forceOverride,
  onForceOverrideChange,
  resettingVramAvg,
  isRemote,
  onResetVramAvg,
}) => {
  const vramExceedsAvail = Boolean(
    limitDedicatedVram &&
      (guardrailLevel === 'balanced' || guardrailLevel === 'strict') &&
      estimate &&
      estimate.available_vram_mib > 0 &&
      estimate.total_vram_mib > estimate.available_vram_mib * 1.02,
  );
  const guardrailBlocks = Boolean(
    !isRemote &&
      estimate &&
      ((guardrailLevel === 'strict' && estimate.guardrail_warning) || vramExceedsAvail),
  );

  return (
    <div className="px-5 py-3.5 bg-[var(--bg-surface-inset)] border-b border-[var(--border-hairline)] space-y-2.5">
      <div className="flex flex-wrap items-center justify-between gap-2 text-[12px]">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-[var(--label-primary)]">
            Estimated Memory Breakdown
          </span>
          {estimate && <FitVerdictBadge verdict={estimate.fit_verdict} />}
        </div>
        {estimate ? (
          <div className="flex items-center gap-3 font-mono text-[12px] tabular-nums">
            <span className="text-[var(--label-secondary)]">
              VRAM{' '}
              <strong className="apple-badge apple-badge-green font-semibold">
                {formatMiB(estimate.total_vram_mib)}
              </strong>
              {estimate.available_vram_mib > 0 && (
                <span className="text-[var(--label-tertiary)] ml-1.5">
                  / {formatMiB(estimate.available_vram_mib)} free
                </span>
              )}
            </span>
            <span className="text-[var(--label-secondary)]">
              RAM{' '}
              <strong className="apple-badge apple-badge-amber font-semibold">
                {formatMiB(estimate.total_ram_mib)}
              </strong>
            </span>
          </div>
        ) : (
          <span className="text-[var(--label-tertiary)] text-[12px]">Calculating…</span>
        )}
      </div>

      {estimate && estimate.hardware_vram_total_mib > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-hairline)] px-3 py-1.5 text-[11px]">
          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-0.5 text-[var(--label-secondary)] font-mono tabular-nums">
            <span>
              Real free VRAM:{' '}
              <strong className="text-[var(--accent-text)]">
                {formatMiB(estimate.available_vram_mib)}
              </strong>
              <span className="text-[var(--label-tertiary)]">
                {' '}
                / {formatMiB(estimate.hardware_vram_total_mib)}
              </span>
            </span>
            <span className="text-[var(--border-strong)]">•</span>
            <span>
              In use (avg{estimate.vram_sample_count > 0 ? ` · ${estimate.vram_sample_count} samples` : ''}):{' '}
              <span className="text-[var(--label-primary)]">
                {formatMiB(estimate.current_vram_used_mib)}
              </span>
            </span>
            <span className="text-[var(--border-strong)]">•</span>
            {estimate.available_vram_mib >= estimate.total_vram_mib ? (
              <span className="text-[var(--signal-green)] font-medium">
                Headroom: +{formatMiB(estimate.available_vram_mib - estimate.total_vram_mib)}
              </span>
            ) : (
              <span className="text-[var(--signal-red)] font-semibold">
                Short by {formatMiB(estimate.total_vram_mib - estimate.available_vram_mib)}
              </span>
            )}
          </div>
          <button
            type="button"
            onClick={onResetVramAvg}
            disabled={resettingVramAvg}
            title={
              isRemote
                ? 'Refresh live VRAM & RAM telemetry from remote machine'
                : 'Reset rolling VRAM consumption average and re-measure free VRAM right now'
            }
            className="bibulm-btn-secondary h-6 px-2 text-[10px]"
          >
            <RotateCcw className={`w-3 h-3 ${resettingVramAvg ? 'animate-spin' : ''}`} />
            <span>{isRemote ? 'Refresh VRAM' : 'Reset Avg'}</span>
          </button>
        </div>
      )}

      {estimate?.breakdown && (
        <ComponentBreakdownView
          breakdown={estimate.breakdown}
          compact
          visionDisabledSavedMib={!config.enable_vision ? estimate.vision_available_mib : 0}
          mtpDisabledAvailableMib={
            !(
              config.mtp_enabled &&
              (!config.mtp_spec_type ||
                config.mtp_spec_type === 'draft-mtp' ||
                config.mtp_spec_type === 'mtp')
            )
              ? estimate.mtp_available_mib
              : 0
          }
        />
      )}

      {(estimate?.guardrail_warning || guardrailBlocks) && (
        <div className="flex flex-col gap-1.5 rounded-xl bg-[var(--signal-red-soft)] border border-[var(--signal-red-border)] px-3 py-2 text-[11px] text-[var(--signal-red)]">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            <span>
              {estimate?.guardrail_warning ??
                `Estimated VRAM (${formatMiB(estimate?.total_vram_mib ?? 0)}) exceeds available dedicated VRAM (${formatMiB(estimate?.available_vram_mib ?? 0)}).`}
            </span>
          </div>
          {guardrailBlocks && (
            <label className="flex items-center gap-2 pl-6 text-[11px] text-[var(--label-primary)] cursor-pointer">
              <input
                type="checkbox"
                checked={forceOverride}
                onChange={(e) => onForceOverrideChange(e.target.checked)}
                className="rounded accent-[var(--signal-red)]"
              />
              <span>Override {guardrailLevel} guardrail &amp; force load anyway</span>
            </label>
          )}
        </div>
      )}
    </div>
  );
};
