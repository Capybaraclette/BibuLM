import React, { useEffect, useState } from 'react';
import {
  Brain,
  ChevronDown,
  ChevronRight,
  FileJson,
  MessageSquare,
  Plus,
  Save,
  SlidersHorizontal,
  Trash2,
  Wrench,
  X,
} from 'lucide-react';
import type { NamedPresetRecord, ReasoningEffort, SamplingParams } from '../types';
import { api } from '../api';
import { Toggle } from './Toggle';

interface InspectorPanelProps {
  sampling: SamplingParams;
  onChange: (next: SamplingParams) => void;
  onPersistDefaults: (next: SamplingParams) => void;
  reasoningEffort: ReasoningEffort;
  onReasoningEffortChange: (effort: ReasoningEffort) => void;
}

export const InspectorPanel: React.FC<InspectorPanelProps> = ({
  sampling,
  onChange,
  onPersistDefaults,
  reasoningEffort,
  onReasoningEffortChange,
}) => {
  const [presets, setPresets] = useState<NamedPresetRecord[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('');
  const [savingName, setSavingName] = useState('');
  const [showSaveInput, setShowSaveInput] = useState(false);
  const [stopDraft, setStopDraft] = useState('');
  const [showAdvancedSamplers, setShowAdvancedSamplers] = useState(false);
  const [savedFlash, setSavedFlash] = useState(false);

  const loadPresets = async () => {
    try {
      const list = await api.listNamedPresets('sampling');
      setPresets(list);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    loadPresets();
  }, []);

  const update = (patch: Partial<SamplingParams>) => {
    onChange({ ...sampling, ...patch });
  };

  const handleSaveNamedPreset = async () => {
    const name = savingName.trim();
    if (!name) return;
    const saved = await api.saveNamedPreset(null, name, 'sampling', JSON.stringify(sampling));
    setSavingName('');
    setShowSaveInput(false);
    setSelectedPresetId(saved.id);
    await loadPresets();
  };

  const handleSelectPreset = (id: string) => {
    setSelectedPresetId(id);
    const found = presets.find((p) => p.id === id);
    if (found) {
      try {
        const parsed = JSON.parse(found.payload_json) as SamplingParams;
        onChange({ ...sampling, ...parsed });
      } catch {
        // ignore
      }
    }
  };

  const handleDeletePreset = async () => {
    if (!selectedPresetId) return;
    await api.deleteNamedPreset(selectedPresetId);
    setSelectedPresetId('');
    await loadPresets();
  };

  const handleSaveDefaults = () => {
    onPersistDefaults(sampling);
    setSavedFlash(true);
    setTimeout(() => setSavedFlash(false), 1500);
  };

  return (
    <aside className="w-[300px] shrink-0 bg-[var(--bg-sidebar)] border-l border-[var(--border-hairline)] flex flex-col overflow-hidden">
      <div className="px-4 py-3 border-b border-[var(--border-hairline)] flex items-center justify-between gap-2">
        <span className="text-[13px] font-semibold text-[var(--label-primary)] tracking-tight">
          Inference Parameters
        </span>
        <button
          type="button"
          onClick={handleSaveDefaults}
          className="h-7 px-2.5 rounded-lg bg-[var(--accent-soft)] hover:bg-[var(--accent-border)] border border-[var(--accent-border)] text-[11px] font-medium text-[var(--accent-text)] flex items-center gap-1 transition-colors cursor-pointer"
        >
          <Save className="w-3 h-3" />
          <span>{savedFlash ? 'Saved' : 'Save Defaults'}</span>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto divide-y divide-[var(--border-hairline)] text-[12px]">
        {/* Named Presets */}
        <section className="p-3.5 space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-[var(--label-primary)]">Preset</span>
            {selectedPresetId && (
              <button
                type="button"
                onClick={handleDeletePreset}
                className="apple-btn-ghost w-7 h-7 hover:text-[var(--signal-red)]"
                title="Delete selected preset"
                aria-label="Delete selected preset"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
          <select
            value={selectedPresetId}
            onChange={(e) => handleSelectPreset(e.target.value)}
            className="w-full apple-select"
          >
            <option value="">Custom / Active Parameters</option>
            {presets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>

          {showSaveInput ? (
            <div className="flex gap-1.5">
              <input
                value={savingName}
                onChange={(e) => setSavingName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSaveNamedPreset()}
                placeholder="Preset name…"
                autoFocus
                className="flex-1 bibulm-input h-7"
              />
              <button
                type="button"
                onClick={handleSaveNamedPreset}
                className="bibulm-btn-primary h-7 px-2.5 text-[11px]"
              >
                Save
              </button>
              <button
                type="button"
                onClick={() => setShowSaveInput(false)}
                className="apple-btn-ghost w-7 h-7"
                aria-label="Cancel save preset"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setShowSaveInput(true)}
              className="w-full bibulm-btn-secondary h-7 text-[11px]"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Save Preset As…</span>
            </button>
          )}
        </section>

        {/* System Prompt */}
        <section className="p-3.5 space-y-2">
          <div className="flex items-center gap-1.5 font-semibold text-[var(--label-primary)]">
            <MessageSquare className="w-3.5 h-3.5 text-[var(--label-tertiary)]" />
            <span>System Prompt</span>
          </div>
          <textarea
            rows={3}
            value={sampling.system_prompt}
            onChange={(e) => update({ system_prompt: e.target.value })}
            placeholder="Optional benchmark or default system prompt…"
            className="w-full p-2.5 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-hairline)] text-[12px] text-[var(--label-primary)] placeholder-[var(--label-quaternary)] resize-y focus:outline-none focus:border-[var(--accent)]"
          />
        </section>

        {/* Reasoning Override */}
        <section className="p-3.5 space-y-2.5">
          <div className="flex items-center gap-1.5 font-semibold text-[var(--label-primary)]">
            <Brain className="w-3.5 h-3.5 text-[var(--signal-purple)]" />
            <span>Reasoning Effort</span>
          </div>
          <div className="grid grid-cols-4 gap-1 p-1 rounded-lg bg-[var(--bg-control)] border border-[var(--border-hairline)]">
            {(['default', 'off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'] as const).map(
              (lvl) => (
                <button
                  key={lvl}
                  type="button"
                  onClick={() => onReasoningEffortChange(lvl)}
                  className={`py-1.5 rounded-md text-[11px] capitalize transition-all cursor-pointer ${
                    reasoningEffort === lvl
                      ? 'bg-[var(--bg-surface)] text-[var(--accent-text)] font-semibold shadow-sm'
                      : 'text-[var(--label-secondary)] hover:text-[var(--label-primary)]'
                  }`}
                >
                  {lvl}
                </button>
              ),
            )}
          </div>
        </section>

        {/* Core Generation Settings */}
        <section className="p-3.5 space-y-3">
          <div className="flex items-center gap-1.5 font-semibold text-[var(--label-primary)]">
            <Wrench className="w-3.5 h-3.5 text-[var(--label-tertiary)]" />
            <span>Generation</span>
          </div>

          <SliderRow
            label="Temperature"
            value={sampling.temperature ?? 0.7}
            min={0}
            max={2}
            step={0.05}
            onChange={(v) => update({ temperature: v })}
          />

          <div className="flex items-center justify-between gap-2">
            <span className="text-[var(--label-secondary)]">Max Tokens</span>
            <input
              type="number"
              min={16}
              max={131072}
              value={sampling.max_tokens ?? 1024}
              onChange={(e) =>
                update({ max_tokens: Math.max(16, parseInt(e.target.value || '1024', 10)) })
              }
              className="w-24 bibulm-input h-7 text-right font-mono tabular-nums"
            />
          </div>

          <div className="space-y-1.5">
            <span className="text-[var(--label-secondary)]">Stop Strings</span>
            <input
              value={stopDraft}
              onChange={(e) => setStopDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && stopDraft.trim()) {
                  update({ stop_strings: [...sampling.stop_strings, stopDraft] });
                  setStopDraft('');
                }
              }}
              placeholder="Enter a string and press Enter"
              className="w-full bibulm-input h-7"
            />
            {sampling.stop_strings.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {sampling.stop_strings.map((s, idx) => (
                  <span
                    key={`${s}-${idx}`}
                    className="apple-badge font-mono text-[10px] py-0.5 px-1.5"
                  >
                    {s}
                    <button
                      type="button"
                      onClick={() =>
                        update({
                          stop_strings: sampling.stop_strings.filter((_, i) => i !== idx),
                        })
                      }
                      className="text-[var(--label-tertiary)] hover:text-[var(--signal-red)] cursor-pointer"
                      aria-label={`Remove stop string ${s}`}
                    >
                      <X className="w-2.5 h-2.5" />
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>
        </section>

        {/* Sampling */}
        <section className="p-3.5 space-y-3">
          <div className="flex items-center gap-1.5 font-semibold text-[var(--label-primary)]">
            <SlidersHorizontal className="w-3.5 h-3.5 text-[var(--label-tertiary)]" />
            <span>Sampling</span>
          </div>

          <div className="flex items-center justify-between gap-2">
            <span className="text-[var(--label-secondary)]">Top K Sampling</span>
            <input
              type="number"
              value={sampling.top_k ?? 40}
              onChange={(e) => update({ top_k: parseInt(e.target.value || '40', 10) })}
              className="w-20 bibulm-input h-7 text-right font-mono tabular-nums"
            />
          </div>

          <SliderRow
            label="Top P Sampling"
            value={sampling.top_p ?? 0.95}
            min={0}
            max={1}
            step={0.01}
            onChange={(v) => update({ top_p: v })}
          />

          <SliderRow
            label="Min P Sampling"
            value={sampling.min_p ?? 0.05}
            min={0}
            max={1}
            step={0.01}
            onChange={(v) => update({ min_p: v })}
          />

          <SliderRow
            label="Repeat Penalty"
            value={sampling.repeat_penalty ?? 1.05}
            min={1}
            max={2}
            step={0.01}
            onChange={(v) => update({ repeat_penalty: v })}
          />

          <button
            type="button"
            onClick={() => setShowAdvancedSamplers(!showAdvancedSamplers)}
            className="flex items-center gap-1 text-[11px] font-medium text-[var(--accent-text)] hover:underline cursor-pointer"
          >
            {showAdvancedSamplers ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5" />
            )}
            <span>Advanced Samplers (DRY, XTC, Penalties)</span>
          </button>

          {showAdvancedSamplers && (
            <div className="space-y-2.5 pt-1 pl-2.5 border-l-2 border-[var(--border-hairline)]">
              <SliderRow
                label="Presence Penalty"
                value={sampling.presence_penalty ?? 0}
                min={-2}
                max={2}
                step={0.05}
                onChange={(v) => update({ presence_penalty: v === 0 ? null : v })}
              />
              <SliderRow
                label="Frequency Penalty"
                value={sampling.frequency_penalty ?? 0}
                min={-2}
                max={2}
                step={0.05}
                onChange={(v) => update({ frequency_penalty: v === 0 ? null : v })}
              />
              <SliderRow
                label="DRY Multiplier"
                value={sampling.dry_multiplier ?? 0}
                min={0}
                max={2}
                step={0.05}
                onChange={(v) => update({ dry_multiplier: v === 0 ? null : v })}
              />
              <SliderRow
                label="XTC Probability"
                value={sampling.xtc_probability ?? 0}
                min={0}
                max={1}
                step={0.05}
                onChange={(v) => update({ xtc_probability: v === 0 ? null : v })}
              />
            </div>
          )}
        </section>

        {/* Structured Output */}
        <section className="p-3.5 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5 font-semibold text-[var(--label-primary)]">
              <FileJson className="w-3.5 h-3.5 text-[var(--label-tertiary)]" />
              <span>Structured Output</span>
            </div>
            <Toggle
              size="sm"
              checked={sampling.structured_output_enabled}
              onChange={(v) => update({ structured_output_enabled: v })}
            />
          </div>
          {sampling.structured_output_enabled && (
            <textarea
              rows={4}
              value={sampling.json_schema ?? ''}
              onChange={(e) => update({ json_schema: e.target.value })}
              placeholder='{"type":"object","properties":{...}}'
              className="w-full p-2 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-hairline)] font-mono text-[11px] text-[var(--label-primary)] focus:outline-none focus:border-[var(--accent)]"
            />
          )}
        </section>
      </div>
    </aside>
  );
};

const SliderRow: React.FC<{
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}> = ({ label, value, min, max, step, onChange }) => (
  <div className="space-y-1.5">
    <div className="flex items-center justify-between">
      <span className="text-[var(--label-secondary)]">{label}</span>
      <input
        type="number"
        step={step}
        min={min}
        max={max}
        value={Number.isFinite(value) ? value : 0}
        onChange={(e) => onChange(parseFloat(e.target.value || '0'))}
        className="w-16 bibulm-input h-6 px-1.5 text-right font-mono text-[11px] tabular-nums"
      />
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={Number.isFinite(value) ? value : 0}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full"
    />
  </div>
);
