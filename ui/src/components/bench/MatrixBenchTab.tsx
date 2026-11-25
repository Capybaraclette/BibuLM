import React, { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, Loader2, Play, Table2 } from 'lucide-react';
import type {
  AppSettings,
  InstalledRuntimeRecord,
  LlamaBenchRow,
  LoadedModelInfo,
  LocalModelEntry,
} from '../../types';
import { api, formatBytes } from '../../api';

interface MatrixBenchTabProps {
  activeModel: LoadedModelInfo | null;
  localModels: LocalModelEntry[];
  runtimes: InstalledRuntimeRecord[];
  settings: AppSettings | null;
  onOpenQuickLoad: () => void;
}

function parseIntList(raw: string, fallback: number[]): number[] {
  const nums = raw
    .split(',')
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => Number.isFinite(n) && n >= 0);
  return nums.length > 0 ? nums : fallback;
}

function parseStringList(raw: string, fallback: string[]): string[] {
  const items = raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  return items.length > 0 ? items : fallback;
}

export const MatrixBenchTab: React.FC<MatrixBenchTabProps> = ({
  activeModel,
  localModels,
  runtimes,
  settings,
  onOpenQuickLoad,
}) => {
  const defaultModelId = useMemo(() => {
    if (activeModel) {
      const match = localModels.find(
        (m) => m.id === activeModel.model_id || m.path === activeModel.load_config.model_path,
      );
      if (match) return match.id;
    }
    return localModels[0]?.id ?? '';
  }, [activeModel?.model_id, localModels]);

  const defaultRuntimeId = useMemo(() => {
    if (settings?.active_runtime_id) {
      const activeRt = runtimes.find((r) => r.id === settings.active_runtime_id);
      if (activeRt) return activeRt.id;
    }
    return runtimes.find((r) => r.bench_exe)?.id ?? runtimes[0]?.id ?? '';
  }, [settings?.active_runtime_id, runtimes]);

  const [selectedModelId, setSelectedModelId] = useState(defaultModelId);
  const [selectedRuntimeId, setSelectedRuntimeId] = useState(defaultRuntimeId);
  const [promptSizesRaw, setPromptSizesRaw] = useState('128, 512');
  const [genSizesRaw, setGenSizesRaw] = useState('64, 128');
  const [nglRaw, setNglRaw] = useState('99');
  const [faOff, setFaOff] = useState(false);
  const [faOn, setFaOn] = useState(true);
  const [cacheTypesKRaw, setCacheTypesKRaw] = useState('f16, q8_0');
  const [repetitions, setRepetitions] = useState(3);
  const [threadsRaw, setThreadsRaw] = useState('');
  const [batchSizeRaw, setBatchSizeRaw] = useState('');

  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rows, setRows] = useState<LlamaBenchRow[]>([]);

  useEffect(() => {
    if (!selectedModelId && defaultModelId) setSelectedModelId(defaultModelId);
  }, [defaultModelId, selectedModelId]);

  useEffect(() => {
    if (!selectedRuntimeId && defaultRuntimeId) setSelectedRuntimeId(defaultRuntimeId);
  }, [defaultRuntimeId, selectedRuntimeId]);

  const chosenModel = useMemo(
    () => localModels.find((m) => m.id === selectedModelId) ?? localModels[0] ?? null,
    [localModels, selectedModelId],
  );

  const chosenRuntime = useMemo(
    () => runtimes.find((r) => r.id === selectedRuntimeId) ?? null,
    [runtimes, selectedRuntimeId],
  );

  const handleRunMatrix = async () => {
    if (!chosenModel || running) return;
    setRunning(true);
    setError(null);
    setRows([]);

    const flashAttnList: boolean[] = [];
    if (faOff) flashAttnList.push(false);
    if (faOn) flashAttnList.push(true);
    if (flashAttnList.length === 0) flashAttnList.push(true);

    const parsedThreads = parseInt(threadsRaw.trim(), 10);
    const parsedBatch = parseInt(batchSizeRaw.trim(), 10);

    try {
      const result = await api.runLlamaBenchMatrix(selectedRuntimeId || null, {
        model_path: chosenModel.path,
        model_id: chosenModel.id,
        runtime_tag: chosenRuntime?.tag ?? 'active',
        prompt_sizes: parseIntList(promptSizesRaw, [128, 512]),
        gen_sizes: parseIntList(genSizesRaw, [64, 128]),
        n_gpu_layers: parseIntList(nglRaw, [99]),
        flash_attn: flashAttnList,
        cache_types_k: parseStringList(cacheTypesKRaw, ['f16']),
        cache_types_v: ['f16'],
        threads: Number.isFinite(parsedThreads) && parsedThreads > 0 ? parsedThreads : null,
        batch_size: Number.isFinite(parsedBatch) && parsedBatch > 0 ? parsedBatch : null,
        ubatch_size: null,
        repetitions: Math.max(1, repetitions),
        extra_args: [],
      });
      setRows(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  };

  const maxSpeed = Math.max(...rows.map((r) => r.avg_ts), 1);

  return (
    <div className="space-y-4">
      <div className="bibulm-card p-4 space-y-3.5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <div className="text-[13px] font-semibold text-[var(--label-primary)] flex items-center gap-1.5">
              <BarChart3 className="w-4 h-4 text-[var(--accent-text)]" />
              <span>llama-bench Sweep Matrix</span>
            </div>
            <div className="text-[11px] text-[var(--label-secondary)] mt-0.5">
              Runs official <code className="font-mono text-[var(--label-primary)]">llama-bench -o json</code>{' '}
              sweeps across prompt sizes, generation lengths, GPU layers, Flash Attention, and KV
              quantization.
            </div>
          </div>

          <button
            type="button"
            onClick={handleRunMatrix}
            disabled={!chosenModel || running || !chosenRuntime?.bench_exe}
            className="bibulm-btn-primary px-4 font-semibold"
          >
            {running ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
            <span>{running ? 'Running Matrix…' : 'Run llama-bench Sweep'}</span>
          </button>
        </div>

        {/* Model & Runtime Selectors */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[12px]">
          <div className="space-y-1">
            <label className="text-[11px] text-[var(--label-secondary)] font-medium">
              Target Local Model
            </label>
            {localModels.length === 0 ? (
              <div className="apple-card-inset flex items-center justify-between p-2 text-[var(--label-secondary)]">
                <span>No local GGUF models indexed.</span>
                <button
                  type="button"
                  onClick={onOpenQuickLoad}
                  className="text-[var(--accent-text)] hover:underline text-[11px] cursor-pointer"
                >
                  Open Library
                </button>
              </div>
            ) : (
              <select
                value={chosenModel?.id ?? ''}
                onChange={(e) => setSelectedModelId(e.target.value)}
                className="w-full apple-select"
              >
                {localModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.display_name} ({m.quantization ?? 'GGUF'} · {formatBytes(m.size_bytes)})
                  </option>
                ))}
              </select>
            )}
          </div>

          <div className="space-y-1">
            <label className="text-[11px] text-[var(--label-secondary)] font-medium flex items-center justify-between">
              <span>llama.cpp Runtime</span>
              {chosenRuntime && (
                <span
                  className={`apple-badge font-mono text-[10px] py-0 px-1.5 ${
                    chosenRuntime.bench_exe ? 'apple-badge-green' : 'apple-badge-amber'
                  }`}
                >
                  {chosenRuntime.bench_exe ? 'llama-bench ready' : 'bench_exe missing'}
                </span>
              )}
            </label>
            <select
              value={selectedRuntimeId}
              onChange={(e) => setSelectedRuntimeId(e.target.value)}
              className="w-full apple-select"
            >
              {runtimes.length === 0 && <option value="">No runtime installed</option>}
              {runtimes.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.display_name} ({r.backend} · {r.tag})
                </option>
              ))}
            </select>
          </div>
        </div>

        {chosenRuntime && !chosenRuntime.bench_exe && (
          <div className="p-2.5 rounded-lg bg-[var(--signal-amber-soft)] border border-[var(--signal-amber-border)] text-[11px] text-[var(--signal-amber)] flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>
              The selected runtime (<code className="font-mono">{chosenRuntime.tag}</code>) does not
              contain a <code className="font-mono">llama-bench</code> binary. Install or update a
              runtime release that bundles <code className="font-mono">llama-bench</code>.
            </span>
          </div>
        )}

        {/* Sweep Grid Parameters */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2.5 text-[12px]">
          <div className="space-y-1">
            <label className="text-[11px] text-[var(--label-secondary)]">Prompt sizes (-p)</label>
            <input
              value={promptSizesRaw}
              onChange={(e) => setPromptSizesRaw(e.target.value)}
              placeholder="128, 512"
              className="w-full bibulm-input font-mono"
            />
          </div>
          <div className="space-y-1">
            <label className="text-[11px] text-[var(--label-secondary)]">Gen sizes (-n)</label>
            <input
              value={genSizesRaw}
              onChange={(e) => setGenSizesRaw(e.target.value)}
              placeholder="64, 128"
              className="w-full bibulm-input font-mono"
            />
          </div>
          <div className="space-y-1">
            <label className="text-[11px] text-[var(--label-secondary)]">GPU layers (-ngl)</label>
            <input
              value={nglRaw}
              onChange={(e) => setNglRaw(e.target.value)}
              placeholder="0, 32, 99"
              className="w-full bibulm-input font-mono"
            />
          </div>
          <div className="space-y-1">
            <label className="text-[11px] text-[var(--label-secondary)]">KV type_k (-ctk)</label>
            <input
              value={cacheTypesKRaw}
              onChange={(e) => setCacheTypesKRaw(e.target.value)}
              placeholder="f16, q8_0"
              className="w-full bibulm-input font-mono"
            />
          </div>
          <div className="space-y-1">
            <label className="text-[11px] text-[var(--label-secondary)]">Flash Attn (-fa)</label>
            <div className="h-[30px] px-2.5 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-hairline)] flex items-center gap-3">
              <label className="flex items-center gap-1 cursor-pointer text-[11px] text-[var(--label-primary)]">
                <input
                  type="checkbox"
                  checked={faOff}
                  onChange={(e) => setFaOff(e.target.checked)}
                  className="accent-[var(--accent)]"
                />
                Off
              </label>
              <label className="flex items-center gap-1 cursor-pointer text-[11px] text-[var(--label-primary)]">
                <input
                  type="checkbox"
                  checked={faOn}
                  onChange={(e) => setFaOn(e.target.checked)}
                  className="accent-[var(--accent)]"
                />
                On
              </label>
            </div>
          </div>
          <div className="space-y-1">
            <label className="text-[11px] text-[var(--label-secondary)]">Repetitions (-r)</label>
            <input
              type="number"
              min={1}
              max={10}
              value={repetitions}
              onChange={(e) => setRepetitions(parseInt(e.target.value || '3', 10))}
              className="w-full bibulm-input font-mono text-right tabular-nums"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5 text-[12px]">
          <div className="space-y-1">
            <label className="text-[11px] text-[var(--label-secondary)]">Threads (optional)</label>
            <input
              value={threadsRaw}
              onChange={(e) => setThreadsRaw(e.target.value)}
              placeholder="Auto"
              className="w-full bibulm-input h-7 font-mono"
            />
          </div>
          <div className="space-y-1">
            <label className="text-[11px] text-[var(--label-secondary)]">
              Batch size (optional)
            </label>
            <input
              value={batchSizeRaw}
              onChange={(e) => setBatchSizeRaw(e.target.value)}
              placeholder="Auto"
              className="w-full bibulm-input h-7 font-mono"
            />
          </div>
        </div>

        {error && (
          <div className="p-3 rounded-lg bg-[var(--signal-red-soft)] border border-[var(--signal-red-border)] text-[12px] text-[var(--signal-red)] font-mono">
            {error}
          </div>
        )}
      </div>

      {/* Results Table & Visual Comparison */}
      {rows.length > 0 && (
        <div className="bibulm-card p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-[13px] font-semibold text-[var(--label-primary)] flex items-center gap-1.5">
              <Table2 className="w-4 h-4 text-[var(--signal-green)]" />
              <span>Matrix Results ({rows.length} measurements)</span>
            </span>
          </div>

          <div className="overflow-x-auto border border-[var(--border-hairline)] rounded-xl bg-[var(--bg-surface)]">
            <table className="w-full text-left border-collapse text-[12px]">
              <thead>
                <tr className="border-b border-[var(--border-hairline)] text-[11px] text-[var(--label-secondary)] bg-[var(--bg-surface-inset)]">
                  <th className="px-3 py-2 font-semibold">Test</th>
                  <th className="px-3 py-2 font-semibold text-right">Prompt / Gen</th>
                  <th className="px-3 py-2 font-semibold text-right">NGL</th>
                  <th className="px-3 py-2 font-semibold text-center">FA</th>
                  <th className="px-3 py-2 font-semibold">KV Type</th>
                  <th className="px-3 py-2 font-semibold text-right">Throughput (tok/s)</th>
                  <th className="px-3 py-2 font-semibold w-40">Relative</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border-hairline)]">
                {rows.map((row, idx) => {
                  const pct = Math.max(4, Math.round((row.avg_ts / maxSpeed) * 100));
                  const isPrompt = row.test_label.startsWith('pp');
                  return (
                    <tr key={idx} className="hover:bg-[var(--bg-control)] transition-colors">
                      <td className="px-3 py-2 font-mono font-semibold">
                        <span
                          className={`apple-badge text-[10px] ${
                            isPrompt ? 'apple-badge-teal' : 'apple-badge-green'
                          }`}
                        >
                          {row.test_label}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-[var(--label-secondary)] tabular-nums">
                        p{row.n_prompt} / n{row.n_gen}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-[var(--label-secondary)] tabular-nums">
                        {row.n_gpu_layers}
                      </td>
                      <td className="px-3 py-2 text-center font-mono text-[var(--label-secondary)]">
                        {row.flash_attn ? 'on' : 'off'}
                      </td>
                      <td className="px-3 py-2 font-mono text-[var(--label-tertiary)]">
                        {row.type_k}/{row.type_v}
                      </td>
                      <td className="px-3 py-2 text-right font-mono font-bold text-[var(--label-primary)] tabular-nums">
                        {row.avg_ts.toFixed(2)}{' '}
                        <span className="text-[10px] font-normal text-[var(--label-tertiary)]">
                          ± {row.stddev_ts.toFixed(2)}
                        </span>
                      </td>
                      <td className="px-3 py-2">
                        <div className="h-2 rounded-full bg-[var(--bg-control)] overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              isPrompt ? 'bg-[#06b6d4]' : 'bg-[var(--signal-green-fill)]'
                            }`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
