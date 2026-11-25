import React, { useEffect, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  ChevronDown,
  ChevronRight,
  Download,
  Gauge,
  History,
  Link2,
  Loader2,
  MessageSquareCode,
  Play,
  Trash2,
  Zap,
} from 'lucide-react';
import type {
  ActiveRemoteModel,
  AppSettings,
  BenchmarkRunRecord,
  InstalledRuntimeRecord,
  LlamaServerMetrics,
  LoadedModelInfo,
  LocalModelEntry,
  QuickTestResult,
  QuickTestStreamChunk,
  ReasoningEffort,
  SamplingParams,
  TelemetrySnapshot,
} from '../types';
import { api, formatBytes, formatMiB, placementTotals } from '../api';
import { MemorySourceBadge, PlacementBadge, PlacementBar } from '../components/PlacementBadge';
import { PageHeader } from '../components/PageHeader';
import { PlaygroundTab } from '../components/bench/PlaygroundTab';
import { MatrixBenchTab } from '../components/bench/MatrixBenchTab';

interface BenchmarkViewProps {
  activeModel: LoadedModelInfo | null;
  activeRemoteModel: ActiveRemoteModel | null;
  localModels: LocalModelEntry[];
  runtimes: InstalledRuntimeRecord[];
  settings: AppSettings | null;
  sampling: SamplingParams;
  reasoningEffort: ReasoningEffort;
  telemetry: TelemetrySnapshot | null;
  onOpenQuickLoad: () => void;
}

const PROMPT_PRESETS = [
  'Short QA',
  'Code Gen',
  'Long Prefill (4K)',
  'Long Prefill (8K)',
  'Reasoning Test',
  'Custom',
];

export const BenchmarkView: React.FC<BenchmarkViewProps> = ({
  activeModel,
  activeRemoteModel,
  localModels,
  runtimes,
  settings,
  sampling,
  reasoningEffort,
  telemetry,
  onOpenQuickLoad,
}) => {
  const [subTab, setSubTab] = useState<'quick' | 'playground' | 'matrix' | 'history'>('quick');
  const [liveMetrics, setLiveMetrics] = useState<LlamaServerMetrics | null>(null);

  // Quick speed test state
  const [promptPreset, setPromptPreset] = useState('Code Gen');
  const [customPrompt, setCustomPrompt] = useState('');
  const [maxTokens, setMaxTokens] = useState(512);
  const [runningQuick, setRunningQuick] = useState(false);
  const [streamReasoning, setStreamReasoning] = useState('');
  const [streamOutput, setStreamOutput] = useState('');
  const [streamTokens, setStreamTokens] = useState(0);
  const [quickResult, setQuickResult] = useState<QuickTestResult | null>(null);
  const [quickError, setQuickError] = useState<string | null>(null);
  const [showThinking, setShowThinking] = useState(true);

  // History state
  const [history, setHistory] = useState<BenchmarkRunRecord[]>([]);

  const loadHistory = async () => {
    try {
      const list = await api.listBenchmarkHistory(60);
      setHistory(list);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  useEffect(() => {
    if (!activeModel) {
      setLiveMetrics(null);
      return;
    }
    let cancelled = false;
    const poll = async () => {
      try {
        const metrics = await api.getLoadedModelMetrics(activeModel.model_id);
        if (!cancelled) setLiveMetrics(metrics);
      } catch {
        // ignore transient scrape errors
      }
    };
    poll();
    const timer = setInterval(poll, 2500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [activeModel?.model_id]);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    listen<QuickTestStreamChunk>('bench-stream-chunk', (ev) => {
      const c = ev.payload;
      if (c.reasoning_delta) setStreamReasoning((prev) => prev + c.reasoning_delta);
      if (c.content_delta) setStreamOutput((prev) => prev + c.content_delta);
      setStreamTokens(c.approx_tokens);
    }).then((fn) => {
      unlisten = fn;
    });
    return () => {
      unlisten?.();
    };
  }, []);

  const targetModelId = activeRemoteModel?.qualifiedId ?? activeModel?.model_id ?? null;

  const handleRunQuick = async () => {
    if (!targetModelId || runningQuick) return;
    setRunningQuick(true);
    setQuickError(null);
    setQuickResult(null);
    setStreamReasoning('');
    setStreamOutput('');
    setStreamTokens(0);
    try {
      const res = await api.runQuickBenchmark(targetModelId, {
        prompt_preset: promptPreset,
        custom_prompt: promptPreset === 'Custom' ? customPrompt : null,
        max_tokens: maxTokens,
        reasoning_effort: reasoningEffort === 'default' ? null : reasoningEffort,
        sampling,
      });
      setQuickResult(res);
      setStreamReasoning(res.reasoning_text);
      setStreamOutput(res.output_text);
      await loadHistory();
    } catch (e) {
      setQuickError(String(e));
    } finally {
      setRunningQuick(false);
    }
  };

  const handleExportCsv = async () => {
    const csv = await api.exportBenchmarksCsv();
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bibulm-benchmarks-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportJson = () => {
    const blob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bibulm-benchmarks-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const pt = activeModel ? placementTotals(activeModel.placement, activeModel.breakdown) : null;
  const liveSpillover = Boolean(
    activeModel &&
      pt &&
      pt.totalGpu > 0 &&
      (activeModel.placement.wddm_shared_spillover_detected ||
        telemetry?.wddm_spillover_warning),
  );
  const livePlacementStatus =
    activeModel && liveSpillover && activeModel.placement_status === 'gpu_only'
      ? 'gpu_spillover_warning'
      : activeModel?.placement_status ?? 'unknown';
  const maxTgHistory = Math.max(...history.map((h) => h.generation_tok_s), 1);

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <PageHeader
        icon={<Gauge className="w-4 h-4" />}
        title="Benchmark & Placement"
        subtitle="Inspect VRAM vs. RAM allocation and measure TTFT, prompt eval (pp), and generation (tg) throughput."
        actions={
          subTab === 'history' && history.length > 0 ? (
            <>
              <button onClick={handleExportCsv} className="bibulm-btn-secondary">
                <Download className="w-3.5 h-3.5" />
                <span>CSV</span>
              </button>
              <button onClick={handleExportJson} className="bibulm-btn-secondary">
                <Download className="w-3.5 h-3.5" />
                <span>JSON</span>
              </button>
            </>
          ) : undefined
        }
      />

      {/* Top Pinned Live Memory Placement & Component Breakdown Inspector */}
      <div className="px-6 py-4 border-b border-[var(--border-hairline)] bg-[var(--bg-canvas)] shrink-0">
        {activeRemoteModel ? (
          <div className="bibulm-card border-[var(--accent-border)] p-4 space-y-2.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2 flex-wrap min-w-0">
                <Link2 className="w-4 h-4 text-[var(--accent-text)] shrink-0" />
                <span className="text-[14px] font-semibold text-[var(--label-primary)] truncate">
                  {activeRemoteModel.model.display_name}
                </span>
                <span className="apple-badge apple-badge-accent font-medium">
                  Remote · {activeRemoteModel.peer.device_name}
                </span>
                {activeRemoteModel.model.quantization && (
                  <span className="apple-badge font-mono">
                    {activeRemoteModel.model.quantization}
                  </span>
                )}
                <code className="text-[11px] font-mono text-[var(--label-tertiary)]">
                  {activeRemoteModel.qualifiedId}
                </code>
              </div>
              <div className="flex items-center gap-3 text-[12px] font-mono tabular-nums">
                <span className="text-[var(--signal-green)] font-semibold">
                  Remote VRAM {formatMiB(activeRemoteModel.peer.vram_used_mib)}/
                  {formatMiB(activeRemoteModel.peer.vram_total_mib)}
                </span>
                <span className="text-[var(--label-secondary)]">
                  {activeRemoteModel.peer.gpu_summary}
                </span>
              </div>
            </div>
          </div>
        ) : activeModel && pt ? (
          <div className="bibulm-card p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2 flex-wrap min-w-0">
                <Gauge className="w-4 h-4 text-[var(--accent-text)] shrink-0" />
                <span className="text-[14px] font-semibold text-[var(--label-primary)] truncate">
                  {activeModel.display_name}
                </span>
                <PlacementBadge status={livePlacementStatus} />
                <MemorySourceBadge source={activeModel.breakdown?.source} />
                <span className="apple-badge font-mono">{activeModel.runtime_tag}</span>
                <span className="text-[11px] font-mono text-[var(--label-tertiary)] tabular-nums">
                  ctx {activeModel.load_config.ctx_size} · fa={activeModel.load_config.flash_attention}
                  {activeModel.load_config.cache_type_k &&
                    ` · kv=${activeModel.load_config.cache_type_k}`}
                  {activeModel.load_config.mtp_enabled && ' · mtp=on'}
                  {activeModel.load_config.mmproj_path &&
                    ` · vision=${activeModel.load_config.enable_vision ? 'on' : 'off'}`}
                </span>
              </div>
              <div className="flex items-center gap-2.5 text-[12px] font-mono flex-wrap tabular-nums">
                <span className="text-[var(--signal-green)] font-semibold">
                  VRAM {formatMiB(pt.totalGpu)}
                </span>
                <span className="text-[var(--signal-amber)] font-semibold">
                  RAM {formatMiB(pt.totalCpu)}
                </span>
                {activeModel.placement.offloaded_layers != null && (
                  <span className="text-[var(--label-secondary)]">
                    Layers {activeModel.placement.offloaded_layers}/
                    {activeModel.placement.total_layers ?? '?'}
                  </span>
                )}
                {activeModel.placement.cpu_moe_layers != null && (
                  <span className="text-[var(--signal-purple)]">
                    MoE CPU: {activeModel.placement.cpu_moe_layers}L
                  </span>
                )}
                {liveMetrics && (
                  <>
                    <span className="apple-badge text-[11px]">
                      Slots {liveMetrics.slots_processing}/{liveMetrics.slots_total}
                    </span>
                    {liveMetrics.kv_cache_usage_ratio != null && (
                      <span className="apple-badge apple-badge-teal text-[11px]">
                        KV {Math.round(liveMetrics.kv_cache_usage_ratio * 100)}%
                      </span>
                    )}
                    {liveMetrics.requests_deferred != null && liveMetrics.requests_deferred > 0 && (
                      <span className="apple-badge apple-badge-amber text-[11px]">
                        Queue {liveMetrics.requests_deferred}
                      </span>
                    )}
                  </>
                )}
              </div>
            </div>

            <PlacementBar placement={activeModel.placement} breakdown={activeModel.breakdown} />

            {liveSpillover && (
              <div className="flex items-center gap-2 rounded-lg bg-[var(--signal-red-soft)] border border-[var(--signal-red-border)] px-3 py-2 text-[12px] text-[var(--signal-red)]">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>
                  <strong>Windows WDDM Shared Memory Spillover Detected</strong> (
                  {formatBytes(telemetry?.shared_gpu_mem_used_bytes ?? 0)} shared GPU memory in
                  use). Reduce context length, quantize KV cache (Q8_0/Q4_0), disable Vision if
                  unused, or offload MoE experts explicitly via <code>--n-cpu-moe</code>!
                </span>
              </div>
            )}
          </div>
        ) : (
          <div className="bibulm-card border-dashed p-5 flex items-center justify-between gap-4">
            <div className="space-y-1">
              <div className="text-[14px] font-semibold text-[var(--label-primary)]">
                No Model Loaded — Memory Placement &amp; Speed Inspector Ready
              </div>
              <div className="text-[12px] text-[var(--label-secondary)]">
                Load any GGUF model to inspect exact GPU VRAM vs System RAM component breakdown and
                measure TTFT, prompt eval (pp), and generation (tg) speeds.
              </div>
            </div>
            <button onClick={onOpenQuickLoad} className="bibulm-btn-primary shrink-0">
              Load a Model (⌃L)
            </button>
          </div>
        )}
      </div>

      {/* Sub-Tabs (Apple Segmented Control) */}
      <div className="px-6 py-2.5 border-b border-[var(--border-hairline)] bg-[var(--bg-canvas)] flex items-center justify-between shrink-0">
        <div className="apple-segmented">
          {[
            { id: 'quick', label: 'Quick Speed Test', icon: <Zap className="w-3.5 h-3.5" /> },
            {
              id: 'playground',
              label: 'API Playground',
              icon: <MessageSquareCode className="w-3.5 h-3.5" />,
            },
            {
              id: 'matrix',
              label: 'Matrix (llama-bench)',
              icon: <BarChart3 className="w-3.5 h-3.5" />,
            },
            {
              id: 'history',
              label: `History & Compare (${history.length})`,
              icon: <History className="w-3.5 h-3.5" />,
            },
          ].map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setSubTab(t.id as 'quick' | 'playground' | 'matrix' | 'history')}
              className={`apple-segmented-item ${
                subTab === t.id ? 'apple-segmented-item-active' : ''
              }`}
            >
              {t.icon}
              <span>{t.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {subTab === 'playground' && (
          <PlaygroundTab
            activeModel={activeModel}
            activeRemoteModel={activeRemoteModel}
            sampling={sampling}
            reasoningEffort={reasoningEffort}
            onOpenQuickLoad={onOpenQuickLoad}
          />
        )}

        {subTab === 'matrix' && (
          <MatrixBenchTab
            activeModel={activeModel}
            localModels={localModels}
            runtimes={runtimes}
            settings={settings}
            onOpenQuickLoad={onOpenQuickLoad}
          />
        )}

        {subTab === 'quick' && (
          <>
            {/* Prompt Preset & Controls */}
            <div className="bibulm-card p-4 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap gap-1.5">
                  {PROMPT_PRESETS.map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => setPromptPreset(p)}
                      className={`h-7 px-2.5 rounded-lg text-[12px] border transition-colors cursor-pointer ${
                        promptPreset === p
                          ? 'bg-[var(--accent-soft)] border-[var(--accent-border)] text-[var(--accent-text)] font-medium'
                          : 'bg-[var(--bg-control)] border-[var(--border-hairline)] text-[var(--label-secondary)] hover:text-[var(--label-primary)]'
                      }`}
                    >
                      {p}
                    </button>
                  ))}
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-[12px] text-[var(--label-secondary)] flex items-center gap-1.5">
                    <span>Max tokens</span>
                    <input
                      type="number"
                      min={32}
                      max={16384}
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(parseInt(e.target.value || '512', 10))}
                      className="w-20 bibulm-input text-right font-mono tabular-nums"
                    />
                  </label>
                  <button
                    disabled={!targetModelId || runningQuick}
                    onClick={handleRunQuick}
                    className="bibulm-btn-primary px-4"
                  >
                    {runningQuick ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span>Running ({streamTokens} tok)…</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-3.5 h-3.5" />
                        <span>Run Speed Test</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {promptPreset === 'Custom' && (
                <textarea
                  rows={3}
                  value={customPrompt}
                  onChange={(e) => setCustomPrompt(e.target.value)}
                  placeholder="Enter custom benchmark prompt…"
                  className="w-full p-2.5 rounded-lg bg-[var(--bg-surface-inset)] border border-[var(--border-hairline)] text-[12px] text-[var(--label-primary)] placeholder-[var(--label-quaternary)] focus:outline-none focus:border-[var(--accent)]"
                />
              )}

              {quickError && (
                <div className="rounded-lg bg-[var(--signal-red-soft)] border border-[var(--signal-red-border)] px-3 py-2 text-[12px] text-[var(--signal-red)]">
                  {quickError}
                </div>
              )}
            </div>

            {/* KPI Cards */}
            {quickResult && (
              <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
                <KpiCard
                  label="Generation (tg)"
                  value={`${quickResult.record.generation_tok_s.toFixed(1)}`}
                  unit="tok/s"
                  accent="text-[var(--signal-green)]"
                />
                <KpiCard
                  label="Prompt Eval (pp)"
                  value={`${quickResult.record.prompt_eval_tok_s.toFixed(1)}`}
                  unit="tok/s"
                  accent="text-[var(--signal-teal)]"
                />
                <KpiCard
                  label="Time to First Token"
                  value={`${quickResult.record.ttft_ms.toFixed(0)}`}
                  unit="ms"
                  accent="text-[var(--accent-text)]"
                />
                <KpiCard
                  label="Tokens (Reason / Out)"
                  value={`${quickResult.record.reasoning_tokens} / ${quickResult.record.output_tokens}`}
                  unit={`pp ${quickResult.record.prompt_tokens}`}
                  accent="text-[var(--signal-purple)]"
                />
                <KpiCard
                  label="Peak VRAM / RAM"
                  value={formatMiB(quickResult.record.peak_vram_mib)}
                  unit={`RAM ${formatMiB(quickResult.record.peak_ram_mib)}`}
                  accent="text-[var(--signal-amber)]"
                />
              </div>
            )}

            {/* Live Stream / Result Inspection (Collapsible Thinking + Answer) */}
            {(streamReasoning || streamOutput || runningQuick) && (
              <div className="space-y-3">
                {streamReasoning && (
                  <div className="rounded-xl bg-[var(--signal-purple-soft)] border border-[var(--signal-purple-border)] overflow-hidden">
                    <button
                      type="button"
                      onClick={() => setShowThinking(!showThinking)}
                      className="w-full px-4 py-2.5 flex items-center justify-between text-[12px] font-medium text-[var(--signal-purple)] hover:opacity-85 cursor-pointer"
                    >
                      <span className="flex items-center gap-1.5">
                        <Brain className="w-3.5 h-3.5" />
                        <span>Reasoning / Thinking Stream</span>
                      </span>
                      {showThinking ? (
                        <ChevronDown className="w-4 h-4" />
                      ) : (
                        <ChevronRight className="w-4 h-4" />
                      )}
                    </button>
                    {showThinking && (
                      <pre className="px-4 pb-3 pt-1 text-[12px] font-mono text-[var(--label-primary)] whitespace-pre-wrap max-h-64 overflow-y-auto">
                        {streamReasoning}
                      </pre>
                    )}
                  </div>
                )}

                <div className="bibulm-card p-4 space-y-2">
                  <div className="text-[12px] font-semibold text-[var(--label-primary)] flex items-center justify-between">
                    <span>Model Output Preview</span>
                    {quickResult && (
                      <span className="text-[11px] font-mono text-[var(--label-tertiary)] tabular-nums">
                        Total {(quickResult.total_duration_ms / 1000).toFixed(2)}s
                      </span>
                    )}
                  </div>
                  <pre className="text-[13px] text-[var(--label-primary)] whitespace-pre-wrap font-sans leading-relaxed max-h-[340px] overflow-y-auto">
                    {streamOutput || (runningQuick ? 'Waiting for first token…' : '')}
                  </pre>
                </div>
              </div>
            )}
          </>
        )}

        {subTab === 'history' && (
          <div className="space-y-3">
            {history.length === 0 ? (
              <div className="bibulm-card py-14 text-center text-[12px] text-[var(--label-tertiary)]">
                <Activity className="w-8 h-8 mx-auto mb-2 text-[var(--label-quaternary)]" />
                No saved benchmark runs yet. Run a Quick Speed Test to build your comparison history.
              </div>
            ) : (
              <div className="bibulm-card overflow-x-auto">
                <table className="w-full text-[12px]">
                  <thead className="bg-[var(--bg-surface-inset)] text-[var(--label-secondary)] border-b border-[var(--border-hairline)]">
                    <tr>
                      <th className="text-left px-3.5 py-2.5 font-semibold">Model / Preset</th>
                      <th className="text-left px-3.5 py-2.5 font-semibold">Runtime</th>
                      <th className="text-left px-3.5 py-2.5 font-semibold">Placement</th>
                      <th className="text-right px-3.5 py-2.5 font-semibold">TTFT</th>
                      <th className="text-right px-3.5 py-2.5 font-semibold">pp tok/s</th>
                      <th className="text-left px-3.5 py-2.5 w-44 font-semibold">tg tok/s</th>
                      <th className="text-right px-3.5 py-2.5 font-semibold">VRAM / RAM</th>
                      <th className="px-2 py-2.5" />
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--border-hairline)]">
                    {history.map((run) => {
                      const t = placementTotals(run.placement);
                      const status =
                        t.totalGpu === 0 && t.totalCpu === 0
                          ? 'unknown'
                          : t.totalGpu === 0
                            ? 'cpu_only'
                            : t.cpuModel > 1 || t.cpuKv > 1
                              ? 'hybrid_gpu_cpu'
                              : run.placement.wddm_shared_spillover_detected
                                ? 'gpu_spillover_warning'
                                : 'gpu_only';

                      return (
                        <tr
                          key={run.id}
                          className="hover:bg-[var(--bg-control)] transition-colors"
                        >
                          <td className="px-3.5 py-2.5">
                            <div className="font-medium text-[var(--label-primary)]">
                              {run.model_id}
                            </div>
                            <div className="text-[11px] text-[var(--label-tertiary)]">
                              {run.prompt_preset} · {new Date(run.created_at).toLocaleString()}
                            </div>
                          </td>
                          <td className="px-3.5 py-2.5 font-mono text-[var(--label-secondary)]">
                            {run.runtime_tag}
                          </td>
                          <td className="px-3.5 py-2.5">
                            <PlacementBadge status={status} compact />
                          </td>
                          <td className="px-3.5 py-2.5 text-right font-mono text-[var(--label-primary)] tabular-nums">
                            {run.ttft_ms.toFixed(0)} ms
                          </td>
                          <td className="px-3.5 py-2.5 text-right font-mono text-[var(--signal-teal)] tabular-nums">
                            {run.prompt_eval_tok_s.toFixed(1)}
                          </td>
                          <td className="px-3.5 py-2.5">
                            <div className="flex items-center gap-2">
                              <div className="flex-1 h-2 rounded-full bg-[var(--bg-control)] overflow-hidden">
                                <div
                                  className="h-full bg-[var(--signal-green-fill)] rounded-full"
                                  style={{
                                    width: `${Math.min(100, (run.generation_tok_s / maxTgHistory) * 100)}%`,
                                  }}
                                />
                              </div>
                              <span className="font-mono font-semibold text-[var(--signal-green)] w-14 text-right tabular-nums">
                                {run.generation_tok_s.toFixed(1)}
                              </span>
                            </div>
                          </td>
                          <td className="px-3.5 py-2.5 text-right font-mono text-[11px] text-[var(--label-secondary)] tabular-nums">
                            {formatMiB(run.peak_vram_mib)} / {formatMiB(run.peak_ram_mib)}
                          </td>
                          <td className="px-2 py-2.5 text-right">
                            <button
                              type="button"
                              onClick={async () => {
                                await api.deleteBenchmarkRun(run.id);
                                loadHistory();
                              }}
                              className="apple-btn-ghost w-7 h-7 hover:text-[var(--signal-red)]"
                              aria-label="Delete benchmark run"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const KpiCard: React.FC<{
  label: string;
  value: string;
  unit: string;
  accent: string;
}> = ({ label, value, unit, accent }) => (
  <div className="bibulm-card p-3.5">
    <div className="text-[11px] text-[var(--label-secondary)] font-medium">{label}</div>
    <div className={`text-[22px] font-bold font-mono tabular-nums mt-1 tracking-tight ${accent}`}>
      {value}
    </div>
    <div className="text-[11px] text-[var(--label-tertiary)] font-mono tabular-nums mt-0.5">
      {unit}
    </div>
  </div>
);
