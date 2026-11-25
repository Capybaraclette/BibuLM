import React, { useEffect, useRef, useState } from 'react';
import {
  Check,
  Copy,
  ArrowUpFromLine,
  Key,
  Link2,
  Pause,
  Play,
  Plus,
  Search,
  Server,
  Settings2,
  Terminal,
  Trash2,
  X,
} from 'lucide-react';
import type {
  ApiTokenRecord,
  AppSettings,
  GatewayStatus,
  LinkStatus,
  LlamaServerMetrics,
  LoadedModelInfo,
  LocalModelEntry,
  ModelAliasRule,
  RemotePeer,
  ServerLogEntry,
} from '../types';
import { api, formatMiB, placementTotals } from '../api';
import { Toggle } from '../components/Toggle';
import { MemorySourceBadge, PlacementBadge, PlacementBar } from '../components/PlacementBadge';
import { PageHeader } from '../components/PageHeader';

interface ServerViewProps {
  status: GatewayStatus | null;
  settings: AppSettings;
  loadedModels: LoadedModelInfo[];
  localModels: LocalModelEntry[];
  linkStatus: LinkStatus | null;
  linkPeers: RemotePeer[];
  logs: ServerLogEntry[];
  onToggleServer: (start: boolean) => void;
  onSaveSettings: (next: AppSettings) => void;
  onOpenQuickLoad: () => void;
  onOpenLinkTab: () => void;
  onUnloadModel: (modelId: string) => void;
  onClearLogs: () => void;
}

const ENDPOINTS = [
  { method: 'GET', path: '/v1/models', desc: 'List loaded + local JIT-available models' },
  {
    method: 'POST',
    path: '/v1/chat/completions',
    desc: 'Chat completions (SSE stream, reasoning_content, tools, vision, JSON schema)',
  },
  {
    method: 'POST',
    path: '/v1/messages',
    desc: 'Anthropic Messages API (Claude Code compatible, x-api-key, SSE, tools, thinking)',
  },
  {
    method: 'POST',
    path: '/v1/messages/count_tokens',
    desc: 'Anthropic token estimate endpoint for Claude clients',
  },
  { method: 'POST', path: '/v1/completions', desc: 'Legacy text completions' },
  { method: 'POST', path: '/v1/embeddings', desc: 'Text embeddings (--embeddings)' },
  { method: 'POST', path: '/v1/responses', desc: 'OpenAI Responses API pass-through' },
];

export const ServerView: React.FC<ServerViewProps> = ({
  status,
  settings,
  loadedModels,
  localModels,
  linkStatus,
  linkPeers,
  logs,
  onToggleServer,
  onSaveSettings,
  onOpenQuickLoad,
  onOpenLinkTab,
  onUnloadModel,
  onClearLogs,
}) => {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [tokensModalOpen, setTokensModalOpen] = useState(false);
  const [tokens, setTokens] = useState<ApiTokenRecord[]>([]);
  const [newTokenName, setNewTokenName] = useState('');
  const [newlyCreatedToken, setNewlyCreatedToken] = useState<ApiTokenRecord | null>(null);
  const [copiedUrl, setCopiedUrl] = useState(false);
  const [copiedCurl, setCopiedCurl] = useState(false);
  const [copiedClaudeEnv, setCopiedClaudeEnv] = useState(false);
  const [modelMetrics, setModelMetrics] = useState<Record<string, LlamaServerMetrics>>({});
  const [aliasPattern, setAliasPattern] = useState('');
  const [aliasTarget, setAliasTarget] = useState('');

  // Logs controls
  const [logSearch, setLogSearch] = useState('');
  const [levelFilter, setLevelFilter] = useState<string>('ALL');
  const [autoScroll, setAutoScroll] = useState(true);
  const logEndRef = useRef<HTMLDivElement>(null);

  const loadTokens = async () => {
    try {
      setTokens(await api.listApiTokens());
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    loadTokens();
  }, []);

  useEffect(() => {
    if (loadedModels.length === 0) {
      setModelMetrics({});
      return;
    }
    let cancelled = false;
    const pollMetrics = async () => {
      const entries = await Promise.all(
        loadedModels.map(async (m) => {
          try {
            const metrics = await api.getLoadedModelMetrics(m.model_id);
            return [m.model_id, metrics] as const;
          } catch {
            return null;
          }
        }),
      );
      if (cancelled) return;
      const next: Record<string, LlamaServerMetrics> = {};
      for (const item of entries) {
        if (item) next[item[0]] = item[1];
      }
      setModelMetrics(next);
    };
    pollMetrics();
    const timer = setInterval(pollMetrics, 2500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [loadedModels.map((m) => m.model_id).join('|')]);

  useEffect(() => {
    if (autoScroll) {
      logEndRef.current?.scrollIntoView({ behavior: 'auto' });
    }
  }, [logs.length, autoScroll]);

  const updateServerCfg = (patch: Partial<AppSettings['server']>) => {
    onSaveSettings({
      ...settings,
      server: { ...settings.server, ...patch },
    });
  };

  const modelAliases: ModelAliasRule[] = settings.server.model_aliases ?? [];

  const addAliasRule = () => {
    const pattern = aliasPattern.trim();
    const target = aliasTarget.trim();
    if (!pattern || !target) return;
    const next = [
      ...modelAliases.filter((a) => a.pattern !== pattern),
      { pattern, target, enabled: true },
    ];
    updateServerCfg({ model_aliases: next });
    setAliasPattern('');
    setAliasTarget('');
  };

  const toggleAliasRule = (pattern: string) => {
    const next = modelAliases.map((a) =>
      a.pattern === pattern ? { ...a, enabled: !a.enabled } : a,
    );
    updateServerCfg({ model_aliases: next });
  };

  const removeAliasRule = (pattern: string) => {
    updateServerCfg({ model_aliases: modelAliases.filter((a) => a.pattern !== pattern) });
  };

  const reachableUrl = status?.reachable_url ?? `http://127.0.0.1:${settings.server.port}`;

  const copyUrl = () => {
    navigator.clipboard.writeText(reachableUrl);
    setCopiedUrl(true);
    setTimeout(() => setCopiedUrl(false), 1500);
  };

  const copyCurl = () => {
    const sampleModel = loadedModels[0]?.model_id ?? 'your-model-id';
    const cmd = `curl ${reachableUrl}/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"${sampleModel}","messages":[{"role":"user","content":"Hello!"}],"stream":true}'`;
    navigator.clipboard.writeText(cmd);
    setCopiedCurl(true);
    setTimeout(() => setCopiedCurl(false), 1500);
  };

  const copyClaudeEnv = () => {
    const sampleModel = loadedModels[0]?.model_id ?? modelAliases[0]?.pattern ?? 'claude-sonnet-4-5';
    const snippet = `ANTHROPIC_BASE_URL=${reachableUrl} ANTHROPIC_API_KEY=bibulm claude --model ${sampleModel}`;
    navigator.clipboard.writeText(snippet);
    setCopiedClaudeEnv(true);
    setTimeout(() => setCopiedClaudeEnv(false), 1500);
  };

  const filteredLogs = logs.filter((l) => {
    if (levelFilter !== 'ALL' && l.level !== levelFilter) return false;
    const q = logSearch.trim().toLowerCase();
    if (
      q &&
      !l.message.toLowerCase().includes(q) &&
      !l.source.toLowerCase().includes(q)
    ) {
      return false;
    }
    return true;
  });

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Unified Header + Server Controls */}
      <PageHeader
        icon={<Server className="w-4 h-4" />}
        title="Model Server"
        subtitle="OpenAI-compatible local gateway with SSE streaming, JIT model loading, and live console."
        actions={
          <>
            <div className="h-8 px-3 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-hairline)] flex items-center gap-2.5 text-[12px]">
              <span className="font-medium">
                <span
                  className={
                    status?.running ? 'text-[var(--signal-green)]' : 'text-[var(--label-tertiary)]'
                  }
                >
                  {status?.running ? 'Running' : 'Stopped'}
                </span>
              </span>
              <Toggle
                size="sm"
                color="green"
                checked={Boolean(status?.running)}
                onChange={(on) => onToggleServer(on)}
              />
            </div>

            <button
              type="button"
              onClick={() => setSettingsOpen(!settingsOpen)}
              className={`h-8 px-3 rounded-lg border text-[12px] font-medium flex items-center gap-1.5 transition-colors cursor-pointer ${
                settingsOpen
                  ? 'bg-[var(--accent-soft)] border-[var(--accent-border)] text-[var(--accent-text)]'
                  : 'bg-[var(--bg-surface)] hover:bg-[var(--bg-control-hover)] border-[var(--border-hairline)] text-[var(--label-primary)]'
              }`}
            >
              <Settings2 className="w-3.5 h-3.5" />
              <span>Settings</span>
            </button>

            <div className="flex items-center gap-1.5 text-[12px]">
              <code className="h-8 px-2.5 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-hairline)] text-[var(--label-primary)] font-mono flex items-center tabular-nums">
                {reachableUrl}
              </code>
              <button
                type="button"
                onClick={copyUrl}
                className="bibulm-btn-secondary h-8 w-8 px-0"
                title="Copy server URL"
                aria-label="Copy server URL"
              >
                {copiedUrl ? (
                  <Check className="w-3.5 h-3.5 text-[var(--signal-green)]" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>
            </div>

            <button type="button" onClick={onOpenQuickLoad} className="bibulm-btn-primary h-8">
              <Plus className="w-3.5 h-3.5" />
              <span>Load Model</span>
            </button>
          </>
        }
      >
        {/* Server Settings Liquid Glass Popover */}
        {settingsOpen && (
          <div
            className="absolute top-[60px] right-6 w-[360px] rounded-2xl apple-glass border p-4 space-y-3.5 text-[12px] z-40"
            style={{ boxShadow: 'var(--shadow-popover)' }}
          >
            <div className="flex items-center justify-between pb-2.5 border-b border-[var(--border-hairline)]">
              <span className="font-semibold text-[var(--label-primary)]">Server Settings</span>
              <button
                type="button"
                onClick={() => setSettingsOpen(false)}
                className="apple-btn-ghost w-7 h-7"
                aria-label="Close server settings"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-[var(--label-primary)]">Server Port</span>
              <input
                type="number"
                min={1024}
                max={65535}
                value={settings.server.port}
                onChange={(e) =>
                  updateServerCfg({ port: Math.max(1024, parseInt(e.target.value || '1234', 10)) })
                }
                className="w-24 bibulm-input h-7 text-right font-mono tabular-nums"
              />
            </div>

            <div className="space-y-2 pt-2 border-t border-[var(--border-hairline)]">
              <div className="flex items-center justify-between">
                <span className="text-[var(--label-primary)]">Require Authentication</span>
                <Toggle
                  size="sm"
                  color="green"
                  checked={settings.server.require_auth}
                  onChange={(v) => updateServerCfg({ require_auth: v })}
                />
              </div>
              <div className="flex items-center justify-between text-[var(--label-secondary)]">
                <span>
                  Active API Keys:{' '}
                  <strong className="apple-badge font-mono tabular-nums">
                    {tokens.filter((t) => t.enabled).length}
                  </strong>
                </span>
                <button
                  type="button"
                  onClick={() => setTokensModalOpen(true)}
                  className="bibulm-btn-secondary h-7 px-2.5 text-[11px]"
                >
                  <Key className="w-3 h-3" /> Manage Tokens
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-[var(--border-hairline)]">
              <span className="text-[var(--label-primary)]">Serve on Local Network (0.0.0.0)</span>
              <Toggle
                size="sm"
                color="green"
                checked={settings.server.serve_on_lan}
                onChange={(v) => updateServerCfg({ serve_on_lan: v })}
              />
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-[var(--border-hairline)]">
              <span className="text-[var(--label-primary)]">Enable CORS</span>
              <Toggle
                size="sm"
                color="green"
                checked={settings.server.enable_cors}
                onChange={(v) => updateServerCfg({ enable_cors: v })}
              />
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-[var(--border-hairline)]">
              <span className="text-[var(--label-primary)]">Just-in-Time Model Loading</span>
              <Toggle
                size="sm"
                color="green"
                checked={settings.server.jit_loading}
                onChange={(v) => updateServerCfg({ jit_loading: v })}
              />
            </div>

            <div className="space-y-2 pt-2 border-t border-[var(--border-hairline)]">
              <div className="flex items-center justify-between">
                <span className="text-[var(--label-primary)]">Auto unload unused JIT models</span>
                <Toggle
                  size="sm"
                  color="green"
                  checked={settings.server.auto_unload_idle}
                  onChange={(v) => updateServerCfg({ auto_unload_idle: v })}
                />
              </div>
              <div className="flex items-center justify-between text-[var(--label-secondary)]">
                <span>Max idle TTL</span>
                <div className="flex items-center gap-1.5">
                  <input
                    type="number"
                    min={1}
                    max={1440}
                    value={settings.server.max_idle_ttl_minutes}
                    onChange={(e) =>
                      updateServerCfg({
                        max_idle_ttl_minutes: Math.max(1, parseInt(e.target.value || '60', 10)),
                      })
                    }
                    className="w-16 bibulm-input h-7 text-right font-mono tabular-nums"
                  />
                  <span>minutes</span>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-[var(--border-hairline)]">
              <span className="text-[var(--label-primary)]">Only Keep Last JIT Loaded Model</span>
              <Toggle
                size="sm"
                color="green"
                checked={settings.server.only_keep_last_jit_model}
                onChange={(v) => updateServerCfg({ only_keep_last_jit_model: v })}
              />
            </div>
          </div>
        )}
      </PageHeader>

      {/* Main Split: Loaded Models + Endpoints (top) & Live Server Logs (bottom) */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto p-6 space-y-5">
          {linkStatus?.state === 'connected' && settings.link.expose_remote_in_openai_api && (
            <div className="rounded-xl bg-[var(--accent-soft)] border border-[var(--accent-border)] px-4 py-2.5 flex flex-wrap items-center justify-between gap-2 text-[12px]">
              <div className="flex items-center gap-2 text-[var(--label-primary)] min-w-0">
                <Link2 className="w-4 h-4 text-[var(--accent-text)] shrink-0" />
                <span className="truncate">
                  <strong>Tailscale Link active</strong> ({linkStatus.tailscale_ip ?? linkStatus.hostname}) —{' '}
                  {linkPeers.filter((p) => p.online && p.expose_local_models).length} peer(s)
                  bridged into <code className="font-mono text-[var(--accent-text)]">{reachableUrl}/v1</code>
                </span>
              </div>
              <button
                type="button"
                onClick={onOpenLinkTab}
                className="bibulm-btn-primary h-7 text-[11px] shrink-0"
              >
                Manage Link
              </button>
            </div>
          )}

          {/* Loaded Models Dashboard */}
          <div className="space-y-2.5">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--label-tertiary)]">
              Loaded Models ({loadedModels.length})
            </div>
            {loadedModels.length === 0 ? (
              <div className="bibulm-card py-10 text-center space-y-1.5">
                <div className="text-[13px] text-[var(--label-secondary)]">
                  Use Bibulm from IDEs, Python SDKs, or curl via the OpenAI-compatible API.
                </div>
                <div className="text-[11px] text-[var(--label-tertiary)]">
                  Press <kbd className="apple-kbd">⌃L</kbd> to load a model or let JIT loading boot
                  models on demand.
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-3">
                {loadedModels.map((m) => {
                  const pt = placementTotals(m.placement, m.breakdown);
                  const liveMetrics = modelMetrics[m.model_id];
                  const kvPct =
                    liveMetrics?.kv_cache_usage_ratio != null
                      ? Math.round(liveMetrics.kv_cache_usage_ratio * 100)
                      : null;
                  return (
                    <div key={m.model_id} className="bibulm-card p-4 space-y-3">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="space-y-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="w-2 h-2 rounded-full bg-[var(--signal-green-fill)]" />
                            <span className="text-[13px] font-semibold text-[var(--label-primary)]">
                              {m.display_name}
                            </span>
                            <PlacementBadge status={m.placement_status} compact />
                            <MemorySourceBadge source={m.breakdown?.source} />
                            {m.is_jit && (
                              <span className="apple-badge apple-badge-accent text-[10px]">
                                JIT
                              </span>
                            )}
                            {liveMetrics && (
                              <div className="flex items-center gap-1.5 text-[10px] font-mono tabular-nums">
                                <span className="apple-badge">
                                  Slots {liveMetrics.slots_processing}/{liveMetrics.slots_total}
                                </span>
                                {kvPct !== null && (
                                  <span
                                    className={`apple-badge ${
                                      kvPct >= 85 ? 'apple-badge-red' : 'apple-badge-teal'
                                    }`}
                                  >
                                    KV {kvPct}%
                                  </span>
                                )}
                                {liveMetrics.requests_deferred != null &&
                                  liveMetrics.requests_deferred > 0 && (
                                    <span className="apple-badge apple-badge-amber">
                                      Queue {liveMetrics.requests_deferred}
                                    </span>
                                  )}
                                {liveMetrics.predicted_tok_s != null &&
                                  liveMetrics.predicted_tok_s > 0 && (
                                    <span className="apple-badge apple-badge-green">
                                      {liveMetrics.predicted_tok_s.toFixed(1)} tok/s
                                    </span>
                                  )}
                              </div>
                            )}
                          </div>
                          <div className="text-[11px] text-[var(--label-secondary)] font-mono flex flex-wrap gap-x-3 gap-y-0.5 tabular-nums">
                            <span>id: {m.alias ?? m.model_id}</span>
                            <span>backend: 127.0.0.1:{m.port}</span>
                            <span>runtime: {m.runtime_tag}</span>
                            <span className="text-[var(--signal-green)]">
                              VRAM {formatMiB(pt.totalGpu)}
                            </span>
                            <span className="text-[var(--signal-amber)]">
                              RAM {formatMiB(pt.totalCpu)}
                            </span>
                            <span>reqs: {m.active_requests}</span>
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={() => onUnloadModel(m.model_id)}
                          className="bibulm-btn-secondary hover:text-[var(--signal-red)] hover:border-[var(--signal-red-border)]"
                        >
                          <ArrowUpFromLine className="w-3.5 h-3.5" /> Eject
                        </button>
                      </div>
                      <PlacementBar placement={m.placement} breakdown={m.breakdown} />
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Model Aliases & Routing */}
          <div className="bibulm-card p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <div className="text-[13px] font-semibold text-[var(--label-primary)]">
                  Model Aliases &amp; Routing
                </div>
                <div className="text-[11px] text-[var(--label-secondary)]">
                  Map external client names (e.g. <code className="font-mono">claude-*</code>,{' '}
                  <code className="font-mono">gpt-4o</code>) to a local model ID.
                </div>
              </div>
              <div className="flex items-center gap-2 text-[11px] text-[var(--label-secondary)]">
                <span>Fallback to first loaded if unknown</span>
                <Toggle
                  size="sm"
                  color="green"
                  checked={settings.server.fallback_to_first_loaded ?? false}
                  onChange={(v) => updateServerCfg({ fallback_to_first_loaded: v })}
                />
              </div>
            </div>

            <div className="flex flex-wrap gap-2 text-[12px]">
              <input
                value={aliasPattern}
                onChange={(e) => setAliasPattern(e.target.value)}
                placeholder="Client model pattern (e.g. claude-* or gpt-4o)"
                className="flex-1 min-w-[180px] bibulm-input font-mono text-[11px]"
              />
              <input
                list="bibulm-local-model-targets"
                value={aliasTarget}
                onChange={(e) => setAliasTarget(e.target.value)}
                placeholder="Target model ID"
                className="flex-1 min-w-[180px] bibulm-input font-mono text-[11px]"
              />
              <datalist id="bibulm-local-model-targets">
                {loadedModels.map((m) => (
                  <option key={`loaded-${m.model_id}`} value={m.model_id} />
                ))}
                {localModels.map((m) => (
                  <option key={`local-${m.id}`} value={m.id} />
                ))}
              </datalist>
              <button
                type="button"
                onClick={addAliasRule}
                disabled={!aliasPattern.trim() || !aliasTarget.trim()}
                className="bibulm-btn-primary text-[11px]"
              >
                <Plus className="w-3 h-3" /> Add Alias
              </button>
            </div>

            {modelAliases.length > 0 && (
              <div className="divide-y divide-[var(--border-hairline)] border border-[var(--border-hairline)] rounded-xl bg-[var(--bg-surface-inset)]">
                {modelAliases.map((rule) => (
                  <div
                    key={rule.pattern}
                    className="px-3 py-2 flex items-center justify-between gap-2 text-[11px]"
                  >
                    <div className="flex items-center gap-2 font-mono min-w-0 truncate">
                      <span className="text-[var(--accent-text)] font-medium">{rule.pattern}</span>
                      <span className="text-[var(--label-tertiary)]">-&gt;</span>
                      <span className="text-[var(--label-primary)] truncate">{rule.target}</span>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <Toggle
                        size="sm"
                        color="green"
                        checked={rule.enabled}
                        onChange={() => toggleAliasRule(rule.pattern)}
                      />
                      <button
                        type="button"
                        onClick={() => removeAliasRule(rule.pattern)}
                        className="apple-btn-ghost w-7 h-7 hover:text-[var(--signal-red)]"
                        title="Remove alias"
                        aria-label="Remove alias"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Supported Endpoints */}
          <div className="bibulm-card p-4 space-y-2.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-[13px] font-semibold text-[var(--label-primary)]">
                Supported OpenAI &amp; Anthropic Endpoints
              </span>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={copyClaudeEnv}
                  className="bibulm-btn-secondary h-7 px-2.5 text-[11px]"
                >
                  {copiedClaudeEnv ? (
                    <>
                      <Check className="w-3 h-3 text-[var(--signal-green)]" /> Copied Claude env
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" /> Copy Claude Code env
                    </>
                  )}
                </button>
                <button
                  type="button"
                  onClick={copyCurl}
                  className="bibulm-btn-secondary h-7 px-2.5 text-[11px]"
                >
                  {copiedCurl ? (
                    <>
                      <Check className="w-3 h-3 text-[var(--signal-green)]" /> Copied curl
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" /> Copy curl example
                    </>
                  )}
                </button>
              </div>
            </div>
            <div className="divide-y divide-[var(--border-hairline)]">
              {ENDPOINTS.map((ep) => (
                <div
                  key={ep.path}
                  className="py-2 flex items-center justify-between gap-4 text-[12px]"
                >
                  <div className="flex items-center gap-2.5 font-mono">
                    <span
                      className={`apple-badge text-[10px] font-bold ${
                        ep.method === 'GET' ? 'apple-badge-green' : 'apple-badge-accent'
                      }`}
                    >
                      {ep.method}
                    </span>
                    <span className="text-[var(--label-primary)]">{ep.path}</span>
                  </div>
                  <span className="text-[11px] text-[var(--label-secondary)] truncate">
                    {ep.desc}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Live Server Logs Console (bottom split, keeping dark terminal contrast) */}
        <div className="h-[260px] shrink-0 border-t border-[var(--border-hairline)] bg-[var(--bg-terminal)] flex flex-col">
          <div className="px-4 py-2 border-b border-white/10 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-[12px] font-semibold text-[#e5e5ea]">
              <Terminal className="w-3.5 h-3.5 text-[#30d158]" />
              <span>Server Logs</span>
              <div className="flex gap-1 ml-2">
                {['ALL', 'INFO', 'LLAMA', 'WARN', 'ERROR'].map((lvl) => (
                  <button
                    key={lvl}
                    type="button"
                    onClick={() => setLevelFilter(lvl)}
                    className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors cursor-pointer ${
                      levelFilter === lvl
                        ? 'bg-white/15 text-white'
                        : 'text-[#8e8e93] hover:text-[#e5e5ea]'
                    }`}
                  >
                    {lvl}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <div className="relative">
                <Search className="w-3 h-3 text-[#8e8e93] absolute left-2 top-1/2 -translate-y-1/2 pointer-events-none" />
                <input
                  value={logSearch}
                  onChange={(e) => setLogSearch(e.target.value)}
                  placeholder="Filter logs…"
                  className="h-6 w-40 pl-6 pr-2 rounded-md bg-white/8 border border-white/12 text-[11px] text-[#f5f5f7] placeholder-[#8e8e93] focus:outline-none focus:border-[#0a84ff]"
                />
              </div>
              <button
                type="button"
                onClick={() => setAutoScroll(!autoScroll)}
                className={`p-1.5 rounded-md text-xs cursor-pointer transition-colors ${
                  autoScroll
                    ? 'text-[#30d158] bg-[#30d158]/15'
                    : 'text-[#8e8e93] hover:text-[#f5f5f7]'
                }`}
                title={autoScroll ? 'Pause auto-scroll' : 'Resume auto-scroll'}
                aria-label={autoScroll ? 'Pause auto-scroll' : 'Resume auto-scroll'}
              >
                {autoScroll ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              </button>
              <button
                type="button"
                onClick={() =>
                  navigator.clipboard.writeText(
                    filteredLogs
                      .map((l) => `${l.timestamp} [${l.level}] [${l.source}] ${l.message}`)
                      .join('\n'),
                  )
                }
                className="p-1.5 rounded-md text-[#a1a1a6] hover:text-white cursor-pointer"
                title="Copy visible logs"
                aria-label="Copy visible logs"
              >
                <Copy className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={onClearLogs}
                className="p-1.5 rounded-md text-[#a1a1a6] hover:text-[#ff453a] cursor-pointer"
                title="Clear logs"
                aria-label="Clear logs"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-4 py-2 font-mono text-[11px] leading-relaxed space-y-0.5 tabular-nums">
            {filteredLogs.length === 0 ? (
              <div className="text-[#636366] py-4 text-center">No log messages yet.</div>
            ) : (
              filteredLogs.map((entry, idx) => {
                const lvlCls =
                  entry.level === 'ERROR'
                    ? 'text-[#ff453a]'
                    : entry.level === 'WARN'
                      ? 'text-[#ff9f0a]'
                      : entry.level === 'LLAMA'
                        ? 'text-[#64d2ff]'
                        : 'text-[#30d158]';
                return (
                  <div key={idx} className="break-all">
                    <span className="text-[#636366]">{entry.timestamp}</span>{' '}
                    <span className={lvlCls}>[{entry.level}]</span>{' '}
                    <span className="text-[#8e8e93]">[{entry.source}]</span>{' '}
                    <span className="text-[#e5e5ea]">{entry.message}</span>
                  </div>
                );
              })
            )}
            <div ref={logEndRef} />
          </div>
        </div>
      </div>

      {/* API Tokens Manager Modal (macOS Sheet) */}
      {tokensModalOpen && (
        <div className="apple-modal-backdrop">
          <div className="w-[520px] max-w-full apple-sheet text-[12px]">
            <div className="px-4 py-3 apple-glass border-b flex items-center justify-between">
              <span className="font-semibold text-[var(--label-primary)] flex items-center gap-2">
                <Key className="w-4 h-4 text-[var(--accent-text)]" /> Bearer API Keys
              </span>
              <button
                type="button"
                onClick={() => {
                  setTokensModalOpen(false);
                  setNewlyCreatedToken(null);
                }}
                className="apple-btn-ghost w-7 h-7"
                aria-label="Close tokens modal"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-4 space-y-3 bg-[var(--bg-surface)]">
              {newlyCreatedToken && (
                <div className="p-3 rounded-xl bg-[var(--signal-green-soft)] border border-[var(--signal-green-border)] space-y-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[var(--signal-green)] font-semibold">
                      Key created for &ldquo;{newlyCreatedToken.name}&rdquo; — copy it now!
                    </span>
                    <button
                      type="button"
                      onClick={() => setNewlyCreatedToken(null)}
                      className="apple-btn-ghost w-6 h-6"
                      aria-label="Dismiss key notice"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <div className="text-[11px] text-[var(--label-secondary)]">
                    For security, Bibulm stores only a SHA-256 hash in SQLite. You won&apos;t be
                    able to reveal this raw key again.
                  </div>
                  <div className="flex items-center gap-2">
                    <code className="flex-1 px-2.5 py-1.5 rounded-lg bg-[var(--bg-terminal)] border border-[var(--signal-green-border)] text-[#30d158] font-mono text-[11px] break-all">
                      {newlyCreatedToken.token}
                    </code>
                    <button
                      type="button"
                      onClick={() => navigator.clipboard.writeText(newlyCreatedToken.token)}
                      className="bibulm-btn-primary h-7 px-2.5 shrink-0"
                    >
                      <Copy className="w-3 h-3" /> Copy
                    </button>
                  </div>
                </div>
              )}

              <div className="flex gap-2">
                <input
                  value={newTokenName}
                  onChange={(e) => setNewTokenName(e.target.value)}
                  onKeyDown={async (e) => {
                    if (e.key === 'Enter' && newTokenName.trim()) {
                      const created = await api.createApiToken(newTokenName.trim());
                      setNewlyCreatedToken(created);
                      setNewTokenName('');
                      loadTokens();
                    }
                  }}
                  placeholder="New token label (e.g. VS Code / Cursor / Claude Code)…"
                  className="flex-1 bibulm-input"
                />
                <button
                  type="button"
                  onClick={async () => {
                    if (!newTokenName.trim()) return;
                    const created = await api.createApiToken(newTokenName.trim());
                    setNewlyCreatedToken(created);
                    setNewTokenName('');
                    loadTokens();
                  }}
                  className="bibulm-btn-primary"
                >
                  Create Key
                </button>
              </div>

              <div className="max-h-64 overflow-y-auto divide-y divide-[var(--border-hairline)] border border-[var(--border-hairline)] rounded-xl bg-[var(--bg-surface-inset)]">
                {tokens.length === 0 ? (
                  <div className="py-6 text-center text-[var(--label-tertiary)]">
                    No API keys created yet.
                  </div>
                ) : (
                  tokens.map((t) => (
                    <div key={t.id} className="p-2.5 flex items-center justify-between gap-2">
                      <div className="min-w-0">
                        <div className="font-medium text-[var(--label-primary)]">{t.name}</div>
                        <div className="font-mono text-[11px] text-[var(--label-secondary)] truncate">
                          {t.token}
                        </div>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <Toggle
                          size="sm"
                          color="green"
                          checked={t.enabled}
                          onChange={async (v) => {
                            await api.setApiTokenEnabled(t.id, v);
                            loadTokens();
                          }}
                        />
                        <button
                          type="button"
                          onClick={async () => {
                            await api.deleteApiToken(t.id);
                            loadTokens();
                          }}
                          className="apple-btn-ghost w-7 h-7 hover:text-[var(--signal-red)]"
                          title="Delete token"
                          aria-label="Delete token"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
