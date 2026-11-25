import React, { useEffect, useMemo, useState } from 'react';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { openUrl } from '@tauri-apps/plugin-opener';
import {
  Brain,
  Check,
  Copy,
  Download,
  Eye,
  Heart,
  Laptop,
  Layers,
  Loader2,
  Lock,
  Search,
  Server,
  Wrench,
} from 'lucide-react';
import type { HfModelDetail, HfModelSummary, HfQuantOption, HfSortOrder, RemotePeer } from '../types';
import { api, formatBytes, formatMiB, formatRelativeDate } from '../api';
import { FitVerdictBadge } from '../components/PlacementBadge';
import { PageHeader } from '../components/PageHeader';

interface SearchViewProps {
  linkPeers: RemotePeer[];
  selectedPeerId: string | null;
  onSelectPeerId: (id: string | null) => void;
  onDownloadStarted: () => void;
  onRemoteDownloadStarted: (peerId: string) => void;
}

const QUICK_TAGS = [
  { label: 'Trending GGUF', query: '' },
  { label: 'Qwen3', query: 'Qwen3' },
  { label: 'DeepSeek', query: 'DeepSeek' },
  { label: 'Gemma', query: 'gemma' },
  { label: 'Llama 3', query: 'Llama-3' },
  { label: 'MoE', query: 'MoE' },
  { label: 'Unsloth', query: 'unsloth' },
  { label: 'Bartowski', query: 'bartowski' },
];

export const SearchView: React.FC<SearchViewProps> = ({
  linkPeers,
  selectedPeerId,
  onSelectPeerId,
  onDownloadStarted,
  onRemoteDownloadStarted,
}) => {
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState<HfSortOrder>('best_match');
  const [results, setResults] = useState<HfModelSummary[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  const [selectedRepo, setSelectedRepo] = useState<string | null>(null);
  const [detail, setDetail] = useState<HfModelDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [selectedMmprojPath, setSelectedMmprojPath] = useState<string>('');
  const [copiedRepo, setCopiedRepo] = useState(false);
  const [queuedQuantId, setQueuedQuantId] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const manageablePeers = linkPeers.filter((p) => p.online && p.allow_remote_manage);
  const selectedPeer = selectedPeerId
    ? manageablePeers.find((p) => p.node_id === selectedPeerId) ?? null
    : null;

  useEffect(() => {
    if (selectedPeerId && !selectedPeer) {
      onSelectPeerId(null);
    }
  }, [selectedPeerId, selectedPeer, onSelectPeerId]);

  const runSearch = async (q: string, s: HfSortOrder) => {
    setSearching(true);
    setSearchError(null);
    try {
      const list = await api.searchHfModels(q, s);
      setResults(list);
      if (list.length > 0 && !selectedRepo) {
        selectRepo(list[0].repo_id);
      }
    } catch (e) {
      setSearchError(String(e));
    } finally {
      setSearching(false);
    }
  };

  const selectRepo = async (repoId: string, peerOverride?: RemotePeer | null) => {
    const target = peerOverride === undefined ? selectedPeer : peerOverride;
    setSelectedRepo(repoId);
    setDetail(null);
    setDetailError(null);
    setDownloadError(null);
    setLoadingDetail(true);
    setSelectedMmprojPath('');
    try {
      const vramBytes = target ? Math.round(target.vram_total_mib * 1024 * 1024) : null;
      const ramBytes = target ? Math.round(target.ram_total_mib * 1024 * 1024) : null;
      const d = await api.getHfModelDetail(repoId, vramBytes, ramBytes);
      setDetail(d);
      if (d.mmproj_files.length > 0) {
        setSelectedMmprojPath(d.mmproj_files[0].path);
      }
    } catch (e) {
      setDetailError(String(e));
    } finally {
      setLoadingDetail(false);
    }
  };

  useEffect(() => {
    runSearch('', 'best_match');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (selectedRepo) {
      selectRepo(selectedRepo, selectedPeer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPeer?.node_id]);

  const handleDownloadQuant = async (quant: HfQuantOption) => {
    if (!detail) return;
    setDownloadError(null);
    const files = [...quant.files];
    if (selectedMmprojPath) {
      const mm = detail.mmproj_files.find((f) => f.path === selectedMmprojPath);
      if (mm) files.push(mm);
    }
    const title = `${detail.summary.repo_id} (${quant.quantization})`;
    try {
      if (selectedPeer) {
        await api.remoteEnqueueModelDownload(
          selectedPeer.node_id,
          title,
          detail.summary.repo_id,
          files,
        );
        setQueuedQuantId(quant.id);
        setTimeout(() => setQueuedQuantId(null), 2000);
        onRemoteDownloadStarted(selectedPeer.node_id);
      } else {
        await api.enqueueModelDownload(title, detail.summary.repo_id, files);
        setQueuedQuantId(quant.id);
        setTimeout(() => setQueuedQuantId(null), 2000);
        onDownloadStarted();
      }
    } catch (e) {
      setDownloadError(String(e));
    }
  };

  const readmeHtml = useMemo(() => {
    if (!detail?.readme_markdown) return '';
    try {
      const rawHtml = marked.parse(detail.readme_markdown, { async: false }) as string;
      return DOMPurify.sanitize(rawHtml, {
        USE_PROFILES: { html: true },
        FORBID_TAGS: ['style', 'iframe', 'object', 'embed', 'form'],
        FORBID_ATTR: ['onerror', 'onload', 'onclick', 'style'],
      });
    } catch {
      return '';
    }
  }, [detail?.readme_markdown]);

  const handleReadmeClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const anchor = (e.target as HTMLElement | null)?.closest('a');
    if (!anchor) return;
    const href = anchor.getAttribute('href');
    if (!href) return;
    e.preventDefault();
    if (/^https?:\/\//i.test(href)) {
      openUrl(href).catch(() => {});
    }
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <PageHeader
        icon={<Search className="w-4 h-4" />}
        title="Model Search"
        subtitle="Discover HuggingFace GGUF repositories with live VRAM & MoE fit verdicts and resumable downloads."
      />
      {manageablePeers.length > 0 && (
        <div className="px-6 py-2.5 border-b border-[var(--border-hairline)] bg-[var(--bg-surface-inset)] flex flex-wrap items-center gap-2 text-[12px] shrink-0">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--label-tertiary)] mr-1">
            Download Target
          </span>
          <button
            type="button"
            onClick={() => onSelectPeerId(null)}
            className={`h-7 px-2.5 rounded-lg font-medium flex items-center gap-1.5 transition-colors cursor-pointer ${
              !selectedPeer
                ? 'bg-[var(--accent)] text-white shadow-sm'
                : 'bg-[var(--bg-surface)] border border-[var(--border-hairline)] hover:bg-[var(--bg-control-hover)] text-[var(--label-primary)]'
            }`}
          >
            <Laptop className="w-3.5 h-3.5" />
            <span>This Machine</span>
          </button>
          {manageablePeers.map((peer) => {
            const active = selectedPeer?.node_id === peer.node_id;
            return (
              <button
                key={peer.node_id}
                type="button"
                onClick={() => onSelectPeerId(peer.node_id)}
                className={`h-7 px-2.5 rounded-lg font-medium flex items-center gap-1.5 transition-colors cursor-pointer ${
                  active
                    ? 'bg-[var(--accent)] text-white shadow-sm'
                    : 'bg-[var(--bg-surface)] border border-[var(--border-hairline)] hover:bg-[var(--bg-control-hover)] text-[var(--label-primary)]'
                }`}
              >
                {peer.is_headless ? (
                  <Server className="w-3.5 h-3.5" />
                ) : (
                  <Laptop className="w-3.5 h-3.5" />
                )}
                <span>{peer.device_name}</span>
                <span className="text-[10px] font-mono opacity-80 tabular-nums">
                  ({formatMiB(peer.vram_total_mib)} VRAM / {formatMiB(peer.ram_total_mib)} RAM)
                </span>
              </button>
            );
          })}
        </div>
      )}
      <div className="flex-1 flex overflow-hidden">
      {/* Left Pane: macOS Master-Detail Sidebar List */}
      <div className="w-[380px] shrink-0 border-r border-[var(--border-hairline)] bg-[var(--bg-sidebar)] flex flex-col overflow-hidden">
        <div className="p-3 space-y-2.5 border-b border-[var(--border-hairline)]">
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-[var(--label-tertiary)] absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && runSearch(query, sort)}
              placeholder="Search HuggingFace GGUF models…"
              className="w-full bibulm-input h-9 pl-9 pr-3"
            />
          </div>

          <div className="flex items-center justify-between gap-2">
            <div className="flex flex-wrap gap-1">
              {QUICK_TAGS.slice(0, 4).map((t) => (
                <button
                  key={t.label}
                  type="button"
                  onClick={() => {
                    setQuery(t.query);
                    runSearch(t.query, sort);
                  }}
                  className="px-2 py-0.5 rounded-md bg-[var(--bg-surface)] border border-[var(--border-hairline)] hover:bg-[var(--bg-control-hover)] text-[11px] text-[var(--label-secondary)] hover:text-[var(--label-primary)] cursor-pointer transition-colors"
                >
                  {t.label}
                </button>
              ))}
            </div>
            <select
              value={sort}
              onChange={(e) => {
                const nextSort = e.target.value as HfSortOrder;
                setSort(nextSort);
                runSearch(query, nextSort);
              }}
              className="apple-select h-7 text-[11px]"
            >
              <option value="best_match">Best Match</option>
              <option value="most_downloads">Most Downloads</option>
              <option value="most_likes">Most Likes</option>
              <option value="recently_updated">Recently Updated</option>
            </select>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {searching ? (
            <div className="py-12 flex flex-col items-center gap-2 text-[12px] text-[var(--label-secondary)]">
              <Loader2 className="w-5 h-5 animate-spin text-[var(--accent)]" />
              <span>Searching HuggingFace Hub…</span>
            </div>
          ) : searchError ? (
            <div className="m-2 p-3 rounded-xl bg-[var(--signal-red-soft)] border border-[var(--signal-red-border)] text-[12px] text-[var(--signal-red)]">
              {searchError}
            </div>
          ) : results.length === 0 ? (
            <div className="py-12 text-center text-[12px] text-[var(--label-tertiary)]">
              No GGUF models matched.
            </div>
          ) : (
            results.map((item) => {
              const active = selectedRepo === item.repo_id;
              return (
                <button
                  key={item.repo_id}
                  type="button"
                  onClick={() => selectRepo(item.repo_id)}
                  className={`w-full text-left p-3 rounded-xl transition-colors cursor-pointer ${
                    active
                      ? 'bg-[var(--accent-soft)] border border-[var(--accent-border)] text-[var(--label-primary)]'
                      : 'hover:bg-[var(--bg-control)] border border-transparent text-[var(--label-primary)]'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div
                        className={`text-[13px] font-semibold truncate flex items-center gap-1.5 ${
                          active ? 'text-[var(--accent-text)]' : 'text-[var(--label-primary)]'
                        }`}
                      >
                        <span className="truncate">{item.name}</span>
                        {item.gated && <Lock className="w-3 h-3 shrink-0 opacity-80" />}
                      </div>
                      <div className="text-[11px] truncate text-[var(--label-secondary)]">
                        {item.author}
                      </div>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      {item.capabilities.vision && (
                        <span title="Vision">
                          <Eye className="w-3.5 h-3.5 text-[var(--signal-amber)]" />
                        </span>
                      )}
                      {item.capabilities.tool_use && (
                        <span title="Tool Use">
                          <Wrench className="w-3.5 h-3.5 text-[var(--signal-teal)]" />
                        </span>
                      )}
                      {item.capabilities.reasoning && (
                        <span title="Reasoning">
                          <Brain className="w-3.5 h-3.5 text-[var(--signal-green)]" />
                        </span>
                      )}
                      {item.capabilities.moe && (
                        <span title="MoE">
                          <Layers className="w-3.5 h-3.5 text-[var(--signal-purple)]" />
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="mt-1.5 flex items-center justify-between text-[10px] font-mono tabular-nums text-[var(--label-tertiary)]">
                    <span className="flex items-center gap-2">
                      <span>↓ {item.downloads.toLocaleString()}</span>
                      <span>♥ {item.likes.toLocaleString()}</span>
                      {item.param_count_label && <span>{item.param_count_label}</span>}
                    </span>
                    <span>{formatRelativeDate(item.last_modified)}</span>
                  </div>
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* Right Pane: Model Detail, Quant Picker & Rendered README */}
      <div className="flex-1 overflow-y-auto p-6 space-y-5 bg-[var(--bg-canvas)]">
        {loadingDetail ? (
          <div className="h-full flex flex-col items-center justify-center gap-2 text-[12px] text-[var(--label-secondary)]">
            <Loader2 className="w-6 h-6 animate-spin text-[var(--accent)]" />
            <span>Resolving GGUF quantizations &amp; hardware fit…</span>
          </div>
        ) : detailError ? (
          <div className="rounded-xl bg-[var(--signal-red-soft)] border border-[var(--signal-red-border)] p-4 text-[12px] text-[var(--signal-red)]">
            {detailError}
          </div>
        ) : !detail ? (
          <div className="h-full flex items-center justify-center text-[12px] text-[var(--label-tertiary)]">
            Select a GGUF repository from the left to inspect quantizations and VRAM fit.
          </div>
        ) : (
          <>
            {/* Header */}
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 min-w-0">
                  <h2 className="text-[18px] font-bold tracking-tight text-[var(--label-primary)] truncate">
                    {detail.summary.repo_id}
                  </h2>
                  <button
                    type="button"
                    onClick={() => {
                      navigator.clipboard.writeText(detail.summary.repo_id);
                      setCopiedRepo(true);
                      setTimeout(() => setCopiedRepo(false), 1500);
                    }}
                    className="apple-btn-ghost w-7 h-7"
                    title="Copy repo ID"
                    aria-label="Copy repo ID"
                  >
                    {copiedRepo ? (
                      <Check className="w-4 h-4 text-[var(--signal-green)]" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </button>
                </div>
                <div className="flex items-center gap-3 text-[12px] text-[var(--label-secondary)] font-mono tabular-nums shrink-0">
                  <span className="flex items-center gap-1">
                    <Download className="w-3.5 h-3.5" />
                    {detail.summary.downloads.toLocaleString()}
                  </span>
                  <span className="flex items-center gap-1">
                    <Heart className="w-3.5 h-3.5" />
                    {detail.summary.likes.toLocaleString()}
                  </span>
                  <span>Updated {formatRelativeDate(detail.summary.last_modified)}</span>
                </div>
              </div>

              {/* Metadata & Capability Badges Card */}
              <div className="bibulm-card p-4 flex flex-wrap items-center gap-4 text-[12px]">
                {detail.summary.param_count_label && (
                  <div>
                    <span className="text-[var(--label-tertiary)] mr-1.5 text-[11px] font-medium">
                      PARAMS
                    </span>
                    <strong className="text-[var(--label-primary)] font-mono">
                      {detail.summary.param_count_label}
                    </strong>
                  </div>
                )}
                {detail.summary.architecture && (
                  <div>
                    <span className="text-[var(--label-tertiary)] mr-1.5 text-[11px] font-medium">
                      ARCH
                    </span>
                    <strong className="text-[var(--label-primary)] font-mono">
                      {detail.summary.architecture}
                    </strong>
                  </div>
                )}
                {detail.summary.context_length && (
                  <div>
                    <span className="text-[var(--label-tertiary)] mr-1.5 text-[11px] font-medium">
                      CONTEXT
                    </span>
                    <strong className="text-[var(--label-primary)] font-mono tabular-nums">
                      {detail.summary.context_length.toLocaleString()}
                    </strong>
                  </div>
                )}
                <span className="apple-badge apple-badge-accent font-mono font-semibold text-[11px]">
                  GGUF
                </span>
                <div className="flex flex-wrap gap-1.5 ml-auto">
                  {detail.summary.capabilities.vision && (
                    <span className="apple-badge apple-badge-amber">Vision</span>
                  )}
                  {detail.summary.capabilities.tool_use && (
                    <span className="apple-badge apple-badge-teal">Tool Use</span>
                  )}
                  {detail.summary.capabilities.reasoning && (
                    <span className="apple-badge apple-badge-green">Reasoning</span>
                  )}
                  {detail.summary.capabilities.moe && (
                    <span className="apple-badge apple-badge-purple">MoE</span>
                  )}
                </div>
              </div>
            </div>

            {/* Download Options (Quantization & Split GGUF Picker) */}
            <div className="space-y-2.5">
              {selectedPeer && (
                <div className="rounded-xl bg-[var(--accent-soft)] border border-[var(--accent-border)] px-3.5 py-2 text-[11px] text-[var(--label-primary)] flex items-center gap-2">
                  <Server className="w-3.5 h-3.5 text-[var(--accent-text)] shrink-0" />
                  <span>
                    Downloading directly to <strong>{selectedPeer.device_name}</strong> — fit verdicts
                    calculated for {formatMiB(selectedPeer.vram_total_mib)} VRAM /{' '}
                    {formatMiB(selectedPeer.ram_total_mib)} RAM.
                  </span>
                </div>
              )}
              {downloadError && (
                <div className="rounded-xl bg-[var(--signal-red-soft)] border border-[var(--signal-red-border)] px-3.5 py-2 text-[12px] text-[var(--signal-red)]">
                  Failed to start download: {downloadError}
                </div>
              )}
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--label-tertiary)]">
                  Download Options ({detail.quant_options.length} quantizations)
                </h3>
                {detail.mmproj_files.length > 0 && (
                  <label className="flex items-center gap-2 text-[12px] text-[var(--label-secondary)]">
                    <span>Include Vision mmproj:</span>
                    <select
                      value={selectedMmprojPath}
                      onChange={(e) => setSelectedMmprojPath(e.target.value)}
                      className="apple-select h-7 text-[11px]"
                    >
                      <option value="">None</option>
                      {detail.mmproj_files.map((mm) => (
                        <option key={mm.path} value={mm.path}>
                          {mm.file_name} ({formatBytes(mm.size_bytes)})
                        </option>
                      ))}
                    </select>
                  </label>
                )}
              </div>

              {detail.quant_options.length === 0 ? (
                <div className="bibulm-card p-4 text-[12px] text-[var(--label-tertiary)]">
                  No `.gguf` files found in the main branch of this repository.
                </div>
              ) : (
                <div className="bibulm-card divide-y divide-[var(--border-hairline)] overflow-hidden">
                  {detail.quant_options.map((q) => (
                    <div
                      key={q.id}
                      className="px-4 py-2.5 flex items-center justify-between gap-3 hover:bg-[var(--bg-control)] transition-colors"
                    >
                      <div className="min-w-0 flex items-center gap-2.5 flex-wrap">
                        <span className="apple-badge font-mono text-[12px] font-semibold text-[var(--label-primary)]">
                          {q.quantization}
                        </span>
                        <span className="text-[12px] text-[var(--label-secondary)] truncate max-w-[300px]">
                          {q.label}
                        </span>
                        <FitVerdictBadge verdict={q.fit_verdict} />
                      </div>
                      <div className="flex items-center gap-3 shrink-0">
                        <span className="text-[12px] font-mono text-[var(--label-secondary)] tabular-nums">
                          {formatBytes(q.total_size_bytes)}
                        </span>
                        <button
                          type="button"
                          onClick={() => handleDownloadQuant(q)}
                          className={`h-7 px-3 rounded-lg text-[12px] font-medium flex items-center gap-1.5 transition-colors cursor-pointer ${
                            queuedQuantId === q.id
                              ? 'bg-[var(--signal-green-fill)] text-white'
                              : 'bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white'
                          }`}
                        >
                          {queuedQuantId === q.id ? (
                            <>
                              <Check className="w-3.5 h-3.5" /> Queued
                            </>
                          ) : (
                            <>
                              <Download className="w-3.5 h-3.5" /> Download
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* README Markdown */}
            {readmeHtml && (
              <div className="space-y-2">
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--label-tertiary)]">
                  README
                </h3>
                <div
                  className="bibulm-card p-5 readme-prose overflow-x-auto"
                  onClick={handleReadmeClick}
                  dangerouslySetInnerHTML={{ __html: readmeHtml }}
                />
              </div>
            )}
          </>
        )}
      </div>
      </div>
    </div>
  );
};
