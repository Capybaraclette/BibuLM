import React, { useEffect, useRef, useState } from 'react';
import {
  ArrowUpFromLine,
  ChevronDown,
  Download,
  Link2,
  Loader2,
  PanelLeft,
  PanelRight,
  Play,
  RotateCcw,
  Search,
  Sliders,
} from 'lucide-react';
import type {
  ActiveRemoteModel,
  DownloadJobState,
  LinkStatus,
  LoadedModelInfo,
  LocalModelEntry,
  RemoteModelSummary,
  RemotePeer,
} from '../types';
import { formatBytes, formatMiB } from '../api';
import { BibuLMLogo } from './BibuLMLogo';
import { PlacementBadge } from './PlacementBadge';
import {
  WindowControls,
  startWindowDrag,
  toggleWindowMaximize,
  useWindowFocus,
} from './TitleBar';

interface TopBarProps {
  appVersion?: string;
  loadedModels: LoadedModelInfo[];
  activeModelId: string | null;
  activeRemoteModel: ActiveRemoteModel | null;
  localModels: LocalModelEntry[];
  linkStatus: LinkStatus | null;
  linkPeers: RemotePeer[];
  downloads: DownloadJobState[];
  remoteActiveDownloadsCount?: number;
  isLoadingModel: boolean;
  quickLoadOpen: boolean;
  setQuickLoadOpen: (open: boolean) => void;
  downloadsOpen: boolean;
  setDownloadsOpen: (open: boolean) => void;
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (v: boolean) => void;
  inspectorOpen: boolean;
  setInspectorOpen: (v: boolean) => void;
  onSelectActiveModel: (modelId: string) => void;
  onSelectRemoteModel: (peer: RemotePeer, model: RemoteModelSummary) => void;
  onRemoteLoadModel: (peer: RemotePeer, model: RemoteModelSummary) => void;
  onConfigureRemoteModel: (peer: RemotePeer, model: RemoteModelSummary) => void;
  onRemoteUnloadModel: (peer: RemotePeer, model: RemoteModelSummary) => void;
  onConfigureModel: (model: LocalModelEntry) => void;
  onUnloadModel: (modelId: string) => void;
  onReloadLast: () => void;
  canReloadLast: boolean;
  onOpenLinkTab: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  appVersion,
  loadedModels,
  activeModelId,
  activeRemoteModel,
  localModels,
  linkStatus,
  linkPeers,
  downloads,
  remoteActiveDownloadsCount = 0,
  isLoadingModel,
  quickLoadOpen,
  setQuickLoadOpen,
  downloadsOpen,
  setDownloadsOpen,
  sidebarCollapsed,
  setSidebarCollapsed,
  inspectorOpen,
  setInspectorOpen,
  onSelectActiveModel,
  onSelectRemoteModel,
  onRemoteLoadModel,
  onConfigureRemoteModel,
  onRemoteUnloadModel,
  onConfigureModel,
  onUnloadModel,
  onReloadLast,
  canReloadLast,
  onOpenLinkTab,
}) => {
  const [filter, setFilter] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const windowFocused = useWindowFocus();

  const activeLoaded = activeRemoteModel
    ? null
    : loadedModels.find((m) => m.model_id === activeModelId) ?? loadedModels[0] ?? null;

  const activeDownloads = downloads.filter(
    (d) => d.status === 'downloading' || d.status === 'queued' || d.status === 'verifying',
  );
  const totalActiveDownloads = activeDownloads.length + remoteActiveDownloadsCount;

  useEffect(() => {
    if (quickLoadOpen) {
      setTimeout(() => inputRef.current?.focus(), 20);
    } else {
      setFilter('');
    }
  }, [quickLoadOpen]);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setQuickLoadOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [setQuickLoadOpen]);

  const q = filter.trim().toLowerCase();

  const filteredLocals = localModels.filter((m) => {
    if (!q) return true;
    return (
      m.display_name.toLowerCase().includes(q) ||
      m.id.toLowerCase().includes(q) ||
      (m.architecture ?? '').toLowerCase().includes(q) ||
      (m.quantization ?? '').toLowerCase().includes(q)
    );
  });

  const exposingPeers = linkPeers.filter((p) => p.online && p.expose_local_models);
  const filteredRemoteEntries: { peer: RemotePeer; model: RemoteModelSummary }[] = [];
  for (const peer of exposingPeers) {
    for (const m of peer.loaded_models) {
      if (
        !q ||
        m.display_name.toLowerCase().includes(q) ||
        m.id.toLowerCase().includes(q) ||
        peer.device_name.toLowerCase().includes(q) ||
        peer.slug.toLowerCase().includes(q)
      ) {
        filteredRemoteEntries.push({ peer, model: m });
      }
    }
    if (peer.allow_remote_load) {
      for (const m of peer.available_models) {
        if (
          !q ||
          m.display_name.toLowerCase().includes(q) ||
          m.id.toLowerCase().includes(q) ||
          peer.device_name.toLowerCase().includes(q) ||
          peer.slug.toLowerCase().includes(q)
        ) {
          filteredRemoteEntries.push({ peer, model: m });
        }
      }
    }
  }

  const linkState = linkStatus?.state ?? 'disconnected';

  return (
    <header
      data-tauri-drag-region
      onMouseDown={startWindowDrag}
      onDoubleClick={toggleWindowMaximize}
      className="h-[52px] shrink-0 apple-glass border-b grid grid-cols-[1fr_auto_1fr] items-center gap-3 px-3 select-none z-30"
    >
      {/* Leading: Sidebar Toggle & Subtle App Identity */}
      <div
        className="flex items-center gap-2.5 justify-self-start min-w-0"
        onMouseDown={(e) => e.stopPropagation()}
        onDoubleClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          aria-label="Toggle navigation sidebar"
          title="Toggle Sidebar"
          className="apple-btn-ghost w-8 h-8"
        >
          <PanelLeft className="w-4 h-4" />
        </button>

        <div className="flex items-center gap-2 min-w-0 pr-1">
          <BibuLMLogo className="w-5 h-5 rounded-[6px]" />
          <span className="text-[13px] font-semibold tracking-tight text-[var(--label-primary)] truncate">
            BibuLM
          </span>
          {appVersion && (
            <span className="hidden sm:inline-block text-[11px] font-mono text-[var(--label-tertiary)] tabular-nums">
              v{appVersion}
            </span>
          )}
        </div>
      </div>

      {/* Center: Xcode / Spotlight-style Model Command Pill (Ctrl+L) */}
      <div
        className="relative flex items-center justify-center gap-1.5 justify-self-center"
        ref={dropdownRef}
        onMouseDown={(e) => e.stopPropagation()}
        onDoubleClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={() => setQuickLoadOpen(!quickLoadOpen)}
          aria-expanded={quickLoadOpen}
          className="h-8 min-w-[320px] max-w-[540px] px-3 rounded-[10px] bg-[var(--bg-control)] hover:bg-[var(--bg-control-hover)] border border-[var(--border-hairline)] flex items-center justify-between gap-3 text-[12px] transition-colors cursor-pointer"
        >
          <div className="flex items-center gap-2 min-w-0">
            {isLoadingModel ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin text-[var(--accent)] shrink-0" />
                <span className="text-[var(--accent-text)] font-medium truncate">
                  Loading llama-server…
                </span>
              </>
            ) : activeRemoteModel ? (
              <>
                <span className="w-2 h-2 rounded-full bg-[var(--accent)] shrink-0" />
                <span className="font-medium text-[var(--label-primary)] truncate">
                  {activeRemoteModel.model.display_name}
                </span>
                <span className="apple-badge apple-badge-accent shrink-0 text-[10px] py-0 px-1.5">
                  <Link2 className="w-2.5 h-2.5" />
                  {activeRemoteModel.peer.device_name}
                </span>
              </>
            ) : activeLoaded ? (
              <>
                <span className="w-2 h-2 rounded-full bg-[var(--signal-green-fill)] shrink-0" />
                <span className="font-medium text-[var(--label-primary)] truncate">
                  {activeLoaded.display_name}
                </span>
                <PlacementBadge status={activeLoaded.placement_status} compact />
              </>
            ) : (
              <span className="text-[var(--label-tertiary)] truncate">
                Select or load a model…
              </span>
            )}
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <kbd className="apple-kbd">⌃L</kbd>
            <ChevronDown className="w-3.5 h-3.5 text-[var(--label-tertiary)]" />
          </div>
        </button>

        {activeLoaded && (
          <button
            type="button"
            onClick={() => onUnloadModel(activeLoaded.model_id)}
            className="h-8 px-2.5 rounded-[10px] bg-[var(--bg-control)] hover:bg-[var(--signal-red-soft)] hover:text-[var(--signal-red)] hover:border-[var(--signal-red-border)] border border-[var(--border-hairline)] text-[var(--label-secondary)] flex items-center gap-1.5 text-[12px] font-medium transition-colors cursor-pointer"
            title="Eject active model"
          >
            <ArrowUpFromLine className="w-3.5 h-3.5" />
            <span>Eject</span>
          </button>
        )}

        {activeRemoteModel && activeRemoteModel.peer.allow_remote_load && (
          <>
            <button
              type="button"
              onClick={() =>
                onConfigureRemoteModel(activeRemoteModel.peer, activeRemoteModel.model)
              }
              className="h-8 w-8 rounded-[10px] bg-[var(--bg-control)] hover:bg-[var(--bg-control-hover)] border border-[var(--border-hairline)] text-[var(--label-secondary)] hover:text-[var(--label-primary)] flex items-center justify-center transition-colors cursor-pointer"
              title="Configure parameters & reload on remote machine"
              aria-label="Configure remote model"
            >
              <Sliders className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={() =>
                onRemoteUnloadModel(activeRemoteModel.peer, activeRemoteModel.model)
              }
              className="h-8 px-2.5 rounded-[10px] bg-[var(--bg-control)] hover:bg-[var(--signal-red-soft)] hover:text-[var(--signal-red)] hover:border-[var(--signal-red-border)] border border-[var(--border-hairline)] text-[var(--label-secondary)] flex items-center gap-1.5 text-[12px] font-medium transition-colors cursor-pointer"
              title="Eject remote model"
            >
              <ArrowUpFromLine className="w-3.5 h-3.5" />
              <span>Eject</span>
            </button>
          </>
        )}

        {!activeLoaded && !activeRemoteModel && canReloadLast && !isLoadingModel && (
          <button
            type="button"
            onClick={onReloadLast}
            className="bibulm-btn-primary h-8"
            title="Reload last used model (Ctrl+R)"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reload</span>
          </button>
        )}

        {/* Spotlight-Inspired Quick Load Popover */}
        {quickLoadOpen && (
          <div
            className="absolute top-10 left-1/2 -translate-x-1/2 w-[560px] max-w-[92vw] max-h-[480px] rounded-2xl apple-glass border flex flex-col overflow-hidden z-50"
            style={{ boxShadow: 'var(--shadow-popover)' }}
          >
            <div className="px-3.5 py-2.5 border-b border-[var(--border-hairline)] flex items-center gap-2.5">
              <Search className="w-4 h-4 text-[var(--label-tertiary)] shrink-0" />
              <input
                ref={inputRef}
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="Search local or remote Link models…"
                className="w-full bg-transparent text-[13px] text-[var(--label-primary)] placeholder-[var(--label-quaternary)] focus:outline-none focus:shadow-none"
              />
            </div>

            <div className="overflow-y-auto divide-y divide-[var(--border-hairline)] bg-[var(--bg-surface)]/80">
              {loadedModels.length > 0 && (
                <div className="p-2">
                  <div className="px-2 pb-1 text-[11px] font-semibold uppercase tracking-wider text-[var(--label-tertiary)]">
                    Currently Loaded ({loadedModels.length})
                  </div>
                  {loadedModels.map((m) => (
                    <div
                      key={m.model_id}
                      className={`flex items-center justify-between gap-2 px-2.5 py-2 rounded-lg cursor-pointer transition-colors ${
                        !activeRemoteModel && m.model_id === activeLoaded?.model_id
                          ? 'bg-[var(--accent-soft)] border border-[var(--accent-border)]'
                          : 'hover:bg-[var(--bg-control)]'
                      }`}
                      onClick={() => {
                        onSelectActiveModel(m.model_id);
                        setQuickLoadOpen(false);
                      }}
                    >
                      <div className="min-w-0 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-[var(--signal-green-fill)] shrink-0" />
                        <span className="text-[12px] font-medium text-[var(--label-primary)] truncate">
                          {m.display_name}
                        </span>
                        <PlacementBadge status={m.placement_status} compact />
                      </div>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onUnloadModel(m.model_id);
                        }}
                        className="text-[11px] px-2 py-1 rounded-md bg-[var(--bg-control)] hover:bg-[var(--signal-red-soft)] hover:text-[var(--signal-red)] text-[var(--label-secondary)] font-medium transition-colors cursor-pointer"
                      >
                        Eject
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Remote Tailscale Link Models */}
              {filteredRemoteEntries.length > 0 && (
                <div className="p-2">
                  <div className="px-2 pb-1 flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider text-[var(--accent-text)]">
                    <span className="flex items-center gap-1">
                      <Link2 className="w-3 h-3" />
                      Remote Peers ({filteredRemoteEntries.length})
                    </span>
                    <button
                      type="button"
                      onClick={() => {
                        setQuickLoadOpen(false);
                        onOpenLinkTab();
                      }}
                      className="text-[var(--label-secondary)] hover:text-[var(--label-primary)] normal-case font-medium cursor-pointer"
                    >
                      Manage Link
                    </button>
                  </div>
                  {filteredRemoteEntries.map(({ peer, model }) => {
                    const qualifiedId = `${peer.slug}/${model.id}`;
                    const isSelected = activeRemoteModel?.qualifiedId === qualifiedId;
                    return (
                      <div
                        key={`${peer.node_id}:${model.id}`}
                        className={`flex items-center justify-between gap-2 px-2.5 py-2 rounded-lg transition-colors ${
                          isSelected
                            ? 'bg-[var(--accent-soft)] border border-[var(--accent-border)]'
                            : 'hover:bg-[var(--bg-control)]'
                        }`}
                      >
                        <div
                          className="min-w-0 flex-1 cursor-pointer"
                          onClick={() => {
                            if (model.loaded) {
                              onSelectRemoteModel(peer, model);
                              setQuickLoadOpen(false);
                            } else {
                              onConfigureRemoteModel(peer, model);
                              setQuickLoadOpen(false);
                            }
                          }}
                        >
                          <div className="flex items-center gap-2">
                            <span
                              className={`w-2 h-2 rounded-full shrink-0 ${
                                model.loaded ? 'bg-[var(--accent)]' : 'bg-[var(--label-quaternary)]'
                              }`}
                            />
                            <span className="text-[12px] font-medium text-[var(--label-primary)] truncate">
                              {model.display_name}
                            </span>
                            <span className="apple-badge apple-badge-accent text-[10px] py-0 px-1.5 shrink-0">
                              {peer.device_name}
                            </span>
                            {model.quantization && (
                              <span className="apple-badge font-mono text-[10px] py-0 px-1.5">
                                {model.quantization}
                              </span>
                            )}
                          </div>
                          <div className="text-[11px] text-[var(--label-tertiary)] flex items-center gap-2 mt-0.5 pl-4 tabular-nums">
                            <span>{model.loaded ? 'Loaded remotely' : 'Available on peer'}</span>
                            <span>
                              • VRAM {formatMiB(peer.vram_used_mib)}/{formatMiB(peer.vram_total_mib)}
                            </span>
                          </div>
                        </div>

                        {model.loaded ? (
                          <div className="flex items-center gap-1 shrink-0">
                            <button
                              type="button"
                              onClick={() => {
                                onSelectRemoteModel(peer, model);
                                setQuickLoadOpen(false);
                              }}
                              className="bibulm-btn-primary h-7 px-2.5"
                            >
                              <Play className="w-3 h-3" />
                              <span>Use</span>
                            </button>
                            {peer.allow_remote_load && (
                              <>
                                <button
                                  type="button"
                                  onClick={() => {
                                    onConfigureRemoteModel(peer, model);
                                    setQuickLoadOpen(false);
                                  }}
                                  className="apple-btn-ghost"
                                  title="Configure parameters & reload on remote peer"
                                >
                                  <Sliders className="w-3.5 h-3.5" />
                                </button>
                                <button
                                  type="button"
                                  onClick={() => onRemoteUnloadModel(peer, model)}
                                  className="apple-btn-ghost hover:text-[var(--signal-red)]"
                                  title="Eject model on remote peer"
                                >
                                  <ArrowUpFromLine className="w-3.5 h-3.5" />
                                </button>
                              </>
                            )}
                          </div>
                        ) : (
                          <div className="flex items-center gap-1 shrink-0">
                            <button
                              type="button"
                              onClick={() => {
                                onConfigureRemoteModel(peer, model);
                                setQuickLoadOpen(false);
                              }}
                              className="bibulm-btn-primary h-7 px-2.5"
                              title="Configure load parameters before loading remotely"
                            >
                              <Sliders className="w-3 h-3" />
                              <span>Load</span>
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                onRemoteLoadModel(peer, model);
                                setQuickLoadOpen(false);
                              }}
                              className="apple-btn-ghost"
                              title="Quick load with saved preset or auto-fit"
                            >
                              <Play className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              <div className="p-2">
                <div className="px-2 pb-1 text-[11px] font-semibold uppercase tracking-wider text-[var(--label-tertiary)]">
                  Local Library ({filteredLocals.length})
                </div>
                {filteredLocals.length === 0 ? (
                  <div className="px-3 py-6 text-center text-[12px] text-[var(--label-tertiary)]">
                    No local GGUF models found. Download models from Model Search or add a folder in
                    Library.
                  </div>
                ) : (
                  filteredLocals.map((m) => (
                    <div
                      key={m.id}
                      className="flex items-center justify-between gap-2 px-2.5 py-2 rounded-lg hover:bg-[var(--bg-control)] transition-colors group"
                    >
                      <div
                        className="min-w-0 flex-1 cursor-pointer"
                        onClick={() => {
                          setQuickLoadOpen(false);
                          onConfigureModel(m);
                        }}
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-[12px] font-medium text-[var(--label-primary)] truncate">
                            {m.display_name}
                          </span>
                          {m.quantization && (
                            <span className="apple-badge font-mono text-[10px] py-0 px-1.5">
                              {m.quantization}
                            </span>
                          )}
                          {m.is_moe && (
                            <span className="apple-badge apple-badge-purple text-[10px] py-0 px-1.5">
                              MoE
                            </span>
                          )}
                          {m.has_mtp && (
                            <span className="apple-badge apple-badge-purple text-[10px] py-0 px-1.5">
                              MTP
                            </span>
                          )}
                        </div>
                        <div className="text-[11px] text-[var(--label-tertiary)] flex items-center gap-2 mt-0.5 tabular-nums">
                          <span>{formatBytes(m.size_bytes)}</span>
                          {m.architecture && <span>• {m.architecture}</span>}
                          {m.layer_count && <span>• {m.layer_count} layers</span>}
                        </div>
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        <button
                          type="button"
                          onClick={() => {
                            setQuickLoadOpen(false);
                            onConfigureModel(m);
                          }}
                          className="bibulm-btn-primary h-7 px-2.5"
                          title="Open load parameters before loading"
                        >
                          <Sliders className="w-3 h-3" />
                          <span>Load</span>
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Trailing: Link Status, Downloads, Inspector Toggle, Divider & Window Controls */}
      <div
        className="flex items-center gap-1.5 justify-self-end min-w-0"
        onMouseDown={(e) => e.stopPropagation()}
        onDoubleClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={onOpenLinkTab}
          className={`h-7 px-2.5 rounded-lg border text-[11px] font-medium flex items-center gap-1.5 transition-colors cursor-pointer ${
            linkState === 'connected'
              ? 'bg-[var(--signal-green-soft)] border-[var(--signal-green-border)] text-[var(--signal-green)]'
              : linkState === 'needs_login' ||
                  linkState === 'connecting' ||
                  linkState === 'starting'
                ? 'bg-[var(--signal-amber-soft)] border-[var(--signal-amber-border)] text-[var(--signal-amber)]'
                : 'bg-[var(--bg-control)] border-[var(--border-hairline)] text-[var(--label-secondary)] hover:text-[var(--label-primary)] hover:bg-[var(--bg-control-hover)]'
          }`}
          title="BibuLM Link (BYO Tailscale)"
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              linkState === 'connected'
                ? 'bg-[var(--signal-green-fill)]'
                : linkState === 'needs_login' ||
                    linkState === 'connecting' ||
                    linkState === 'starting'
                  ? 'bg-[var(--signal-amber-fill)] animate-pulse'
                  : 'bg-[var(--label-quaternary)]'
            }`}
          />
          <Link2 className="w-3.5 h-3.5" />
          <span>Link</span>
          {linkState === 'connected' && (
            <span className="font-mono text-[10px] tabular-nums opacity-90">
              {linkPeers.length}
            </span>
          )}
        </button>

        <button
          type="button"
          onClick={() => setDownloadsOpen(!downloadsOpen)}
          aria-label="Downloads Manager"
          title="Downloads Manager"
          className={`relative apple-btn-ghost w-8 h-7 ${
            downloadsOpen
              ? 'bg-[var(--accent-soft)] text-[var(--accent-text)]'
              : ''
          }`}
        >
          <Download className="w-4 h-4" />
          {totalActiveDownloads > 0 && (
            <span className="absolute -top-0.5 -right-0.5 min-w-[16px] h-4 px-1 rounded-full bg-[var(--accent)] text-[10px] font-bold text-white flex items-center justify-center tabular-nums">
              {totalActiveDownloads}
            </span>
          )}
        </button>

        <button
          type="button"
          onClick={() => setInspectorOpen(!inspectorOpen)}
          aria-label="Toggle Inference Parameters Inspector"
          title="Toggle Inference Parameters Inspector"
          className={`apple-btn-ghost w-8 h-7 ${
            inspectorOpen
              ? 'bg-[var(--accent-soft)] text-[var(--accent-text)]'
              : ''
          }`}
        >
          <PanelRight className="w-4 h-4" />
        </button>

        <div className="w-px h-4 bg-[var(--border-hairline)] mx-1" aria-hidden="true" />

        <WindowControls focused={windowFocused} />
      </div>
    </header>
  );
};
