import React, { useCallback, useEffect, useRef, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import {
  Activity,
  AlertTriangle,
  ArrowDownToLine,
  Cpu,
  Gauge,
  Library,
  Link2,
  Search,
  Server,
  Settings,
  X,
} from 'lucide-react';
import type {
  ActiveRemoteModel,
  AppSettings,
  AppUpdateInfo,
  AppUpdateProgress,
  DownloadJobState,
  GatewayStatus,
  InstalledRuntimeRecord,
  LinkStatus,
  LoadConfig,
  LoadedModelInfo,
  LocalModelEntry,
  NavTab,
  ReasoningEffort,
  RemoteModelInspectResponse,
  RemoteModelSummary,
  RemotePeer,
  SamplingParams,
  ServerLogEntry,
  SystemHardwareProfile,
  TelemetrySnapshot,
} from './types';
import { api, formatBytes } from './api';
import { TopBar } from './components/TopBar';
import { DownloadsPopover } from './components/DownloadsPopover';
import { InspectorPanel } from './components/InspectorPanel';
import { LoadConfigModal } from './components/LoadConfigModal';
import { BenchmarkView } from './views/BenchmarkView';
import { ServerView } from './views/ServerView';
import { LinkView } from './views/LinkView';
import { LibraryView } from './views/LibraryView';
import { SearchView } from './views/SearchView';
import { RuntimeView } from './views/RuntimeView';
import { HardwareView } from './views/HardwareView';
import { SettingsView } from './views/SettingsView';

const DEFAULT_SAMPLING: SamplingParams = {
  system_prompt: '',
  temperature: 0.7,
  top_k: 40,
  top_p: 0.95,
  min_p: 0.05,
  typical_p: null,
  repeat_penalty: 1.05,
  repeat_last_n: 64,
  presence_penalty: null,
  frequency_penalty: null,
  dry_multiplier: null,
  dry_base: null,
  dry_allowed_length: null,
  xtc_probability: null,
  xtc_threshold: null,
  dynatemp_range: null,
  mirostat: null,
  mirostat_tau: null,
  mirostat_eta: null,
  max_tokens: 1024,
  stop_strings: [],
  structured_output_enabled: false,
  json_schema: null,
  grammar: null,
};

export const App: React.FC = () => {
  const [tab, setTab] = useState<NavTab>('benchmark');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [quickLoadOpen, setQuickLoadOpen] = useState(false);
  const [downloadsOpen, setDownloadsOpen] = useState(false);

  // Core app state
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [hwProfile, setHwProfile] = useState<SystemHardwareProfile | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetrySnapshot | null>(null);
  const [localModels, setLocalModels] = useState<LocalModelEntry[]>([]);
  const [scanningLibrary, setScanningLibrary] = useState(false);
  const [loadedModels, setLoadedModels] = useState<LoadedModelInfo[]>([]);
  const [activeModelId, setActiveModelId] = useState<string | null>(null);
  const [activeRemoteModel, setActiveRemoteModel] = useState<ActiveRemoteModel | null>(null);
  const [lastLoadedModel, setLastLoadedModel] = useState<LocalModelEntry | null>(null);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [runtimes, setRuntimes] = useState<InstalledRuntimeRecord[]>([]);
  const [serverStatus, setServerStatus] = useState<GatewayStatus | null>(null);
  const [linkStatus, setLinkStatus] = useState<LinkStatus | null>(null);
  const [linkPeers, setLinkPeers] = useState<RemotePeer[]>([]);
  const [logs, setLogs] = useState<ServerLogEntry[]>([]);
  const [downloads, setDownloads] = useState<DownloadJobState[]>([]);
  const [runtimeTargetPeerId, setRuntimeTargetPeerId] = useState<string | null>(null);
  const [searchTargetPeerId, setSearchTargetPeerId] = useState<string | null>(null);
  const [downloadsTargetPeerId, setDownloadsTargetPeerId] = useState<string | null>(null);
  const [remoteDownloads, setRemoteDownloads] = useState<Record<string, DownloadJobState[]>>({});
  const remoteDownloadsRef = useRef<Record<string, DownloadJobState[]>>({});

  // Inspector & Modal state
  const [sampling, setSampling] = useState<SamplingParams>(DEFAULT_SAMPLING);
  const [reasoningEffort, setReasoningEffort] = useState<ReasoningEffort>('default');
  const [configuringModel, setConfiguringModel] = useState<LocalModelEntry | null>(null);
  const [configuringRemote, setConfiguringRemote] = useState<{
    peer: RemotePeer;
    inspect: RemoteModelInspectResponse;
  } | null>(null);
  const [bannerError, setBannerError] = useState<string | null>(null);

  // App Auto-Updater state
  const [appVersion, setAppVersion] = useState<string>('');
  const [updateInfo, setUpdateInfo] = useState<AppUpdateInfo | null>(null);
  const [updateProgress, setUpdateProgress] = useState<AppUpdateProgress | null>(null);
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const [installingUpdate, setInstallingUpdate] = useState(false);
  const [updateError, setUpdateError] = useState<string | null>(null);
  const [updateBannerDismissed, setUpdateBannerDismissed] = useState(false);

  const syncLinkPeers = useCallback((nextPeers: RemotePeer[]) => {
    setLinkPeers(nextPeers);
    setActiveRemoteModel((cur) => {
      if (!cur) return null;
      const peer = nextPeers.find(
        (p) => p.node_id === cur.peer.node_id && p.online && p.expose_local_models,
      );
      if (!peer) return null;
      // Only keep activeRemoteModel while the model is actually loaded on the peer.
      const stillLoaded = peer.loaded_models.find((m) => m.id === cur.model.id);
      return stillLoaded
        ? { peer, model: stillLoaded, qualifiedId: `${peer.slug}/${stillLoaded.id}` }
        : null;
    });
  }, []);

  const refreshLibrary = useCallback(async () => {
    setScanningLibrary(true);
    try {
      setLocalModels(await api.scanLocalLibrary());
    } finally {
      setScanningLibrary(false);
    }
  }, []);

  const refreshRuntimes = useCallback(async () => {
    try {
      setRuntimes(await api.listInstalledRuntimes());
    } catch {
      // ignore
    }
  }, []);

  const refreshDownloads = useCallback(async () => {
    try {
      setDownloads(await api.listDownloads());
    } catch {
      // ignore
    }
  }, []);

  const refreshRemoteDownloads = useCallback(
    async (peerId: string) => {
      try {
        const list = await api.remoteListDownloads(peerId);
        const prevActive = new Set(
          (remoteDownloadsRef.current[peerId] ?? [])
            .filter(
              (j) =>
                j.status === 'downloading' || j.status === 'queued' || j.status === 'verifying',
            )
            .map((j) => j.id),
        );
        const completedAny = list.some((j) => j.status === 'completed' && prevActive.has(j.id));
        const nextMap = { ...remoteDownloadsRef.current, [peerId]: list };
        remoteDownloadsRef.current = nextMap;
        setRemoteDownloads(nextMap);
        if (completedAny) {
          const peers = await api.refreshLinkPeers();
          syncLinkPeers(peers);
        }
      } catch {
        // ignore transient mesh errors while polling
      }
    },
    [syncLinkPeers],
  );

  const pollingPeerIdsKey = (() => {
    const peerIds = new Set<string>();
    if (downloadsOpen && downloadsTargetPeerId) {
      peerIds.add(downloadsTargetPeerId);
    }
    for (const p of linkPeers) {
      if (p.online && p.allow_remote_manage && p.active_downloads_count > 0) {
        peerIds.add(p.node_id);
      }
    }
    for (const [pid, jobs] of Object.entries(remoteDownloads)) {
      if (
        jobs.some(
          (j) => j.status === 'downloading' || j.status === 'queued' || j.status === 'verifying',
        ) &&
        linkPeers.some((p) => p.node_id === pid && p.online && p.allow_remote_manage)
      ) {
        peerIds.add(pid);
      }
    }
    return Array.from(peerIds).sort().join(',');
  })();

  // Poll remote downloads only when the Downloads popover is viewing a remote peer
  // or a manageable peer has active downloads in flight.
  useEffect(() => {
    if (!pollingPeerIdsKey) return;
    const peerIds = pollingPeerIdsKey.split(',');

    const tick = () => {
      peerIds.forEach((pid) => {
        refreshRemoteDownloads(pid);
      });
    };
    tick();
    const timer = setInterval(tick, 2000);
    return () => clearInterval(timer);
  }, [pollingPeerIdsKey, refreshRemoteDownloads]);

  // Initial hydration
  useEffect(() => {
    (async () => {
      try {
        const [s, hw, tel, loaded, rts, srv, initialLogs, dls, ver, lSt, lPeers] = await Promise.all([
          api.getSettings(),
          api.getHardwareProfile(),
          api.getTelemetrySnapshot(),
          api.listLoadedModels(),
          api.listInstalledRuntimes(),
          api.getServerStatus(),
          api.getServerLogs(),
          api.listDownloads(),
          api.getAppVersion(),
          api.getLinkStatus(),
          api.listLinkPeers(),
        ]);
        setSettings(s);
        setSampling(s.default_sampling ?? DEFAULT_SAMPLING);
        setHwProfile(hw);
        setTelemetry(tel);
        setLoadedModels(loaded);
        if (loaded.length > 0) setActiveModelId(loaded[0].model_id);
        setRuntimes(rts);
        setServerStatus(srv);
        setLogs(initialLogs);
        setDownloads(dls);
        setAppVersion(ver);
        setLinkStatus(lSt);
        syncLinkPeers(lPeers);
        await refreshLibrary();
      } catch (e) {
        setBannerError(`Initialization error: ${e}`);
      }
    })();
  }, [refreshLibrary, syncLinkPeers]);

  // Subscribe to real-time Tauri backend events
  useEffect(() => {
    const unsubs: (() => void)[] = [];

    listen<TelemetrySnapshot>('telemetry-update', (ev) => {
      setTelemetry(ev.payload);
    }).then((fn) => unsubs.push(fn));

    listen<ServerLogEntry>('server-log', (ev) => {
      setLogs((prev) => {
        const next = [...prev, ev.payload];
        return next.length > 1500 ? next.slice(next.length - 1500) : next;
      });
    }).then((fn) => unsubs.push(fn));

    listen<LoadedModelInfo[]>('loaded-models-changed', (ev) => {
      setLoadedModels(ev.payload);
      if (ev.payload.length > 0) {
        setActiveModelId((cur) =>
          ev.payload.some((m) => m.model_id === cur) ? cur : ev.payload[0].model_id,
        );
      } else {
        setActiveModelId(null);
      }
    }).then((fn) => unsubs.push(fn));

    listen<GatewayStatus>('server-status-changed', (ev) => {
      setServerStatus(ev.payload);
    }).then((fn) => unsubs.push(fn));

    listen<LinkStatus>('link-status-changed', (ev) => {
      setLinkStatus(ev.payload);
    }).then((fn) => unsubs.push(fn));

    listen<RemotePeer[]>('link-peers-changed', (ev) => {
      syncLinkPeers(ev.payload);
    }).then((fn) => unsubs.push(fn));

    listen<DownloadJobState>('download-progress', (ev) => {
      const updated = ev.payload;
      setDownloads((prev) => {
        const idx = prev.findIndex((d) => d.id === updated.id);
        if (idx === -1) return [updated, ...prev];
        const copy = [...prev];
        copy[idx] = updated;
        return copy;
      });
      if (updated.status === 'completed' && updated.kind === 'model') {
        refreshLibrary();
      }
    }).then((fn) => unsubs.push(fn));

    listen<AppUpdateInfo>('app-update-available', (ev) => {
      setUpdateInfo(ev.payload);
      setUpdateBannerDismissed(false);
    }).then((fn) => unsubs.push(fn));

    listen<AppUpdateProgress>('app-update-progress', (ev) => {
      setUpdateProgress(ev.payload);
      if (ev.payload.stage === 'error') {
        setUpdateError(ev.payload.message);
        setInstallingUpdate(false);
      }
    }).then((fn) => unsubs.push(fn));

    return () => {
      unsubs.forEach((u) => u());
    };
  }, [refreshLibrary, syncLinkPeers]);

  const handleCheckAppUpdate = useCallback(async () => {
    setCheckingUpdate(true);
    setUpdateError(null);
    try {
      const info = await api.checkAppUpdate();
      setUpdateInfo(info);
      if (info.update_available) {
        setUpdateBannerDismissed(false);
      }
    } catch (e) {
      setUpdateError(String(e));
    } finally {
      setCheckingUpdate(false);
    }
  }, []);

  const handleInstallAppUpdate = useCallback(async (info: AppUpdateInfo) => {
    setInstallingUpdate(true);
    setUpdateError(null);
    setUpdateProgress({
      stage: 'downloading',
      downloaded_bytes: 0,
      total_bytes: info.asset_size_bytes,
      percent: 0,
      message: 'Starting update download...',
    });
    try {
      await api.installAppUpdate(info);
    } catch (e) {
      setUpdateError(String(e));
      setInstallingUpdate(false);
    }
  }, []);

  const handleSaveSettings = async (next: AppSettings) => {
    const saved = await api.saveSettings(next);
    setSettings(saved);
  };

  const handleExecuteLoad = useCallback(
    async (
      model: LocalModelEntry,
      config: LoadConfig,
      rememberPreset: boolean,
      forceOverride?: boolean,
    ) => {
      setConfiguringModel(null);
      setIsLoadingModel(true);
      setBannerError(null);
      try {
        const info = await api.loadModel(
          model.id,
          model.display_name,
          config,
          rememberPreset,
          forceOverride,
        );
        setLastLoadedModel(model);
        setActiveRemoteModel(null);
        setActiveModelId(info.model_id);
        setLoadedModels(await api.listLoadedModels());
      } catch (e) {
        setBannerError(String(e));
      } finally {
        setIsLoadingModel(false);
      }
    },
    [],
  );

  const handleRequestLoad = useCallback((model: LocalModelEntry) => {
    setConfiguringModel(model);
  }, []);

  const handleUnloadModel = async (modelId: string) => {
    await api.unloadModel(modelId);
    setLoadedModels(await api.listLoadedModels());
  };

  const handleSelectRemoteModel = useCallback((peer: RemotePeer, model: RemoteModelSummary) => {
    setActiveRemoteModel({
      peer,
      model,
      qualifiedId: `${peer.slug}/${model.id}`,
    });
  }, []);

  const handleRemoteLoadAndSelect = useCallback(
    async (peer: RemotePeer, model: RemoteModelSummary) => {
      setIsLoadingModel(true);
      setBannerError(null);
      try {
        await api.remoteLoadModel(peer.node_id, model.id);
        const refreshed = await api.listLinkPeers();
        syncLinkPeers(refreshed);
        const updatedPeer = refreshed.find((p) => p.node_id === peer.node_id) ?? peer;
        const updatedModel = updatedPeer.loaded_models.find((m) => m.id === model.id);
        if (updatedModel) {
          setActiveRemoteModel({
            peer: updatedPeer,
            model: updatedModel,
            qualifiedId: `${updatedPeer.slug}/${updatedModel.id}`,
          });
        }
      } catch (e) {
        setBannerError(String(e));
      } finally {
        setIsLoadingModel(false);
      }
    },
    [syncLinkPeers],
  );

  const handleRequestRemoteConfigure = useCallback(
    async (peer: RemotePeer, model: RemoteModelSummary) => {
      setBannerError(null);
      try {
        const inspect = await api.remoteInspectModel(peer.node_id, model.id);
        setConfiguringRemote({ peer, inspect });
      } catch (e) {
        setBannerError(String(e));
      }
    },
    [],
  );

  const handleExecuteRemoteLoad = useCallback(
    async (
      peer: RemotePeer,
      model: LocalModelEntry,
      config: LoadConfig,
      rememberPreset: boolean,
    ) => {
      setConfiguringRemote(null);
      setIsLoadingModel(true);
      setBannerError(null);
      try {
        const res = await api.remoteLoadModel(peer.node_id, model.id, config, rememberPreset);
        const refreshed = await api.listLinkPeers();
        syncLinkPeers(refreshed);
        const updatedPeer = refreshed.find((p) => p.node_id === peer.node_id) ?? peer;
        const updatedModel = updatedPeer.loaded_models.find(
          (m) => m.id === res.model_id || m.id === model.id,
        );
        if (updatedModel) {
          setActiveRemoteModel({
            peer: updatedPeer,
            model: updatedModel,
            qualifiedId: `${updatedPeer.slug}/${updatedModel.id}`,
          });
        }
      } catch (e) {
        setBannerError(String(e));
      } finally {
        setIsLoadingModel(false);
      }
    },
    [syncLinkPeers],
  );

  const handleRemoteUnload = useCallback(
    async (peer: RemotePeer, model: RemoteModelSummary) => {
      setBannerError(null);
      try {
        await api.remoteUnloadModel(peer.node_id, model.id);
        const refreshed = await api.listLinkPeers();
        syncLinkPeers(refreshed);
      } catch (e) {
        setBannerError(String(e));
      }
    },
    [syncLinkPeers],
  );

  // Global keyboard shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && !e.shiftKey && e.key.toLowerCase() === 'l') {
        e.preventDefault();
        setQuickLoadOpen((v) => !v);
      } else if (e.ctrlKey && !e.shiftKey && e.key.toLowerCase() === 'r') {
        if (lastLoadedModel && !isLoadingModel) {
          e.preventDefault();
          handleRequestLoad(lastLoadedModel);
        }
      } else if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'r') {
        e.preventDefault();
        setTab('runtime');
      } else if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'h') {
        e.preventDefault();
        setTab('hardware');
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [lastLoadedModel, isLoadingModel, handleRequestLoad]);

  const activeLoaded = activeRemoteModel
    ? null
    : loadedModels.find((m) => m.model_id === activeModelId) ?? loadedModels[0] ?? null;

  const remoteActiveDownloadsCount = linkPeers
    .filter((p) => p.online && p.allow_remote_manage)
    .reduce((acc, p) => {
      const tracked = (remoteDownloads[p.node_id] ?? []).filter(
        (j) => j.status === 'downloading' || j.status === 'queued' || j.status === 'verifying',
      ).length;
      return acc + Math.max(p.active_downloads_count, tracked);
    }, 0);

  const primaryNav: { id: NavTab; label: string; icon: React.ReactNode; badge?: React.ReactNode }[] = [
    { id: 'benchmark', label: 'Benchmark', icon: <Gauge className="w-4 h-4" /> },
    { id: 'server', label: 'Model Server', icon: <Server className="w-4 h-4" /> },
    {
      id: 'link',
      label: 'Link',
      icon: <Link2 className="w-4 h-4" />,
      badge:
        linkStatus?.state === 'connected' ? (
          <span className="w-2 h-2 rounded-full bg-emerald-400 shrink-0" />
        ) : linkStatus?.state === 'needs_login' ? (
          <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse shrink-0" />
        ) : undefined,
    },
    { id: 'library', label: 'Library', icon: <Library className="w-4 h-4" /> },
    { id: 'search', label: 'Model Search', icon: <Search className="w-4 h-4" /> },
  ];

  const systemNav: { id: NavTab; label: string; icon: React.ReactNode; shortcut?: string }[] = [
    {
      id: 'runtime',
      label: 'Runtime',
      icon: <Activity className="w-4 h-4" />,
      shortcut: '⌃⇧R',
    },
    {
      id: 'hardware',
      label: 'Hardware',
      icon: <Cpu className="w-4 h-4" />,
      shortcut: '⌃⇧H',
    },
    { id: 'settings', label: 'Settings', icon: <Settings className="w-4 h-4" /> },
  ];

  return (
    <div className="h-screen w-screen flex flex-col bg-[var(--bg-canvas)] text-[var(--label-primary)] overflow-hidden">
      {/* Unified macOS Liquid Glass Toolbar */}
      <TopBar
        appVersion={appVersion}
        loadedModels={loadedModels}
        activeModelId={activeModelId}
        activeRemoteModel={activeRemoteModel}
        localModels={localModels}
        linkStatus={linkStatus}
        linkPeers={linkPeers}
        downloads={downloads}
        remoteActiveDownloadsCount={remoteActiveDownloadsCount}
        isLoadingModel={isLoadingModel}
        quickLoadOpen={quickLoadOpen}
        setQuickLoadOpen={setQuickLoadOpen}
        downloadsOpen={downloadsOpen}
        setDownloadsOpen={setDownloadsOpen}
        sidebarCollapsed={sidebarCollapsed}
        setSidebarCollapsed={setSidebarCollapsed}
        inspectorOpen={inspectorOpen}
        setInspectorOpen={setInspectorOpen}
        onSelectActiveModel={(id) => {
          setActiveRemoteModel(null);
          setActiveModelId(id);
        }}
        onSelectRemoteModel={handleSelectRemoteModel}
        onRemoteLoadModel={handleRemoteLoadAndSelect}
        onConfigureRemoteModel={handleRequestRemoteConfigure}
        onRemoteUnloadModel={handleRemoteUnload}
        onConfigureModel={handleRequestLoad}
        onUnloadModel={handleUnloadModel}
        onReloadLast={() => lastLoadedModel && handleRequestLoad(lastLoadedModel)}
        canReloadLast={Boolean(lastLoadedModel)}
        onOpenLinkTab={() => setTab('link')}
      />

      {/* App Update Available Banner */}
      {updateInfo?.update_available && !updateBannerDismissed && (
        <div className="px-4 py-2 bg-[var(--accent-soft)] border-b border-[var(--accent-border)] flex flex-wrap items-center justify-between gap-3 text-[12px] text-[var(--label-primary)] shrink-0">
          <div className="flex items-center gap-2.5 min-w-0">
            <ArrowDownToLine className="w-4 h-4 text-[var(--accent-text)] shrink-0" />
            <span className="truncate">
              <strong>Bibulm v{updateInfo.latest_version}</strong> is available (current v
              {updateInfo.current_version}).
              {installingUpdate && updateProgress && (
                <span className="ml-2 text-[var(--accent-text)] font-mono tabular-nums">
                  {updateProgress.message} ({updateProgress.percent.toFixed(0)}%)
                </span>
              )}
            </span>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {updateInfo.asset_url && (
              <button
                onClick={() => handleInstallAppUpdate(updateInfo)}
                disabled={installingUpdate}
                className="bibulm-btn-primary h-7"
              >
                {installingUpdate ? 'Updating…' : 'Update & Restart'}
              </button>
            )}
            <button
              onClick={() => setTab('settings')}
              className="bibulm-btn-secondary h-7"
            >
              Details
            </button>
            {!installingUpdate && (
              <button
                onClick={() => setUpdateBannerDismissed(true)}
                className="apple-btn-ghost w-7 h-7"
                title="Dismiss"
                aria-label="Dismiss update banner"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>
      )}

      {/* Global Error Banner */}
      {bannerError && (
        <div className="px-4 py-2 bg-[var(--signal-red-soft)] border-b border-[var(--signal-red-border)] flex items-center justify-between gap-3 text-[12px] text-[var(--signal-red)] shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span className="truncate font-medium">{bannerError}</span>
          </div>
          <button
            onClick={() => setBannerError(null)}
            className="apple-btn-ghost w-7 h-7 text-[var(--signal-red)] shrink-0"
            aria-label="Dismiss error"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Middle 3-Column Shell */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left macOS Source List Sidebar */}
        <nav
          aria-label="Workspace sections"
          className={`${
            sidebarCollapsed ? 'w-[56px]' : 'w-[216px]'
          } shrink-0 bg-[var(--bg-sidebar)] border-r border-[var(--border-hairline)] flex flex-col py-3 px-2.5 transition-all duration-150 select-none overflow-y-auto`}
        >
          <div className="space-y-0.5">
            {!sidebarCollapsed && (
              <div className="px-2.5 pb-1.5 text-[11px] font-semibold uppercase tracking-wider text-[var(--label-tertiary)]">
                Workspace
              </div>
            )}
            {primaryNav.map((item) => {
              const active = tab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setTab(item.id)}
                  title={item.label}
                  aria-current={active ? 'page' : undefined}
                  className={`w-full h-8 px-2.5 rounded-[8px] flex items-center justify-between gap-2 text-[13px] transition-colors cursor-pointer ${
                    active
                      ? 'bg-[var(--accent-soft)] text-[var(--accent-text)] font-semibold'
                      : 'text-[var(--label-secondary)] hover:text-[var(--label-primary)] hover:bg-[var(--bg-control)] font-medium'
                  }`}
                >
                  <span className="flex items-center gap-2.5 min-w-0">
                    <span className="shrink-0">{item.icon}</span>
                    {!sidebarCollapsed && <span className="truncate">{item.label}</span>}
                  </span>
                  {!sidebarCollapsed && item.badge}
                </button>
              );
            })}
          </div>

          <div className="space-y-0.5 mt-4 pt-3 border-t border-[var(--border-hairline)]">
            {!sidebarCollapsed && (
              <div className="px-2.5 pb-1.5 text-[11px] font-semibold uppercase tracking-wider text-[var(--label-tertiary)]">
                System
              </div>
            )}
            {systemNav.map((item) => {
              const active = tab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setTab(item.id)}
                  title={item.label}
                  aria-current={active ? 'page' : undefined}
                  className={`w-full h-8 px-2.5 rounded-[8px] flex items-center justify-between text-[13px] transition-colors cursor-pointer ${
                    active
                      ? 'bg-[var(--accent-soft)] text-[var(--accent-text)] font-semibold'
                      : 'text-[var(--label-secondary)] hover:text-[var(--label-primary)] hover:bg-[var(--bg-control)] font-medium'
                  }`}
                >
                  <span className="flex items-center gap-2.5 min-w-0">
                    <span className="shrink-0">{item.icon}</span>
                    {!sidebarCollapsed && <span className="truncate">{item.label}</span>}
                  </span>
                  {!sidebarCollapsed && item.shortcut && (
                    <span
                      className={`text-[10px] font-mono ${
                        active ? 'text-[var(--accent-text)] opacity-80' : 'text-[var(--label-tertiary)]'
                      }`}
                    >
                      {item.shortcut}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </nav>

        {/* Main Workspace View */}
        <main className="flex-1 overflow-hidden bg-[var(--bg-canvas)]">
          {tab === 'benchmark' && (
            <BenchmarkView
              activeModel={activeLoaded}
              activeRemoteModel={activeRemoteModel}
              localModels={localModels}
              runtimes={runtimes}
              settings={settings}
              sampling={sampling}
              reasoningEffort={reasoningEffort}
              telemetry={telemetry}
              onOpenQuickLoad={() => setQuickLoadOpen(true)}
            />
          )}
          {tab === 'server' && settings && (
            <ServerView
              status={serverStatus}
              settings={settings}
              loadedModels={loadedModels}
              localModels={localModels}
              linkStatus={linkStatus}
              linkPeers={linkPeers}
              logs={logs}
              onToggleServer={async (start) => {
                try {
                  setServerStatus(start ? await api.startServer() : await api.stopServer());
                } catch (e) {
                  setBannerError(String(e));
                }
              }}
              onSaveSettings={handleSaveSettings}
              onOpenQuickLoad={() => setQuickLoadOpen(true)}
              onOpenLinkTab={() => setTab('link')}
              onUnloadModel={handleUnloadModel}
              onClearLogs={async () => {
                await api.clearServerLogs();
                setLogs([]);
              }}
            />
          )}
          {tab === 'link' && settings && (
            <LinkView
              status={linkStatus}
              peers={linkPeers}
              settings={settings}
              activeRemoteModel={activeRemoteModel}
              gatewayPort={serverStatus?.config.port ?? settings.server.port}
              onSaveSettings={handleSaveSettings}
              onStatusChange={setLinkStatus}
              onPeersChange={syncLinkPeers}
              onSelectRemoteModel={handleSelectRemoteModel}
              onConfigureRemoteModel={handleRequestRemoteConfigure}
              onManagePeerRuntimes={(peerId) => {
                setRuntimeTargetPeerId(peerId);
                setTab('runtime');
              }}
              onDownloadPeerModels={(peerId) => {
                setSearchTargetPeerId(peerId);
                setTab('search');
              }}
              onOpenPeerDownloads={(peerId) => {
                setDownloadsTargetPeerId(peerId);
                refreshRemoteDownloads(peerId);
                setDownloadsOpen(true);
              }}
              onNavigateBenchmark={() => setTab('benchmark')}
            />
          )}
          {tab === 'library' && settings && (
            <LibraryView
              models={localModels}
              loadedModels={loadedModels}
              settings={settings}
              scanning={scanningLibrary}
              onRescan={refreshLibrary}
              onSaveSettings={handleSaveSettings}
              onConfigureLoad={handleRequestLoad}
            />
          )}
          {tab === 'search' && (
            <SearchView
              linkPeers={linkPeers}
              selectedPeerId={searchTargetPeerId}
              onSelectPeerId={setSearchTargetPeerId}
              onDownloadStarted={() => {
                setDownloadsTargetPeerId(null);
                refreshDownloads();
                setDownloadsOpen(true);
              }}
              onRemoteDownloadStarted={(peerId) => {
                setDownloadsTargetPeerId(peerId);
                refreshRemoteDownloads(peerId);
                setDownloadsOpen(true);
              }}
            />
          )}
          {tab === 'runtime' && settings && (
            <RuntimeView
              installed={runtimes}
              settings={settings}
              linkPeers={linkPeers}
              selectedPeerId={runtimeTargetPeerId}
              onSelectPeerId={setRuntimeTargetPeerId}
              onRefreshInstalled={refreshRuntimes}
              onRefreshPeers={() => {
                api.refreshLinkPeers().then(syncLinkPeers).catch(() => {});
              }}
              onOpenPeerDownloads={(peerId) => {
                setDownloadsTargetPeerId(peerId);
                refreshRemoteDownloads(peerId);
                setDownloadsOpen(true);
              }}
              onSaveSettings={handleSaveSettings}
              onSetActiveRuntime={async (id) => {
                setSettings(await api.setActiveRuntime(id));
              }}
            />
          )}
          {tab === 'hardware' && settings && (
            <HardwareView
              profile={hwProfile}
              telemetry={telemetry}
              settings={settings}
              onSaveSettings={handleSaveSettings}
              onRefreshProfile={async () => setHwProfile(await api.refreshHardwareProfile())}
            />
          )}
          {tab === 'settings' && settings && (
            <SettingsView
              settings={settings}
              onSaveSettings={handleSaveSettings}
              onRescanLibrary={refreshLibrary}
              appVersion={appVersion}
              updateInfo={updateInfo}
              updateProgress={updateProgress}
              checkingUpdate={checkingUpdate}
              installingUpdate={installingUpdate}
              updateError={updateError}
              onCheckUpdate={handleCheckAppUpdate}
              onInstallUpdate={handleInstallAppUpdate}
            />
          )}
        </main>

        {/* Right Collapsible Inference Parameters Inspector */}
        {inspectorOpen && (
          <InspectorPanel
            sampling={sampling}
            onChange={setSampling}
            onPersistDefaults={(next) => {
              if (!settings) return;
              handleSaveSettings({ ...settings, default_sampling: next });
            }}
            reasoningEffort={reasoningEffort}
            onReasoningEffortChange={setReasoningEffort}
          />
        )}
      </div>

      {/* Slim Always-Visible Bottom Resource Status Bar */}
      <footer className="h-7 shrink-0 bg-[var(--bg-sidebar)] border-t border-[var(--border-hairline)] px-3.5 flex items-center justify-between text-[11px] font-mono tabular-nums text-[var(--label-secondary)] select-none">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setTab('server')}
            className="flex items-center gap-1.5 hover:text-[var(--label-primary)] transition-colors cursor-pointer"
          >
            <span
              className={`w-2 h-2 rounded-full ${
                serverStatus?.running ? 'bg-[var(--signal-green-fill)]' : 'bg-[var(--label-quaternary)]'
              }`}
            />
            <span>
              Server :{serverStatus?.config.port ?? settings?.server.port ?? 1234}
            </span>
          </button>
          <span className="text-[var(--border-strong)]">|</span>
          <button
            onClick={() => setTab('link')}
            className="flex items-center gap-1.5 hover:text-[var(--label-primary)] transition-colors cursor-pointer"
            title="Bibulm Link (BYO Tailscale)"
          >
            <span
              className={`w-2 h-2 rounded-full ${
                linkStatus?.state === 'connected'
                  ? 'bg-[var(--signal-green-fill)]'
                  : linkStatus?.state === 'needs_login'
                    ? 'bg-[var(--signal-amber-fill)] animate-pulse'
                    : 'bg-[var(--label-quaternary)]'
              }`}
            />
            <span>
              Link:{' '}
              {linkStatus?.state === 'connected'
                ? `${linkPeers.length} peer${linkPeers.length === 1 ? '' : 's'}`
                : linkStatus?.state === 'needs_login'
                  ? 'Sign in'
                  : 'Off'}
            </span>
          </button>
          <span className="text-[var(--border-strong)]">|</span>
          <button
            onClick={() => setTab('runtime')}
            className="hover:text-[var(--label-primary)] transition-colors truncate max-w-[200px] cursor-pointer"
          >
            Runtime:{' '}
            {runtimes.find((r) => r.id === settings?.active_runtime_id)?.tag ??
              runtimes[0]?.tag ??
              'None installed'}
          </button>
          <span className="text-[var(--border-strong)]">|</span>
          <button
            onClick={() => setTab('settings')}
            className="hover:text-[var(--label-primary)] transition-colors flex items-center gap-1 cursor-pointer"
            title="Application version & updates"
          >
            <span>v{appVersion}</span>
            {updateInfo?.update_available && (
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent)]" />
            )}
          </button>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setTab('hardware')}
            className="flex items-center gap-2 hover:text-[var(--label-primary)] transition-colors cursor-pointer"
          >
            <span>
              VRAM{' '}
              <strong className="text-[var(--signal-green)]">
                {formatBytes(telemetry?.vram_used_bytes ?? 0)}
              </strong>
              /{formatBytes(telemetry?.vram_total_bytes ?? 0)}
            </span>
            {telemetry && telemetry.shared_gpu_mem_used_bytes > 0 && (
              <span
                className={
                  telemetry.wddm_spillover_warning
                    ? 'text-[var(--signal-red)] font-semibold'
                    : 'text-[var(--label-tertiary)]'
                }
              >
                ({formatBytes(telemetry.shared_gpu_mem_used_bytes)} shared)
              </span>
            )}
          </button>
          <span className="text-[var(--border-strong)]">|</span>
          <span>
            RAM{' '}
            <strong className="text-[var(--accent-text)]">
              {formatBytes(telemetry?.ram_used_bytes ?? 0)}
            </strong>
            /{formatBytes(telemetry?.ram_total_bytes ?? 0)}
          </span>
          <span className="text-[var(--border-strong)]">|</span>
          <span>CPU {(telemetry?.cpu_utilization_pct ?? 0).toFixed(0)}%</span>
          {telemetry?.gpu_temp_celsius != null && (
            <>
              <span className="text-[var(--border-strong)]">|</span>
              <span>{telemetry.gpu_temp_celsius}°C</span>
            </>
          )}
        </div>
      </footer>

      {/* Downloads Popover */}
      <DownloadsPopover
        open={downloadsOpen}
        onClose={() => setDownloadsOpen(false)}
        downloads={downloads}
        modelsDir={settings?.models_dirs[0] ?? ''}
        linkPeers={linkPeers}
        remoteDownloads={remoteDownloads}
        selectedPeerId={downloadsTargetPeerId}
        onSelectPeerId={setDownloadsTargetPeerId}
        onRefresh={refreshDownloads}
        onRefreshRemote={refreshRemoteDownloads}
        onRefreshPeers={() => {
          api.refreshLinkPeers().then(syncLinkPeers).catch(() => {});
        }}
        onModelReadyScan={() => {
          refreshLibrary();
          setTab('library');
        }}
      />

      {/* Full Load Configuration Modal (Local) */}
      {configuringModel && settings && (
        <LoadConfigModal
          model={configuringModel}
          localModels={localModels}
          runtimes={runtimes}
          settings={settings}
          cpuLogicalCores={hwProfile?.cpu.logical_cores ?? 8}
          onCancel={() => setConfiguringModel(null)}
          onConfirmLoad={(cfg, remember, forceOverride) =>
            handleExecuteLoad(configuringModel, cfg, remember, forceOverride)
          }
        />
      )}

      {/* Full Load Configuration Modal (Remote Link Peer) */}
      {configuringRemote && settings && (
        <LoadConfigModal
          model={configuringRemote.inspect.model}
          localModels={localModels}
          runtimes={runtimes}
          settings={settings}
          cpuLogicalCores={configuringRemote.inspect.cpu_logical_cores}
          remoteTarget={{
            peer: configuringRemote.peer,
            inspect: configuringRemote.inspect,
            onRefreshInspect: async () => {
              const fresh = await api.remoteInspectModel(
                configuringRemote.peer.node_id,
                configuringRemote.inspect.model.id,
              );
              setConfiguringRemote((cur) => (cur ? { ...cur, inspect: fresh } : cur));
              return fresh;
            },
          }}
          onCancel={() => setConfiguringRemote(null)}
          onConfirmLoad={(cfg, remember) =>
            handleExecuteRemoteLoad(
              configuringRemote.peer,
              configuringRemote.inspect.model,
              cfg,
              remember,
            )
          }
        />
      )}
    </div>
  );
};

export default App;
