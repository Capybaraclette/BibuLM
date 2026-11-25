import React, { useCallback, useEffect, useState } from 'react';
import { open } from '@tauri-apps/plugin-dialog';
import {
  Activity,
  Check,
  ChevronDown,
  ChevronRight,
  Cpu,
  Download,
  FolderOpen,
  FolderPlus,
  Laptop,
  Loader2,
  Lock,
  RefreshCw,
  Server,
  Sparkles,
  Trash2,
} from 'lucide-react';
import type {
  AppSettings,
  InstalledRuntimeRecord,
  LlamaRelease,
  RemotePeer,
  RemoteRuntimesResponse,
  RuntimeBackendVariant,
} from '../types';
import { api, formatBytes, formatRelativeDate } from '../api';
import { Toggle } from '../components/Toggle';
import { PageHeader } from '../components/PageHeader';

interface RuntimeViewProps {
  installed: InstalledRuntimeRecord[];
  settings: AppSettings;
  linkPeers: RemotePeer[];
  selectedPeerId: string | null;
  onSelectPeerId: (id: string | null) => void;
  onRefreshInstalled: () => void;
  onRefreshPeers: () => void;
  onOpenPeerDownloads: (peerId: string) => void;
  onSaveSettings: (next: AppSettings) => void;
  onSetActiveRuntime: (id: string) => void;
}

function formatPlatformLabel(raw?: string | null): string {
  if (!raw) return 'this platform';
  const low = raw.toLowerCase();
  if (low.includes('windows') && (low.includes('x86_64') || low.includes('x64'))) {
    return 'Windows x64';
  }
  if (low.includes('windows') && (low.includes('aarch64') || low.includes('arm64'))) {
    return 'Windows ARM64';
  }
  if (low.includes('linux') && (low.includes('x86_64') || low.includes('x64'))) {
    return 'Linux x64';
  }
  if (low.includes('linux') && (low.includes('aarch64') || low.includes('arm64'))) {
    return 'Linux ARM64';
  }
  if (low.includes('macos') && (low.includes('aarch64') || low.includes('arm64'))) {
    return 'macOS arm64';
  }
  if (low.includes('macos')) return 'macOS x64';
  return raw;
}

export const RuntimeView: React.FC<RuntimeViewProps> = ({
  installed,
  settings,
  linkPeers,
  selectedPeerId,
  onSelectPeerId,
  onRefreshInstalled,
  onRefreshPeers,
  onOpenPeerDownloads,
  onSaveSettings,
  onSetActiveRuntime,
}) => {
  const [releases, setReleases] = useState<LlamaRelease[]>([]);
  const [selectedTag, setSelectedTag] = useState<string>('');
  const [loadingReleases, setLoadingReleases] = useState(false);
  const [releaseError, setReleaseError] = useState<string | null>(null);
  const [compatibleOnly, setCompatibleOnly] = useState(true);
  const [showNotes, setShowNotes] = useState(false);
  const [installingId, setInstallingId] = useState<string | null>(null);
  const [installError, setInstallError] = useState<string | null>(null);
  const [remoteNotice, setRemoteNotice] = useState<string | null>(null);

  const [remoteData, setRemoteData] = useState<RemoteRuntimesResponse | null>(null);
  const [loadingRemote, setLoadingRemote] = useState(false);

  const onlinePeers = linkPeers.filter((p) => p.online);
  const selectedPeer = selectedPeerId
    ? onlinePeers.find((p) => p.node_id === selectedPeerId) ?? null
    : null;
  const isRemote = Boolean(selectedPeer);
  const remoteManageable = Boolean(selectedPeer?.allow_remote_manage);

  useEffect(() => {
    if (selectedPeerId && !selectedPeer) {
      onSelectPeerId(null);
    }
  }, [selectedPeerId, selectedPeer, onSelectPeerId]);

  const loadDataForTarget = useCallback(async () => {
    setInstallError(null);
    setReleaseError(null);
    setRemoteNotice(null);

    if (selectedPeer) {
      if (!selectedPeer.allow_remote_manage) {
        setRemoteData(null);
        setReleases([]);
        return;
      }
      setLoadingRemote(true);
      setLoadingReleases(true);
      try {
        const [rtRes, relRes] = await Promise.all([
          api.remoteListRuntimes(selectedPeer.node_id),
          api.remoteFetchRuntimeReleases(selectedPeer.node_id),
        ]);
        setRemoteData(rtRes);
        setReleases(relRes);
        setSelectedTag((cur) =>
          relRes.some((r) => r.tag_name === cur) ? cur : (relRes[0]?.tag_name ?? ''),
        );
      } catch (e) {
        setReleaseError(String(e));
      } finally {
        setLoadingRemote(false);
        setLoadingReleases(false);
      }
      return;
    }

    setRemoteData(null);
    setLoadingReleases(true);
    try {
      const list = await api.fetchGithubReleases();
      setReleases(list);
      setSelectedTag((cur) =>
        list.some((r) => r.tag_name === cur) ? cur : (list[0]?.tag_name ?? ''),
      );
    } catch (e) {
      setReleaseError(String(e));
    } finally {
      setLoadingReleases(false);
    }
  }, [selectedPeer?.node_id, selectedPeer?.allow_remote_manage]);

  useEffect(() => {
    loadDataForTarget();
  }, [loadDataForTarget]);

  useEffect(() => {
    if (selectedPeer && selectedPeer.allow_remote_manage) {
      api
        .remoteListRuntimes(selectedPeer.node_id)
        .then(setRemoteData)
        .catch(() => {});
    }
  }, [selectedPeer?.node_id, selectedPeer?.allow_remote_manage, selectedPeer?.runtimes_count]);

  const effectiveInstalled = selectedPeer ? (remoteData?.installed ?? []) : installed;
  const effectiveActiveId = selectedPeer
    ? (remoteData?.active_runtime_id ?? null)
    : settings.active_runtime_id;
  const targetPlatformLabel = selectedPeer
    ? formatPlatformLabel(remoteData?.platform || selectedPeer.platform)
    : 'this platform';

  const selectedRelease =
    releases.find((r) => r.tag_name === selectedTag) ?? releases[0] ?? null;

  const visibleVariants = (selectedRelease?.variants ?? []).filter((v) =>
    compatibleOnly ? v.compatible : true,
  );

  const handleInstall = async (variant: RuntimeBackendVariant) => {
    setInstallingId(variant.id);
    setInstallError(null);
    setRemoteNotice(null);
    try {
      if (selectedPeer) {
        await api.remoteInstallRuntime(
          selectedPeer.node_id,
          variant,
          selectedRelease?.release_notes ?? null,
        );
        setRemoteNotice(
          `Started installing ${variant.tag} (${variant.backend}) on ${selectedPeer.device_name}.`,
        );
        onOpenPeerDownloads(selectedPeer.node_id);
      } else {
        const rec = await api.installRuntime(variant, selectedRelease?.release_notes ?? null);
        await onRefreshInstalled();
        if (!settings.active_runtime_id) {
          onSetActiveRuntime(rec.id);
        }
      }
    } catch (e) {
      setInstallError(String(e));
    } finally {
      setInstallingId(null);
    }
  };

  const handleSetActive = async (runtimeId: string) => {
    setInstallError(null);
    if (selectedPeer) {
      try {
        await api.remoteSetActiveRuntime(selectedPeer.node_id, runtimeId);
        const fresh = await api.remoteListRuntimes(selectedPeer.node_id);
        setRemoteData(fresh);
        onRefreshPeers();
      } catch (e) {
        setInstallError(String(e));
      }
    } else {
      onSetActiveRuntime(runtimeId);
    }
  };

  const handleUninstall = async (runtimeId: string) => {
    setInstallError(null);
    try {
      if (selectedPeer) {
        await api.remoteUninstallRuntime(selectedPeer.node_id, runtimeId);
        const fresh = await api.remoteListRuntimes(selectedPeer.node_id);
        setRemoteData(fresh);
        onRefreshPeers();
      } else {
        await api.uninstallRuntime(runtimeId);
        onRefreshInstalled();
      }
    } catch (e) {
      setInstallError(String(e));
    }
  };

  const handleAddCustom = async () => {
    const selected = await open({
      multiple: false,
      title: 'Select custom llama-server.exe',
      filters: [{ name: 'Executable', extensions: ['exe'] }],
    });
    if (typeof selected === 'string') {
      try {
        await api.registerCustomRuntime('Custom llama.cpp', selected);
        onRefreshInstalled();
      } catch (e) {
        setInstallError(String(e));
      }
    }
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <PageHeader
        icon={<Activity className="w-4 h-4" />}
        title="Runtime Manager"
        subtitle="Download official ggml-org/llama.cpp builds (CUDA + CUDART, Vulkan, ROCm, SYCL, OpenVINO, CPU) locally or on a linked server."
        actions={
          <>
            {!isRemote && (
              <button onClick={handleAddCustom} className="bibulm-btn-secondary">
                <FolderPlus className="w-3.5 h-3.5" />
                <span>Add Custom Binary</span>
              </button>
            )}
            <button
              onClick={loadDataForTarget}
              disabled={loadingReleases}
              className="bibulm-btn-secondary"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loadingReleases ? 'animate-spin' : ''}`} />
              <span>Check GitHub Releases</span>
            </button>
          </>
        }
      />

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-5xl mx-auto space-y-5">
          {/* Target Machine Selector */}
          {onlinePeers.length > 0 && (
            <div className="bibulm-card p-3 flex flex-wrap items-center gap-2 text-[12px]">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--label-tertiary)] mr-1">
                Target Machine
              </span>
              <div className="apple-segmented flex-wrap">
                <button
                  type="button"
                  onClick={() => onSelectPeerId(null)}
                  className={`apple-segmented-item ${
                    !selectedPeer ? 'apple-segmented-item-active' : ''
                  }`}
                >
                  <Laptop className="w-3.5 h-3.5" />
                  <span>This Machine</span>
                </button>
                {onlinePeers.map((peer) => {
                  const active = selectedPeer?.node_id === peer.node_id;
                  return (
                    <button
                      key={peer.node_id}
                      type="button"
                      onClick={() => onSelectPeerId(peer.node_id)}
                      title={
                        peer.allow_remote_manage
                          ? `${peer.device_name} (${formatPlatformLabel(peer.platform)})`
                          : 'Remote management disabled on this peer'
                      }
                      className={`apple-segmented-item ${
                        active ? 'apple-segmented-item-active' : ''
                      }`}
                    >
                      {peer.is_headless ? (
                        <Server className="w-3.5 h-3.5" />
                      ) : (
                        <Laptop className="w-3.5 h-3.5" />
                      )}
                      <span>{peer.device_name}</span>
                      {peer.platform && (
                        <span className="text-[10px] font-mono opacity-75">
                          ({formatPlatformLabel(peer.platform)})
                        </span>
                      )}
                      {!peer.allow_remote_manage && (
                        <Lock className="w-3 h-3 text-[var(--signal-amber)]" />
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {selectedPeer && !remoteManageable ? (
            <div className="rounded-xl bg-[var(--signal-amber-soft)] border border-[var(--signal-amber-border)] p-4 text-[12px] text-[var(--signal-amber)] flex items-start gap-2.5">
              <Lock className="w-4 h-4 shrink-0 mt-0.5" />
              <div>
                <div className="font-semibold">
                  Remote runtime &amp; model management is disabled on {selectedPeer.device_name}
                </div>
                <p className="text-[var(--label-secondary)] mt-1">
                  Enable <strong>Allow remote install &amp; downloads</strong> in that device&apos;s
                  BibuLM Link settings (or restart the headless CLI without{' '}
                  <code className="font-mono">--no-remote-manage</code>).
                </p>
              </div>
            </div>
          ) : (
            <>
              {selectedPeer && (
                <div className="rounded-xl bg-[var(--accent-soft)] border border-[var(--accent-border)] px-4 py-3 text-[12px] text-[var(--label-primary)] flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <Cpu className="w-4 h-4 text-[var(--accent-text)] shrink-0" />
                    <span className="truncate">
                      Managing <strong>{selectedPeer.device_name}</strong> ({targetPlatformLabel})
                      {remoteData?.gpu_summary
                        ? ` — ${
                            remoteData.gpu_summary
                              .split('\n')
                              .find((l) => l.startsWith('GPU #')) ??
                            remoteData.gpu_summary.split('\n')[0]
                          }`
                        : ''}
                    </span>
                  </div>
                  {loadingRemote && (
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-[var(--accent-text)]" />
                  )}
                </div>
              )}

              {/* Active Runtime Selection Card */}
              <div className="bibulm-card divide-y divide-[var(--border-hairline)] text-[12px]">
                <div className="p-4 flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="font-semibold text-[var(--label-primary)]">
                      {selectedPeer
                        ? `Active Default Runtime on ${selectedPeer.device_name}`
                        : 'Active Default GGUF Runtime'}
                    </div>
                    <div className="text-[11px] text-[var(--label-secondary)]">
                      Used for JIT loading, Model Server, and Quick Load unless overridden per-model.
                    </div>
                  </div>
                  <select
                    value={effectiveActiveId ?? ''}
                    onChange={(e) => e.target.value && handleSetActive(e.target.value)}
                    className="apple-select min-w-[260px]"
                  >
                    {effectiveInstalled.length === 0 && (
                      <option value="">No runtime installed yet</option>
                    )}
                    {effectiveInstalled.map((rt) => (
                      <option key={rt.id} value={rt.id}>
                        {rt.display_name} ({rt.tag})
                      </option>
                    ))}
                  </select>
                </div>

                {!isRemote && (
                  <div className="p-4 flex items-center justify-between">
                    <div>
                      <div className="font-medium text-[var(--label-primary)]">
                        Check for new llama.cpp releases on launch
                      </div>
                      <div className="text-[11px] text-[var(--label-secondary)]">
                        Queries GitHub Releases directly from ggml-org/llama.cpp
                      </div>
                    </div>
                    <Toggle
                      checked={settings.auto_check_runtime_updates}
                      onChange={(v) =>
                        onSaveSettings({ ...settings, auto_check_runtime_updates: v })
                      }
                    />
                  </div>
                )}
              </div>

              {remoteNotice && (
                <div className="rounded-xl bg-[var(--signal-green-soft)] border border-[var(--signal-green-border)] p-3.5 text-[12px] text-[var(--signal-green)]">
                  {remoteNotice}
                </div>
              )}

              {installError && (
                <div className="rounded-xl bg-[var(--signal-red-soft)] border border-[var(--signal-red-border)] p-3.5 text-[12px] text-[var(--signal-red)]">
                  {installError}
                </div>
              )}

              {/* Installed Runtimes */}
              {effectiveInstalled.length > 0 && (
                <div className="space-y-2.5">
                  <h2 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--label-tertiary)]">
                    Installed Builds ({effectiveInstalled.length})
                  </h2>
                  <div className="bibulm-card divide-y divide-[var(--border-hairline)]">
                    {effectiveInstalled.map((rt) => {
                      const isActive = effectiveActiveId === rt.id;
                      return (
                        <div
                          key={rt.id}
                          className="p-3.5 flex flex-wrap items-center justify-between gap-3 text-[12px]"
                        >
                          <div className="space-y-0.5 min-w-0">
                            <div className="flex items-center gap-2 flex-wrap">
                              <Activity className="w-3.5 h-3.5 text-[var(--accent-text)]" />
                              <span className="font-semibold text-[var(--label-primary)]">
                                {rt.display_name}
                              </span>
                              <span className="apple-badge font-mono text-[10px]">{rt.tag}</span>
                              {rt.is_custom && (
                                <span className="apple-badge apple-badge-purple text-[10px]">
                                  Custom
                                </span>
                              )}
                              {isActive && (
                                <span className="apple-badge apple-badge-green text-[10px] font-semibold">
                                  ACTIVE
                                </span>
                              )}
                            </div>
                            <div className="text-[11px] text-[var(--label-tertiary)] font-mono truncate tabular-nums">
                              {rt.server_exe} · {formatBytes(rt.size_bytes)}
                            </div>
                          </div>
                          <div className="flex items-center gap-1.5 shrink-0">
                            {!isActive && (
                              <button
                                type="button"
                                onClick={() => handleSetActive(rt.id)}
                                className="bibulm-btn-secondary h-7 px-2.5 text-[11px]"
                              >
                                Set Active
                              </button>
                            )}
                            {!isRemote && (
                              <button
                                type="button"
                                onClick={() => api.revealInExplorer(rt.install_dir)}
                                className="apple-btn-ghost w-7 h-7"
                                title="Open folder"
                                aria-label="Open runtime folder"
                              >
                                <FolderOpen className="w-3.5 h-3.5" />
                              </button>
                            )}
                            <button
                              type="button"
                              onClick={() => handleUninstall(rt.id)}
                              className="apple-btn-ghost w-7 h-7 hover:text-[var(--signal-red)]"
                              title="Remove runtime"
                              aria-label="Remove runtime"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Upstream GitHub Releases Browser */}
              <div className="space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2.5">
                    <h2 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--label-tertiary)]">
                      Official GitHub Releases
                    </h2>
                    {releases.length > 0 && (
                      <select
                        value={selectedRelease?.tag_name ?? ''}
                        onChange={(e) => setSelectedTag(e.target.value)}
                        className="apple-select font-mono"
                      >
                        {releases.map((r, i) => (
                          <option key={r.tag_name} value={r.tag_name}>
                            {r.tag_name} {i === 0 ? '(Latest)' : ''} —{' '}
                            {formatRelativeDate(r.published_at)}
                          </option>
                        ))}
                      </select>
                    )}
                  </div>

                  <select
                    value={compatibleOnly ? 'compatible' : 'all'}
                    onChange={(e) => setCompatibleOnly(e.target.value === 'compatible')}
                    className="apple-select"
                  >
                    <option value="compatible">
                      {selectedPeer
                        ? `Compatible with ${selectedPeer.device_name}`
                        : 'Compatible with my hardware'}
                    </option>
                    <option value="all">
                      {selectedPeer
                        ? `Show all ${targetPlatformLabel} builds`
                        : 'Show all builds for this platform'}
                    </option>
                  </select>
                </div>

                {loadingReleases ? (
                  <div className="bibulm-card py-12 flex flex-col items-center gap-2 text-[12px] text-[var(--label-secondary)]">
                    <Loader2 className="w-5 h-5 animate-spin text-[var(--accent)]" />
                    <span>Fetching latest builds from github.com/ggml-org/llama.cpp…</span>
                  </div>
                ) : releaseError ? (
                  <div className="rounded-xl bg-[var(--signal-red-soft)] border border-[var(--signal-red-border)] p-4 text-[12px] text-[var(--signal-red)]">
                    Failed to query GitHub Releases: {releaseError}. You can configure a GitHub PAT
                    in Settings if rate-limited.
                  </div>
                ) : selectedRelease ? (
                  <div className="bibulm-card overflow-hidden">
                    <div className="divide-y divide-[var(--border-hairline)]">
                      {visibleVariants.map((v) => {
                        const alreadyInstalled = effectiveInstalled.some((r) => r.id === v.id);
                        const isBusy = installingId === v.id;
                        return (
                          <div
                            key={v.id}
                            className="p-4 flex flex-wrap items-center justify-between gap-3 text-[12px] hover:bg-[var(--bg-control)] transition-colors"
                          >
                            <div className="space-y-1 min-w-0">
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className="font-semibold text-[var(--label-primary)]">
                                  {v.display_name}
                                </span>
                                <span className="apple-badge font-mono text-[10px]">{v.tag}</span>
                                {v.recommended && (
                                  <span className="apple-badge apple-badge-accent text-[10px] font-medium">
                                    <Sparkles className="w-3 h-3" /> Recommended
                                  </span>
                                )}
                                {!v.compatible && (
                                  <span className="apple-badge apple-badge-amber text-[10px]">
                                    Driver/GPU mismatch
                                  </span>
                                )}
                              </div>
                              <div className="text-[11px] text-[var(--label-secondary)]">
                                {v.description}
                              </div>
                              <div className="text-[10px] text-[var(--label-tertiary)] font-mono">
                                Assets: {v.assets.map((a) => a.name).join(' + ')}
                              </div>
                            </div>

                            <div className="shrink-0">
                              {alreadyInstalled ? (
                                <span className="inline-flex items-center gap-1.5 text-[var(--signal-green)] font-medium px-3 py-1.5">
                                  <Check className="w-4 h-4" /> Installed
                                </span>
                              ) : (
                                <button
                                  type="button"
                                  disabled={installingId !== null}
                                  onClick={() => handleInstall(v)}
                                  className="bibulm-btn-primary px-3.5 font-semibold"
                                >
                                  {isBusy ? (
                                    <>
                                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                      <span>{selectedPeer ? 'Starting…' : 'Installing…'}</span>
                                    </>
                                  ) : (
                                    <>
                                      <Download className="w-3.5 h-3.5" />
                                      <span>
                                        {selectedPeer ? 'Install on Server' : 'Install'}{' '}
                                        {formatBytes(v.total_size_bytes)}
                                      </span>
                                    </>
                                  )}
                                </button>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {selectedRelease.release_notes && (
                      <div className="border-t border-[var(--border-hairline)] px-4 py-2.5 bg-[var(--bg-surface-inset)]">
                        <button
                          type="button"
                          onClick={() => setShowNotes(!showNotes)}
                          className="flex items-center gap-1 text-[12px] text-[var(--label-secondary)] hover:text-[var(--label-primary)] cursor-pointer"
                        >
                          {showNotes ? (
                            <ChevronDown className="w-3.5 h-3.5" />
                          ) : (
                            <ChevronRight className="w-3.5 h-3.5" />
                          )}
                          <span>{selectedRelease.tag_name} — Release notes</span>
                        </button>
                        {showNotes && (
                          <pre className="mt-2 text-[11px] font-mono text-[var(--label-secondary)] whitespace-pre-wrap max-h-56 overflow-y-auto">
                            {selectedRelease.release_notes}
                          </pre>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="bibulm-card py-10 px-4 text-center space-y-2 text-[12px] text-[var(--label-secondary)]">
                    <div>No matching llama.cpp releases found from GitHub.</div>
                    <button
                      type="button"
                      onClick={loadDataForTarget}
                      className="bibulm-btn-secondary inline-flex"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                      <span>Retry</span>
                    </button>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
