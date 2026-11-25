import React, { useEffect, useRef, useState } from 'react';
import { openUrl } from '@tauri-apps/plugin-opener';
import {
  Activity,
  AlertTriangle,
  ArrowUpFromLine,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  Cpu,
  Download,
  ExternalLink,
  Globe,
  HardDrive,
  Laptop,
  Link2,
  Loader2,
  LogOut,
  Play,
  Power,
  RefreshCw,
  ScrollText,
  Server,
  Shield,
  Sliders,
  SlidersHorizontal,
  Terminal,
  Trash2,
  Wifi,
  Zap,
} from 'lucide-react';
import type {
  ActiveRemoteModel,
  AppSettings,
  LinkAccessPolicy,
  LinkStatus,
  RemoteModelSummary,
  RemotePeer,
  ServerLogEntry,
} from '../types';
import { api, formatBytes, formatMiB } from '../api';
import { Toggle } from '../components/Toggle';
import { PageHeader } from '../components/PageHeader';

interface LinkViewProps {
  status: LinkStatus | null;
  peers: RemotePeer[];
  settings: AppSettings;
  activeRemoteModel: ActiveRemoteModel | null;
  gatewayPort: number;
  onSaveSettings: (next: AppSettings) => Promise<void> | void;
  onStatusChange: (next: LinkStatus) => void;
  onPeersChange: (next: RemotePeer[]) => void;
  onSelectRemoteModel: (peer: RemotePeer, model: RemoteModelSummary) => void;
  onConfigureRemoteModel: (peer: RemotePeer, model: RemoteModelSummary) => Promise<void> | void;
  onManagePeerRuntimes: (peerId: string) => void;
  onDownloadPeerModels: (peerId: string) => void;
  onOpenPeerDownloads: (peerId: string) => void;
  onNavigateBenchmark: () => void;
}

const HEADLESS_SNIPPETS = [
  { label: '1. Authenticate server once (prints login URL)', cmd: 'bibulm login --name gpu-server' },
  { label: '2. Start headless inference + Link daemon', cmd: 'bibulm serve --name gpu-server' },
  {
    label: '3. Docker Compose (with optional TS_AUTHKEY)',
    cmd: 'TS_AUTHKEY=tskey-auth-... docker compose up -d',
  },
  {
    label: '4. Install Linux systemd service',
    cmd: 'sudo bibulm service systemd --name gpu-server --install',
  },
];

export const LinkView: React.FC<LinkViewProps> = ({
  status,
  peers,
  settings,
  activeRemoteModel,
  gatewayPort,
  onSaveSettings,
  onStatusChange,
  onPeersChange,
  onSelectRemoteModel,
  onConfigureRemoteModel,
  onManagePeerRuntimes,
  onDownloadPeerModels,
  onOpenPeerDownloads,
  onNavigateBenchmark,
}) => {
  const [busy, setBusy] = useState(false);
  const [refreshingPeers, setRefreshingPeers] = useState(false);
  const [confirmLogout, setConfirmLogout] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [deployOpen, setDeployOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [loadingRemoteKey, setLoadingRemoteKey] = useState<string | null>(null);
  const [confirmDeleteModelKey, setConfirmDeleteModelKey] = useState<string | null>(null);
  const [expandedLibraryPeers, setExpandedLibraryPeers] = useState<Record<string, boolean>>({});
  const [peerLogsOpen, setPeerLogsOpen] = useState<Record<string, boolean>>({});
  const [peerLogs, setPeerLogs] = useState<Record<string, ServerLogEntry[]>>({});
  const [loadingLogsPeer, setLoadingLogsPeer] = useState<string | null>(null);
  const lastOpenedAuthUrl = useRef<string | null>(null);

  // Automatically open Tailscale login page in the user's browser when auth_url arrives.
  useEffect(() => {
    if (
      status?.state === 'needs_login' &&
      status.auth_url &&
      status.auth_url !== lastOpenedAuthUrl.current
    ) {
      lastOpenedAuthUrl.current = status.auth_url;
      openUrl(status.auth_url).catch(() => {
        // User can still click the open/copy button manually
      });
    }
    if (status?.state === 'connected' || status?.state === 'disconnected') {
      lastOpenedAuthUrl.current = null;
    }
  }, [status?.state, status?.auth_url]);

  const linkCfg = settings.link;
  const state = status?.state ?? 'disconnected';

  const updateLinkCfg = (patch: Partial<AppSettings['link']>) => {
    onSaveSettings({
      ...settings,
      link: { ...settings.link, ...patch },
    });
  };

  const copyText = (key: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey((cur) => (cur === key ? null : cur)), 1600);
  };

  const handleConnect = async () => {
    setBusy(true);
    setError(null);
    try {
      const next = await api.linkConnect();
      onStatusChange(next);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleDisconnect = async () => {
    setBusy(true);
    setError(null);
    setConfirmLogout(false);
    try {
      const next = await api.linkDisconnect();
      onStatusChange(next);
      onPeersChange([]);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleLogout = async () => {
    setBusy(true);
    setError(null);
    setConfirmLogout(false);
    try {
      const next = await api.linkLogout();
      onStatusChange(next);
      onPeersChange([]);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleRefreshPeers = async () => {
    setRefreshingPeers(true);
    setError(null);
    try {
      const list = await api.refreshLinkPeers();
      onPeersChange(list);
    } catch (e) {
      setError(String(e));
    } finally {
      setRefreshingPeers(false);
    }
  };

  const handleRemoteLoad = async (peer: RemotePeer, model: RemoteModelSummary) => {
    const key = `load:${peer.node_id}:${model.id}`;
    setLoadingRemoteKey(key);
    setError(null);
    try {
      await api.remoteLoadModel(peer.node_id, model.id);
      const updated = await api.listLinkPeers();
      onPeersChange(updated);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadingRemoteKey(null);
    }
  };

  const handleRemoteUnload = async (peer: RemotePeer, model: RemoteModelSummary) => {
    const key = `unload:${peer.node_id}:${model.id}`;
    setLoadingRemoteKey(key);
    setError(null);
    try {
      await api.remoteUnloadModel(peer.node_id, model.id);
      const updated = await api.listLinkPeers();
      onPeersChange(updated);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadingRemoteKey(null);
    }
  };

  const handleRemoteConfigure = async (peer: RemotePeer, model: RemoteModelSummary) => {
    const key = `inspect:${peer.node_id}:${model.id}`;
    setLoadingRemoteKey(key);
    setError(null);
    try {
      await onConfigureRemoteModel(peer, model);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadingRemoteKey(null);
    }
  };

  const handleRefreshPeerLogs = async (peer: RemotePeer) => {
    setLoadingLogsPeer(peer.node_id);
    setError(null);
    try {
      const entries = await api.remoteGetLogs(peer.node_id);
      setPeerLogs((prev) => ({ ...prev, [peer.node_id]: entries }));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadingLogsPeer(null);
    }
  };

  const handleTogglePeerLogs = async (peer: RemotePeer) => {
    const next = !peerLogsOpen[peer.node_id];
    setPeerLogsOpen((prev) => ({ ...prev, [peer.node_id]: next }));
    if (next) {
      await handleRefreshPeerLogs(peer);
    }
  };

  const handleRemoteDeleteModel = async (peer: RemotePeer, model: RemoteModelSummary) => {
    const key = `delete:${peer.node_id}:${model.id}`;
    setLoadingRemoteKey(key);
    setConfirmDeleteModelKey(null);
    setError(null);
    try {
      await api.remoteDeleteModel(peer.node_id, model.id);
      const updated = await api.refreshLinkPeers();
      onPeersChange(updated);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadingRemoteKey(null);
    }
  };

  const totalRemoteLoaded = peers.reduce(
    (acc, p) => acc + (p.expose_local_models ? p.loaded_models.length : 0),
    0,
  );

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <PageHeader
        icon={<Link2 className="w-4 h-4" />}
        title="BibuLM Link"
        badge={
          <span className="apple-badge font-mono uppercase tracking-wider text-[10px]">
            BYO Tailscale
          </span>
        }
        subtitle="Link your devices using your own Tailscale account — no third-party servers, no system VPN install required."
        actions={
          state === 'connected' ? (
            <>
              <button
                onClick={handleRefreshPeers}
                disabled={refreshingPeers}
                className="bibulm-btn-secondary"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${refreshingPeers ? 'animate-spin' : ''}`} />
                <span>Scan Peers</span>
              </button>
              <button
                onClick={handleDisconnect}
                disabled={busy}
                className="bibulm-btn-secondary"
              >
                <Power className="w-3.5 h-3.5" />
                <span>Disconnect</span>
              </button>
            </>
          ) : undefined
        }
      />

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-5xl mx-auto space-y-5">
      {/* Error alert */}
      {(error || status?.error) && (
        <div className="rounded-xl bg-[var(--signal-red-soft)] border border-[var(--signal-red-border)] px-4 py-3 text-[12px] text-[var(--signal-red)] flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <div className="min-w-0 flex-1 break-words">{error ?? status?.error}</div>
        </div>
      )}

      {/* Sidecar missing hint */}
      {status && !status.sidecar_available && (
        <div className="rounded-xl bg-[var(--signal-amber-soft)] border border-[var(--signal-amber-border)] px-4 py-3 text-[12px] text-[var(--label-primary)] space-y-1.5">
          <div className="font-semibold text-[var(--signal-amber)] flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" />
            <span>bibulm-linkd sidecar binary not found</span>
          </div>
          <p className="text-[var(--label-secondary)]">
            In development mode, build the embedded Tailscale userspace sidecar once using Go:
          </p>
          <code className="block px-3 py-1.5 rounded-lg bg-[var(--bg-terminal)] font-mono text-[11px] text-[#e5e5ea]">
            .\scripts\build-linkd.ps1
          </code>
        </div>
      )}

      {/* STATE 1: DISCONNECTED / ERROR — Clean 1-Click Onboarding */}
      {(state === 'disconnected' || state === 'error') && (
        <div className="bibulm-card p-6 space-y-5">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="space-y-1.5 max-w-xl">
              <h2 className="text-[15px] font-semibold text-[var(--label-primary)] tracking-tight">
                Sign in with your own Tailscale account
              </h2>
              <p className="text-[12px] text-[var(--label-secondary)] leading-relaxed">
                Click below to authenticate this machine in your browser. Once signed in on two or
                more devices, BibuLM automatically discovers your GPUs, streams remote models
                over encrypted WireGuard tunnels, and exposes them seamlessly on{' '}
                <code className="text-[var(--label-primary)] font-mono">
                  http://127.0.0.1:{gatewayPort}/v1
                </code>
                .
              </p>
            </div>

            <button
              type="button"
              onClick={handleConnect}
              disabled={busy || (status != null && !status.sidecar_available)}
              className="bibulm-btn-primary h-10 px-5 text-[13px] font-semibold shrink-0"
            >
              {busy ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Link2 className="w-4 h-4" />
              )}
              <span>Connect with Tailscale</span>
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-3 border-t border-[var(--border-hairline)]">
            <div className="apple-card-inset p-3.5 space-y-1">
              <div className="flex items-center gap-2 text-[12px] font-semibold text-[var(--signal-green)]">
                <Shield className="w-3.5 h-3.5" />
                <span>Non-Invasive Userspace</span>
              </div>
              <p className="text-[11px] text-[var(--label-secondary)]">
                Runs completely in userspace (
                <code className="text-[var(--label-primary)]">tsnet</code>). No admin rights, no
                OS routing changes, zero conflict with existing VPNs.
              </p>
            </div>
            <div className="apple-card-inset p-3.5 space-y-1">
              <div className="flex items-center gap-2 text-[12px] font-semibold text-[var(--accent-text)]">
                <Zap className="w-3.5 h-3.5" />
                <span>Zero Manual Keys</span>
              </div>
              <p className="text-[11px] text-[var(--label-secondary)]">
                Peers authenticate cryptographically via Tailscale{' '}
                <code className="text-[var(--label-primary)]">WhoIs</code>. Only your own account
                can connect by default.
              </p>
            </div>
            <div className="apple-card-inset p-3.5 space-y-1">
              <div className="flex items-center gap-2 text-[12px] font-semibold text-[var(--signal-purple)]">
                <Server className="w-3.5 h-3.5" />
                <span>Works with Any Client</span>
              </div>
              <p className="text-[11px] text-[var(--label-secondary)]">
                Cursor, Claude Code, Open WebUI, or scripts call{' '}
                <code className="text-[var(--label-primary)]">localhost:{gatewayPort}</code> and
                route directly to your home GPU.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* STATE 2: STARTING / CONNECTING / NEEDS LOGIN */}
      {(state === 'starting' || state === 'connecting' || state === 'needs_login') && (
        <div className="bibulm-card border-[var(--accent-border)] p-6 space-y-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-[var(--accent-soft)] border border-[var(--accent-border)] flex items-center justify-center">
                <Loader2 className="w-4 h-4 text-[var(--accent-text)] animate-spin" />
              </div>
              <div>
                <h2 className="text-[14px] font-semibold text-[var(--label-primary)]">
                  {state === 'needs_login'
                    ? 'Complete Tailscale sign-in in your browser'
                    : 'Connecting to your Tailnet…'}
                </h2>
                <p className="text-[12px] text-[var(--label-secondary)]">
                  {state === 'needs_login'
                    ? 'Approve this device on your Tailscale account — BibuLM will connect automatically as soon as you finish.'
                    : `Initializing userspace node ${status?.hostname ?? ''}…`}
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={handleDisconnect}
              disabled={busy}
              className="bibulm-btn-secondary"
            >
              Cancel
            </button>
          </div>

          {status?.auth_url && (
            <div className="apple-card-inset p-3.5 flex flex-wrap items-center justify-between gap-3">
              <code className="text-[12px] font-mono text-[var(--accent-text)] break-all flex-1">
                {status.auth_url}
              </code>
              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  onClick={() => openUrl(status.auth_url!)}
                  className="bibulm-btn-primary"
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                  <span>Open Login Page</span>
                </button>
                <button
                  type="button"
                  onClick={() => copyText('auth_url', status.auth_url!)}
                  className="bibulm-btn-secondary"
                >
                  {copiedKey === 'auth_url' ? (
                    <Check className="w-3.5 h-3.5 text-[var(--signal-green)]" />
                  ) : (
                    <Copy className="w-3.5 h-3.5" />
                  )}
                  <span>{copiedKey === 'auth_url' ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* STATE 3: CONNECTED — This Device + Live Peers */}
      {state === 'connected' && status && (
        <>
          {/* This Device Card */}
          <div className="bibulm-card p-5 space-y-4">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="space-y-1.5">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="apple-badge apple-badge-green text-[11px] font-medium">
                    <span className="w-1.5 h-1.5 rounded-full bg-[var(--signal-green-fill)] animate-pulse" />
                    Connected
                  </span>
                  <span className="text-[14px] font-semibold text-[var(--label-primary)]">
                    {status.hostname}
                  </span>
                  {status.tailscale_ip && (
                    <button
                      type="button"
                      onClick={() => copyText('self_ip', status.tailscale_ip!)}
                      className="apple-badge hover:bg-[var(--bg-control-hover)] text-[11px] font-mono cursor-pointer transition-colors"
                      title="Copy Tailscale IP"
                    >
                      <span>{status.tailscale_ip}</span>
                      {copiedKey === 'self_ip' ? (
                        <Check className="w-3 h-3 text-[var(--signal-green)]" />
                      ) : (
                        <Copy className="w-3 h-3 text-[var(--label-tertiary)]" />
                      )}
                    </button>
                  )}
                </div>
                <div className="flex items-center gap-3 text-[12px] text-[var(--label-secondary)] flex-wrap">
                  {status.account_login && (
                    <span>
                      Account:{' '}
                      <strong className="text-[var(--label-primary)]">{status.account_login}</strong>
                    </span>
                  )}
                  {status.tailnet_name && (
                    <span>
                      Tailnet:{' '}
                      <strong className="text-[var(--label-primary)]">{status.tailnet_name}</strong>
                    </span>
                  )}
                  <span>
                    Policy:{' '}
                    <strong className="text-[var(--label-primary)]">
                      {linkCfg.access_policy === 'self_only' ? 'My Account Only' : 'Whole Tailnet'}
                    </strong>
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {!confirmLogout ? (
                  <button
                    type="button"
                    onClick={() => setConfirmLogout(true)}
                    disabled={busy}
                    className="bibulm-btn-secondary hover:text-[var(--signal-red)] hover:border-[var(--signal-red-border)]"
                  >
                    <LogOut className="w-3.5 h-3.5" />
                    <span>Logout &amp; Unlink</span>
                  </button>
                ) : (
                  <div className="flex items-center gap-1.5 bg-[var(--signal-red-soft)] border border-[var(--signal-red-border)] rounded-lg px-2.5 py-1 text-[12px]">
                    <span className="text-[var(--signal-red)] font-medium">
                      Remove device from Tailnet?
                    </span>
                    <button
                      type="button"
                      onClick={handleLogout}
                      disabled={busy}
                      className="apple-btn-destructive h-6 px-2 text-[11px]"
                    >
                      Confirm
                    </button>
                    <button
                      type="button"
                      onClick={() => setConfirmLogout(false)}
                      className="bibulm-btn-secondary h-6 px-2 text-[11px]"
                    >
                      Cancel
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Instant Host / Routing Toggles */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-3 border-t border-[var(--border-hairline)]">
              <div className="apple-card-inset px-3.5 py-2.5 flex items-center justify-between gap-3">
                <div>
                  <div className="text-[12px] font-medium text-[var(--label-primary)]">
                    Share my models
                  </div>
                  <div className="text-[11px] text-[var(--label-secondary)]">
                    Turn off on laptops to run client-only
                  </div>
                </div>
                <Toggle
                  size="sm"
                  color="green"
                  checked={linkCfg.expose_local_models}
                  onChange={(v) => updateLinkCfg({ expose_local_models: v })}
                />
              </div>

              <div className="apple-card-inset px-3.5 py-2.5 flex items-center justify-between gap-3">
                <div>
                  <div className="text-[12px] font-medium text-[var(--label-primary)]">
                    Allow remote load
                  </div>
                  <div className="text-[11px] text-[var(--label-secondary)]">
                    Let other devices load/unload models here
                  </div>
                </div>
                <Toggle
                  size="sm"
                  color="blue"
                  checked={linkCfg.allow_remote_load}
                  onChange={(v) => updateLinkCfg({ allow_remote_load: v })}
                />
              </div>

              <div className="apple-card-inset px-3.5 py-2.5 flex items-center justify-between gap-3">
                <div>
                  <div className="text-[12px] font-medium text-[var(--label-primary)]">
                    Allow remote install &amp; downloads
                  </div>
                  <div className="text-[11px] text-[var(--label-secondary)]">
                    Let linked devices install runtimes &amp; download models here
                  </div>
                </div>
                <Toggle
                  size="sm"
                  color="blue"
                  checked={Boolean(linkCfg.allow_remote_manage)}
                  onChange={(v) => updateLinkCfg({ allow_remote_manage: v })}
                />
              </div>

              <div className="apple-card-inset px-3.5 py-2.5 flex items-center justify-between gap-3">
                <div>
                  <div className="text-[12px] font-medium text-[var(--label-primary)]">
                    Bridge into Local API
                  </div>
                  <div className="text-[11px] text-[var(--label-secondary)]">
                    Serve remote models on localhost:{gatewayPort}
                  </div>
                </div>
                <Toggle
                  size="sm"
                  color="blue"
                  checked={linkCfg.expose_remote_in_openai_api}
                  onChange={(v) => updateLinkCfg({ expose_remote_in_openai_api: v })}
                />
              </div>
            </div>
          </div>

          {/* Discovered Remote Peers */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <h2 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--label-tertiary)]">
                  Connected BibuLM Devices ({peers.length})
                </h2>
                {totalRemoteLoaded > 0 && (
                  <span className="apple-badge apple-badge-accent text-[11px]">
                    {totalRemoteLoaded} remote model{totalRemoteLoaded > 1 ? 's' : ''} ready
                  </span>
                )}
              </div>
            </div>

            {peers.length === 0 ? (
              <div className="bibulm-card border-dashed p-8 text-center space-y-2">
                <Laptop className="w-7 h-7 text-[var(--label-quaternary)] mx-auto" />
                <div className="text-[14px] font-medium text-[var(--label-primary)]">
                  Waiting for other BibuLM devices on your Tailnet
                </div>
                <p className="text-[12px] text-[var(--label-secondary)] max-w-md mx-auto">
                  Open BibuLM on another PC or laptop and sign in with{' '}
                  <strong className="text-[var(--label-primary)]">
                    {status.account_login ?? 'the same Tailscale account'}
                  </strong>
                  . It will appear here automatically within a few seconds.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {peers.map((peer) => {
                  const vramPct =
                    peer.vram_total_mib > 0
                      ? Math.min(100, (peer.vram_used_mib / peer.vram_total_mib) * 100)
                      : 0;
                  const libExpanded = Boolean(expandedLibraryPeers[peer.node_id]);

                  return (
                    <div key={peer.node_id} className="bibulm-card p-4 space-y-3.5">
                      {/* Peer Header */}
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="flex items-center gap-2.5 min-w-0">
                          <span className="w-2.5 h-2.5 rounded-full bg-[var(--signal-green-fill)] shrink-0" />
                          <div className="min-w-0">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="text-[14px] font-semibold text-[var(--label-primary)] truncate">
                                {peer.device_name}
                              </span>
                              <span className="apple-badge font-mono text-[10px]">
                                {peer.slug}
                              </span>
                              {peer.is_headless && (
                                <span className="apple-badge apple-badge-accent text-[10px]">
                                  <Server className="w-2.5 h-2.5" />
                                  Headless Server
                                </span>
                              )}
                              {peer.platform && (
                                <span className="apple-badge font-mono text-[10px]">
                                  {peer.platform}
                                </span>
                              )}
                              <span
                                className={`apple-badge text-[10px] ${
                                  peer.connection_type === 'direct'
                                    ? 'apple-badge-green'
                                    : 'apple-badge-amber'
                                }`}
                              >
                                <Wifi className="w-2.5 h-2.5" />
                                {peer.connection_type === 'direct'
                                  ? 'Direct P2P'
                                  : `DERP${peer.relay ? ` (${peer.relay})` : ''}`}
                                {peer.latency_ms != null && ` · ${peer.latency_ms} ms`}
                              </span>
                              {peer.runtimes_count > 0 ? (
                                <span className="apple-badge apple-badge-teal font-mono text-[10px]">
                                  Runtime: {peer.active_runtime_tag ?? `${peer.runtimes_count} installed`}
                                </span>
                              ) : (
                                peer.expose_local_models &&
                                (peer.allow_remote_manage || peer.platform !== '') && (
                                  <span className="apple-badge apple-badge-amber text-[10px]">
                                    No runtime installed
                                  </span>
                                )
                              )}
                              {peer.active_downloads_count > 0 && (
                                <button
                                  type="button"
                                  onClick={() => onOpenPeerDownloads(peer.node_id)}
                                  className="apple-badge apple-badge-green text-[10px] cursor-pointer hover:brightness-110"
                                >
                                  <Loader2 className="w-2.5 h-2.5 animate-spin" />
                                  Downloading ({peer.active_downloads_count})
                                </button>
                              )}
                              {!peer.expose_local_models && (
                                <span className="apple-badge text-[10px]">Client only</span>
                              )}
                            </div>
                            <div className="text-[11px] text-[var(--label-secondary)] flex items-center gap-3 mt-0.5 flex-wrap">
                              <span className="flex items-center gap-1">
                                <Cpu className="w-3 h-3 text-[var(--label-tertiary)]" />
                                {peer.gpu_summary || 'Unknown hardware'}
                              </span>
                              <span className="font-mono text-[var(--label-tertiary)]">
                                {peer.tailscale_ip}
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* Live Remote VRAM & RAM */}
                        <div className="w-56 space-y-1 shrink-0 tabular-nums">
                          <div className="flex items-center justify-between text-[11px] font-mono">
                            <span className="text-[var(--label-secondary)]">VRAM</span>
                            <span className="text-[var(--signal-green)] font-semibold">
                              {formatMiB(peer.vram_used_mib)} / {formatMiB(peer.vram_total_mib)}
                            </span>
                          </div>
                          <div className="h-1.5 rounded-full bg-[var(--bg-control)] overflow-hidden">
                            <div
                              className="h-full bg-[var(--signal-green-fill)] rounded-full transition-all"
                              style={{ width: `${vramPct}%` }}
                            />
                          </div>
                          <div className="flex items-center justify-between text-[10px] font-mono text-[var(--label-tertiary)]">
                            <span>RAM</span>
                            <span>
                              {formatMiB(peer.ram_used_mib)} / {formatMiB(peer.ram_total_mib)}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Remote Management Quick Actions & Setup Callouts */}
                      {peer.allow_remote_manage && (
                        <div className="space-y-2.5 pt-2.5 border-t border-[var(--border-hairline)]">
                          <div className="flex flex-wrap items-center gap-2">
                            <button
                              type="button"
                              onClick={() => onManagePeerRuntimes(peer.node_id)}
                              className="bibulm-btn-secondary h-7 px-2.5 text-[11px]"
                            >
                              <Activity className="w-3.5 h-3.5 text-[var(--accent-text)]" />
                              <span>Runtimes ({peer.runtimes_count})</span>
                            </button>
                            <button
                              type="button"
                              onClick={() => onDownloadPeerModels(peer.node_id)}
                              className="bibulm-btn-secondary h-7 px-2.5 text-[11px]"
                            >
                              <Download className="w-3.5 h-3.5 text-[var(--accent-text)]" />
                              <span>Download Models</span>
                            </button>
                            <button
                              type="button"
                              onClick={() => onOpenPeerDownloads(peer.node_id)}
                              className="bibulm-btn-secondary h-7 px-2.5 text-[11px]"
                            >
                              <HardDrive className="w-3.5 h-3.5 text-[var(--signal-green)]" />
                              <span>
                                Downloads
                                {peer.active_downloads_count > 0
                                  ? ` (${peer.active_downloads_count})`
                                  : ''}
                              </span>
                            </button>
                            <button
                              type="button"
                              onClick={() => handleTogglePeerLogs(peer)}
                              className={`h-7 px-2.5 rounded-lg text-[11px] font-medium flex items-center gap-1.5 border transition-colors cursor-pointer ${
                                peerLogsOpen[peer.node_id]
                                  ? 'bg-[var(--accent)] border-transparent text-white'
                                  : 'bg-[var(--bg-surface)] border-[var(--border-hairline)] hover:bg-[var(--bg-control-hover)] text-[var(--label-primary)]'
                              }`}
                            >
                              <ScrollText className="w-3.5 h-3.5" />
                              <span>Logs</span>
                            </button>
                          </div>

                          {peer.runtimes_count === 0 && (
                            <div className="rounded-xl bg-[var(--signal-amber-soft)] border border-[var(--signal-amber-border)] px-3 py-2 flex flex-wrap items-center justify-between gap-2 text-[12px] text-[var(--signal-amber)]">
                              <span>
                                Step 1: Install a <code className="font-mono">llama.cpp</code> runtime
                                on <strong>{peer.device_name}</strong> before loading models.
                              </span>
                              <button
                                type="button"
                                onClick={() => onManagePeerRuntimes(peer.node_id)}
                                className="h-7 px-2.5 rounded-lg bg-[var(--signal-amber-fill)] hover:brightness-105 text-zinc-950 font-semibold text-[11px] cursor-pointer"
                              >
                                Install Runtime
                              </button>
                            </div>
                          )}

                          {peer.loaded_models.length === 0 && peer.available_models.length === 0 && (
                            <div className="rounded-xl bg-[var(--accent-soft)] border border-[var(--accent-border)] px-3 py-2 flex flex-wrap items-center justify-between gap-2 text-[12px] text-[var(--label-primary)]">
                              <span>
                                {peer.runtimes_count === 0 ? 'Step 2: ' : ''}No GGUF models found on{' '}
                                <strong>{peer.device_name}</strong> yet.
                              </span>
                              <button
                                type="button"
                                onClick={() => onDownloadPeerModels(peer.node_id)}
                                className="bibulm-btn-primary h-7 text-[11px]"
                              >
                                Download Models
                              </button>
                            </div>
                          )}

                          {peerLogsOpen[peer.node_id] && (
                            <div className="rounded-xl bg-[var(--bg-terminal)] border border-[var(--border-hairline)] p-3 space-y-2">
                              <div className="flex items-center justify-between text-[11px]">
                                <span className="font-semibold text-[#e5e5ea]">
                                  Remote Server Logs ({peer.device_name})
                                </span>
                                <button
                                  type="button"
                                  onClick={() => handleRefreshPeerLogs(peer)}
                                  disabled={loadingLogsPeer === peer.node_id}
                                  className="px-2 py-0.5 rounded bg-white/10 hover:bg-white/15 text-[#e5e5ea] flex items-center gap-1 cursor-pointer"
                                >
                                  <RefreshCw
                                    className={`w-3 h-3 ${
                                      loadingLogsPeer === peer.node_id ? 'animate-spin' : ''
                                    }`}
                                  />
                                  <span>Refresh</span>
                                </button>
                              </div>
                              <div className="max-h-48 overflow-y-auto font-mono text-[10px] space-y-0.5 text-[#a1a1a6] tabular-nums">
                                {(peerLogs[peer.node_id] ?? []).length === 0 ? (
                                  <div className="text-[#636366] italic">No log entries yet.</div>
                                ) : (
                                  (peerLogs[peer.node_id] ?? []).slice(-100).map((entry, i) => (
                                    <div key={i} className="break-all">
                                      <span className="text-[#636366]">{entry.timestamp}</span>{' '}
                                      <span
                                        className={
                                          entry.level === 'ERROR'
                                            ? 'text-[#ff453a]'
                                            : entry.level === 'WARN'
                                              ? 'text-[#ff9f0a]'
                                              : 'text-[#30d158]'
                                        }
                                      >
                                        [{entry.level}]
                                      </span>{' '}
                                      <span className="text-[#64d2ff]">[{entry.source}]</span>{' '}
                                      <span className="text-[#e5e5ea]">{entry.message}</span>
                                    </div>
                                  ))
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Loaded Models on Peer */}
                      {peer.expose_local_models && (
                        <div className="space-y-2 pt-2.5 border-t border-[var(--border-hairline)]">
                          <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--label-tertiary)]">
                            Loaded Models ({peer.loaded_models.length})
                          </div>
                          {peer.loaded_models.length === 0 ? (
                            <div className="text-[12px] text-[var(--label-tertiary)] italic">
                              No model currently loaded on this device.
                            </div>
                          ) : (
                            <div className="space-y-1.5">
                              {peer.loaded_models.map((m) => {
                                const qualifiedId = `${peer.slug}/${m.id}`;
                                const isActive = activeRemoteModel?.qualifiedId === qualifiedId;
                                const unloading =
                                  loadingRemoteKey === `unload:${peer.node_id}:${m.id}`;

                                return (
                                  <div
                                    key={m.id}
                                    className={`flex flex-wrap items-center justify-between gap-2 px-3 py-2 rounded-xl border ${
                                      isActive
                                        ? 'bg-[var(--accent-soft)] border-[var(--accent-border)]'
                                        : 'bg-[var(--bg-surface-inset)] border-[var(--border-hairline)]'
                                    }`}
                                  >
                                    <div className="min-w-0 flex items-center gap-2 flex-wrap">
                                      <span className="w-2 h-2 rounded-full bg-[var(--signal-green-fill)] shrink-0" />
                                      <span className="text-[12px] font-medium text-[var(--label-primary)] truncate">
                                        {m.display_name}
                                      </span>
                                      {m.quantization && (
                                        <span className="apple-badge font-mono text-[10px]">
                                          {m.quantization}
                                        </span>
                                      )}
                                      <code className="text-[10px] font-mono text-[var(--label-tertiary)]">
                                        {qualifiedId}
                                      </code>
                                    </div>
                                    <div className="flex items-center gap-1.5 shrink-0">
                                      <button
                                        type="button"
                                        onClick={() => copyText(qualifiedId, qualifiedId)}
                                        className="bibulm-btn-secondary h-7 px-2 text-[11px]"
                                        title="Copy OpenAI API model ID"
                                      >
                                        {copiedKey === qualifiedId ? (
                                          <Check className="w-3 h-3 text-[var(--signal-green)]" />
                                        ) : (
                                          <Copy className="w-3 h-3" />
                                        )}
                                        <span>API ID</span>
                                      </button>
                                      <button
                                        type="button"
                                        onClick={() => {
                                          onSelectRemoteModel(peer, m);
                                          onNavigateBenchmark();
                                        }}
                                        className="bibulm-btn-primary h-7 px-2.5 text-[11px]"
                                      >
                                        <Play className="w-3 h-3" />
                                        <span>{isActive ? 'Selected' : 'Use'}</span>
                                      </button>
                                      {peer.allow_remote_load && (
                                        <>
                                          <button
                                            type="button"
                                            onClick={() => handleRemoteConfigure(peer, m)}
                                            disabled={loadingRemoteKey !== null}
                                            className="bibulm-btn-secondary h-7 px-2 text-[11px]"
                                            title="Configure parameters & reload on remote machine"
                                          >
                                            {loadingRemoteKey === `inspect:${peer.node_id}:${m.id}` ? (
                                              <Loader2 className="w-3 h-3 animate-spin" />
                                            ) : (
                                              <Sliders className="w-3 h-3" />
                                            )}
                                            <span>Config</span>
                                          </button>
                                          <button
                                            type="button"
                                            onClick={() => handleRemoteUnload(peer, m)}
                                            disabled={unloading}
                                            className="bibulm-btn-secondary h-7 px-2 text-[11px] hover:text-[var(--signal-red)] hover:border-[var(--signal-red-border)]"
                                            title="Unload model on remote machine"
                                          >
                                            {unloading ? (
                                              <Loader2 className="w-3 h-3 animate-spin" />
                                            ) : (
                                              <ArrowUpFromLine className="w-3 h-3" />
                                            )}
                                            <span>Eject</span>
                                          </button>
                                        </>
                                      )}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          )}

                          {/* Remote Library (Available for remote load / JIT) */}
                          {peer.available_models.length > 0 && (
                            <div className="pt-1">
                              <button
                                type="button"
                                onClick={() =>
                                  setExpandedLibraryPeers((prev) => ({
                                    ...prev,
                                    [peer.node_id]: !prev[peer.node_id],
                                  }))
                                }
                                className="flex items-center gap-1.5 text-[11px] font-medium text-[var(--label-secondary)] hover:text-[var(--label-primary)] cursor-pointer"
                              >
                                {libExpanded ? (
                                  <ChevronDown className="w-3.5 h-3.5" />
                                ) : (
                                  <ChevronRight className="w-3.5 h-3.5" />
                                )}
                                <HardDrive className="w-3 h-3" />
                                <span>
                                  Remote Library ({peer.available_models.length} model
                                  {peer.available_models.length > 1 ? 's' : ''})
                                </span>
                              </button>

                              {libExpanded && (
                                <div className="mt-2 max-h-56 overflow-y-auto space-y-1.5 pr-1">
                                  {peer.available_models.map((m) => {
                                    const loadKey = `load:${peer.node_id}:${m.id}`;
                                    const isLoadingThis = loadingRemoteKey === loadKey;
                                    return (
                                      <div
                                        key={m.id}
                                        className="apple-card-inset flex items-center justify-between gap-2 px-3 py-1.5"
                                      >
                                        <div className="min-w-0 flex items-center gap-2">
                                          <span className="text-[12px] text-[var(--label-primary)] truncate">
                                            {m.display_name}
                                          </span>
                                          {m.quantization && (
                                            <span className="apple-badge font-mono text-[10px]">
                                              {m.quantization}
                                            </span>
                                          )}
                                          {m.size_bytes > 0 && (
                                            <span className="text-[10px] text-[var(--label-tertiary)] tabular-nums">
                                              {formatBytes(m.size_bytes)}
                                            </span>
                                          )}
                                        </div>
                                        {peer.allow_remote_load ? (
                                          <div className="flex items-center gap-1.5 shrink-0">
                                            <button
                                              type="button"
                                              onClick={() => handleRemoteConfigure(peer, m)}
                                              disabled={loadingRemoteKey !== null}
                                              className="bibulm-btn-primary h-6 px-2.5 text-[11px]"
                                              title="Configure load parameters & load on remote machine"
                                            >
                                              {loadingRemoteKey === `inspect:${peer.node_id}:${m.id}` ? (
                                                <Loader2 className="w-3 h-3 animate-spin" />
                                              ) : (
                                                <Sliders className="w-3 h-3" />
                                              )}
                                              <span>Configure</span>
                                            </button>
                                            <button
                                              type="button"
                                              onClick={() => handleRemoteLoad(peer, m)}
                                              disabled={loadingRemoteKey !== null}
                                              className="bibulm-btn-secondary h-6 px-2 text-[11px]"
                                              title="Quick load using saved preset or auto-fit"
                                            >
                                              {isLoadingThis ? (
                                                <Loader2 className="w-3 h-3 animate-spin" />
                                              ) : (
                                                <Play className="w-3 h-3" />
                                              )}
                                              <span>{isLoadingThis ? 'Loading…' : 'Load'}</span>
                                            </button>
                                            {peer.allow_remote_manage && (
                                              confirmDeleteModelKey === `del:${peer.node_id}:${m.id}` ? (
                                                <div className="flex items-center gap-1">
                                                  <button
                                                    type="button"
                                                    onClick={() => handleRemoteDeleteModel(peer, m)}
                                                    disabled={loadingRemoteKey !== null}
                                                    className="apple-btn-destructive h-6 px-2 text-[10px]"
                                                  >
                                                    Delete
                                                  </button>
                                                  <button
                                                    type="button"
                                                    onClick={() => setConfirmDeleteModelKey(null)}
                                                    className="bibulm-btn-secondary h-6 px-1.5 text-[10px]"
                                                  >
                                                    Cancel
                                                  </button>
                                                </div>
                                              ) : (
                                                <button
                                                  type="button"
                                                  onClick={() =>
                                                    setConfirmDeleteModelKey(`del:${peer.node_id}:${m.id}`)
                                                  }
                                                  disabled={loadingRemoteKey !== null}
                                                  className="apple-btn-ghost w-6 h-6 hover:text-[var(--signal-red)]"
                                                  title="Delete model files from remote server"
                                                  aria-label="Delete remote model"
                                                >
                                                  <Trash2 className="w-3 h-3" />
                                                </button>
                                              )
                                            )}
                                          </div>
                                        ) : (
                                          <span className="text-[10px] text-[var(--label-tertiary)]">
                                            Remote load off
                                          </span>
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </>
      )}

      {/* Deploy Headless Server Accordion */}
      <div className="bibulm-card overflow-hidden">
        <button
          type="button"
          onClick={() => setDeployOpen((v) => !v)}
          className="w-full px-4 py-3 flex items-center justify-between text-[12px] font-medium text-[var(--label-primary)] hover:bg-[var(--bg-control)] transition-colors cursor-pointer"
        >
          <span className="flex items-center gap-2">
            <Terminal className="w-3.5 h-3.5 text-[var(--accent-text)]" />
            <span>Deploy a Headless Server (CLI / Docker / systemd)</span>
          </span>
          {deployOpen ? (
            <ChevronDown className="w-4 h-4 text-[var(--label-tertiary)]" />
          ) : (
            <ChevronRight className="w-4 h-4 text-[var(--label-tertiary)]" />
          )}
        </button>

        {deployOpen && (
          <div className="px-4 pb-4 pt-3 border-t border-[var(--border-hairline)] space-y-3 text-[12px]">
            <p className="text-[var(--label-secondary)] text-[11px]">
              Install the <code className="text-[var(--label-primary)] font-mono">bibulm</code> CLI
              (with <code className="text-[var(--label-primary)] font-mono">bibulm-linkd</code>{' '}
              alongside it or in{' '}
              <code className="text-[var(--label-primary)] font-mono">PATH</code>) on a GPU server,
              or launch via Docker. Once signed in with the same Tailscale account, the server
              appears above so you can install runtimes, download models, and configure inference
              remotely.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
              {HEADLESS_SNIPPETS.map((item, idx) => (
                <div key={idx} className="apple-card-inset p-3 space-y-1.5">
                  <div className="text-[11px] text-[var(--label-secondary)]">{item.label}</div>
                  <div className="flex items-center justify-between gap-2">
                    <code className="text-[11px] font-mono text-[var(--accent-text)] break-all">
                      {item.cmd}
                    </code>
                    <button
                      type="button"
                      onClick={() => copyText(`cli-${idx}`, item.cmd)}
                      className="apple-btn-ghost w-7 h-7 shrink-0"
                      title="Copy command"
                      aria-label="Copy command"
                    >
                      {copiedKey === `cli-${idx}` ? (
                        <Check className="w-3 h-3 text-[var(--signal-green)]" />
                      ) : (
                        <Copy className="w-3 h-3" />
                      )}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Advanced Settings Accordion */}
      <div className="bibulm-card overflow-hidden">
        <button
          type="button"
          onClick={() => setAdvancedOpen((v) => !v)}
          className="w-full px-4 py-3 flex items-center justify-between text-[12px] font-medium text-[var(--label-primary)] hover:bg-[var(--bg-control)] transition-colors cursor-pointer"
        >
          <span className="flex items-center gap-2">
            <SlidersHorizontal className="w-3.5 h-3.5 text-[var(--label-tertiary)]" />
            <span>Advanced Link Settings (Device Name, Policy, Headscale)</span>
          </span>
          {advancedOpen ? (
            <ChevronDown className="w-4 h-4 text-[var(--label-tertiary)]" />
          ) : (
            <ChevronRight className="w-4 h-4 text-[var(--label-tertiary)]" />
          )}
        </button>

        {advancedOpen && (
          <div className="px-4 pb-4 pt-3 border-t border-[var(--border-hairline)] space-y-3.5 text-[12px]">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-1">
                <label className="text-[var(--label-primary)] font-medium">Custom Device Name</label>
                <input
                  type="text"
                  value={linkCfg.device_name ?? ''}
                  onChange={(e) =>
                    updateLinkCfg({ device_name: e.target.value.trim() ? e.target.value : null })
                  }
                  placeholder={status?.hostname ?? 'Auto (OS hostname)'}
                  className="w-full bibulm-input"
                />
                <p className="text-[11px] text-[var(--label-tertiary)]">
                  Registered in your Tailnet as{' '}
                  <code className="text-[var(--label-secondary)]">bibulm-&lt;name&gt;</code>{' '}
                  (reconnect to apply).
                </p>
              </div>

              <div className="space-y-1">
                <label className="text-[var(--label-primary)] font-medium">Access Policy</label>
                <select
                  value={linkCfg.access_policy}
                  onChange={(e) =>
                    updateLinkCfg({ access_policy: e.target.value as LinkAccessPolicy })
                  }
                  className="w-full apple-select"
                >
                  <option value="self_only">My Tailscale Account Only (Recommended)</option>
                  <option value="same_tailnet">Anyone in My Tailnet (Tailscale ACLs)</option>
                </select>
                <p className="text-[11px] text-[var(--label-tertiary)]">
                  Verified cryptographically via Tailscale WhoIs on every connection.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-1">
                <label className="text-[var(--label-primary)] font-medium flex items-center gap-1.5">
                  <Globe className="w-3.5 h-3.5 text-[var(--label-tertiary)]" />
                  <span>Custom Control URL (Headscale, optional)</span>
                </label>
                <input
                  type="text"
                  value={linkCfg.control_url ?? ''}
                  onChange={(e) =>
                    updateLinkCfg({ control_url: e.target.value.trim() ? e.target.value : null })
                  }
                  placeholder="https://controlplane.tailscale.com (default)"
                  className="w-full bibulm-input font-mono text-[11px]"
                />
              </div>

              <div className="apple-card-inset flex items-center justify-between px-3 py-2">
                <div>
                  <div className="font-medium text-[var(--label-primary)]">
                    Auto-connect on launch
                  </div>
                  <div className="text-[11px] text-[var(--label-tertiary)]">
                    Reconnect automatically when BibuLM starts
                  </div>
                </div>
                <Toggle
                  size="sm"
                  checked={linkCfg.auto_connect}
                  onChange={(v) => updateLinkCfg({ auto_connect: v })}
                />
              </div>
            </div>
          </div>
        )}
      </div>
        </div>
      </div>
    </div>
  );
};
