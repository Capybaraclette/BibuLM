import React, { useEffect, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  FolderOpen,
  HardDriveDownload,
  Laptop,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  Server,
  Trash2,
  X,
} from 'lucide-react';
import type { DownloadJobState, RemotePeer } from '../types';
import { api, formatBytes, formatEta } from '../api';

interface DownloadsPopoverProps {
  open: boolean;
  onClose: () => void;
  downloads: DownloadJobState[];
  modelsDir: string;
  linkPeers: RemotePeer[];
  remoteDownloads: Record<string, DownloadJobState[]>;
  selectedPeerId: string | null;
  onSelectPeerId: (id: string | null) => void;
  onRefresh: () => void;
  onRefreshRemote: (peerId: string) => void;
  onRefreshPeers: () => void;
  onModelReadyScan: () => void;
}

function isActiveStatus(status: string): boolean {
  return status === 'downloading' || status === 'queued' || status === 'verifying';
}

export const DownloadsPopover: React.FC<DownloadsPopoverProps> = ({
  open,
  onClose,
  downloads,
  modelsDir,
  linkPeers,
  remoteDownloads,
  selectedPeerId,
  onSelectPeerId,
  onRefresh,
  onRefreshRemote,
  onRefreshPeers,
  onModelReadyScan,
}) => {
  const [filter, setFilter] = useState('');
  const [tab, setTab] = useState<'all' | 'active' | 'completed'>('all');

  const manageablePeers = linkPeers.filter((p) => p.online && p.allow_remote_manage);
  const activePeer = selectedPeerId
    ? manageablePeers.find((p) => p.node_id === selectedPeerId) ?? null
    : null;

  useEffect(() => {
    if (selectedPeerId && !activePeer) {
      onSelectPeerId(null);
    }
  }, [selectedPeerId, activePeer, onSelectPeerId]);

  useEffect(() => {
    if (open && activePeer) {
      onRefreshRemote(activePeer.node_id);
    }
  }, [open, activePeer?.node_id, onRefreshRemote]);

  if (!open) return null;

  const currentJobs = activePeer ? (remoteDownloads[activePeer.node_id] ?? []) : downloads;

  const filtered = currentJobs.filter((d) => {
    const q = filter.trim().toLowerCase();
    if (q && !d.title.toLowerCase().includes(q) && !(d.repo_id ?? '').toLowerCase().includes(q)) {
      return false;
    }
    if (tab === 'active') {
      return ['queued', 'downloading', 'verifying', 'paused'].includes(d.status);
    }
    if (tab === 'completed') {
      return ['completed', 'failed', 'cancelled'].includes(d.status);
    }
    return true;
  });

  const handleClear = async () => {
    if (activePeer) {
      await api.remoteClearCompletedDownloads(activePeer.node_id);
      onRefreshRemote(activePeer.node_id);
    } else {
      await api.clearCompletedDownloads();
      onRefresh();
    }
  };

  const localActiveCount = downloads.filter((d) => isActiveStatus(d.status)).length;

  return (
    <div
      className="fixed top-[58px] right-3 w-[450px] max-w-[94vw] max-h-[580px] rounded-2xl apple-glass border flex flex-col overflow-hidden z-50"
      style={{ boxShadow: 'var(--shadow-popover)' }}
    >
      <div className="px-4 py-3 border-b border-[var(--border-hairline)] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[14px] font-semibold text-[var(--label-primary)] tracking-tight">
            Downloads
          </span>
          {activePeer && (
            <span className="apple-badge apple-badge-accent font-mono text-[10px]">
              {activePeer.device_name}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {activePeer && (
            <button
              type="button"
              onClick={() => onRefreshRemote(activePeer.node_id)}
              className="apple-btn-ghost w-7 h-7"
              title="Refresh remote downloads"
              aria-label="Refresh remote downloads"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          )}
          <button
            type="button"
            onClick={onClose}
            className="apple-btn-ghost w-7 h-7"
            aria-label="Close downloads"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Target machine selector when manageable remote peers are online */}
      {manageablePeers.length > 0 && (
        <div className="px-3 py-2 border-b border-[var(--border-hairline)] bg-[var(--bg-surface-inset)] flex items-center gap-1.5 overflow-x-auto">
          <button
            type="button"
            onClick={() => onSelectPeerId(null)}
            className={`h-7 px-2.5 rounded-lg text-[11px] font-medium flex items-center gap-1.5 shrink-0 transition-colors cursor-pointer ${
              !activePeer
                ? 'bg-[var(--accent)] text-white shadow-sm'
                : 'bg-[var(--bg-control)] text-[var(--label-secondary)] hover:text-[var(--label-primary)]'
            }`}
          >
            <Laptop className="w-3 h-3" />
            <span>This Machine</span>
            {localActiveCount > 0 && (
              <span className="px-1 rounded bg-black/20 text-[10px] font-mono tabular-nums">
                {localActiveCount}
              </span>
            )}
          </button>
          {manageablePeers.map((peer) => {
            const selected = activePeer?.node_id === peer.node_id;
            const tracked = (remoteDownloads[peer.node_id] ?? []).filter((j) =>
              isActiveStatus(j.status),
            ).length;
            const badgeCount = Math.max(peer.active_downloads_count, tracked);
            return (
              <button
                key={peer.node_id}
                type="button"
                onClick={() => onSelectPeerId(peer.node_id)}
                className={`h-7 px-2.5 rounded-lg text-[11px] font-medium flex items-center gap-1.5 shrink-0 transition-colors cursor-pointer ${
                  selected
                    ? 'bg-[var(--accent)] text-white shadow-sm'
                    : 'bg-[var(--bg-control)] text-[var(--label-secondary)] hover:text-[var(--label-primary)]'
                }`}
              >
                {peer.is_headless ? (
                  <Server className="w-3 h-3" />
                ) : (
                  <Laptop className="w-3 h-3" />
                )}
                <span className="truncate max-w-[120px]">{peer.device_name}</span>
                {badgeCount > 0 && (
                  <span className="px-1 rounded bg-white/20 text-[10px] font-mono tabular-nums">
                    {badgeCount}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}

      <div className="p-3 border-b border-[var(--border-hairline)] space-y-2.5 bg-[var(--bg-surface)]/75">
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter downloads…"
          className="w-full bibulm-input"
        />
        <div className="flex items-center justify-between text-[11px]">
          <div className="apple-segmented">
            {(['all', 'active', 'completed'] as const).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setTab(t)}
                className={`apple-segmented-item capitalize py-0.5 px-2.5 text-[11px] ${
                  tab === t ? 'apple-segmented-item-active' : ''
                }`}
              >
                {t}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={handleClear}
            className="text-[var(--label-secondary)] hover:text-[var(--label-primary)] flex items-center gap-1 cursor-pointer"
          >
            <Trash2 className="w-3 h-3" />
            <span>Clear History</span>
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto divide-y divide-[var(--border-hairline)] bg-[var(--bg-surface)]/85">
        {filtered.length === 0 ? (
          <div className="py-12 px-4 text-center text-[12px] text-[var(--label-tertiary)]">
            {activePeer
              ? `No downloads on ${activePeer.device_name} yet. Select it as target in Runtime Manager or Model Search.`
              : 'No downloads yet. Search HuggingFace GGUFs or install a llama.cpp runtime release.'}
          </div>
        ) : (
          filtered.map((job) => {
            const pct =
              job.total_bytes > 0
                ? Math.min(100, (job.downloaded_bytes / job.total_bytes) * 100)
                : job.status === 'completed'
                  ? 100
                  : 0;
            const isActive = isActiveStatus(job.status);

            return (
              <div
                key={job.id}
                className="px-4 py-3 hover:bg-[var(--bg-control)] transition-colors"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-start gap-2.5 min-w-0">
                    <div className="mt-0.5 shrink-0">
                      {job.kind === 'runtime' ? (
                        <Cpu className="w-4 h-4 text-[var(--accent-text)]" />
                      ) : (
                        <HardDriveDownload className="w-4 h-4 text-[var(--accent-text)]" />
                      )}
                    </div>
                    <div className="min-w-0">
                      <div className="text-[12px] font-medium text-[var(--label-primary)] truncate">
                        {job.title}
                      </div>
                      <div className="text-[11px] text-[var(--label-secondary)] flex items-center gap-1.5 mt-0.5 tabular-nums">
                        <span className="capitalize">{job.kind}</span>
                        <span>•</span>
                        <span>{formatBytes(job.total_bytes)}</span>
                        {job.status === 'completed' && (
                          <CheckCircle2 className="w-3.5 h-3.5 text-[var(--signal-green)]" />
                        )}
                        {job.status === 'failed' && (
                          <AlertCircle className="w-3.5 h-3.5 text-[var(--signal-red)]" />
                        )}
                      </div>
                      {activePeer && (
                        <div className="text-[10px] font-mono text-[var(--label-tertiary)] truncate mt-0.5">
                          {job.dest_dir}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-1 shrink-0">
                    {isActive && job.kind === 'model' && (
                      <button
                        type="button"
                        onClick={async () => {
                          if (activePeer) {
                            await api.remotePauseDownload(activePeer.node_id, job.id);
                            onRefreshRemote(activePeer.node_id);
                          } else {
                            await api.pauseDownload(job.id);
                            onRefresh();
                          }
                        }}
                        className="apple-btn-ghost w-7 h-7"
                        title="Pause"
                        aria-label="Pause download"
                      >
                        <Pause className="w-3.5 h-3.5" />
                      </button>
                    )}
                    {(job.status === 'paused' || job.status === 'failed') &&
                      job.kind === 'model' && (
                        <button
                          type="button"
                          onClick={async () => {
                            if (activePeer) {
                              await api.remoteResumeDownload(activePeer.node_id, job.id);
                              onRefreshRemote(activePeer.node_id);
                            } else {
                              await api.resumeDownload(job.id);
                              onRefresh();
                            }
                          }}
                          className="apple-btn-ghost w-7 h-7 text-[var(--accent-text)]"
                          title="Resume / Retry"
                          aria-label="Resume download"
                        >
                          <RotateCcw className="w-3.5 h-3.5" />
                        </button>
                      )}
                    {isActive && job.kind === 'model' && (
                      <button
                        type="button"
                        onClick={async () => {
                          if (activePeer) {
                            await api.remoteCancelDownload(activePeer.node_id, job.id);
                            onRefreshRemote(activePeer.node_id);
                          } else {
                            await api.cancelDownload(job.id);
                            onRefresh();
                          }
                        }}
                        className="apple-btn-ghost w-7 h-7 text-[var(--signal-red)]"
                        title="Cancel"
                        aria-label="Cancel download"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    )}
                    {job.status === 'completed' && job.kind === 'model' && (
                      <button
                        type="button"
                        onClick={() => {
                          if (activePeer) {
                            onRefreshPeers();
                          } else {
                            onModelReadyScan();
                          }
                          onClose();
                        }}
                        className="apple-btn-ghost w-7 h-7 text-[var(--signal-green)]"
                        title={activePeer ? 'Refresh remote library' : 'Refresh Library'}
                        aria-label="Open in Library"
                      >
                        <Play className="w-3.5 h-3.5" />
                      </button>
                    )}
                    {!activePeer && (
                      <button
                        type="button"
                        onClick={() => api.revealInExplorer(job.dest_dir)}
                        className="apple-btn-ghost w-7 h-7"
                        title="Open folder in Explorer"
                        aria-label="Open folder"
                      >
                        <FolderOpen className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>

                {(isActive || job.status === 'paused') && (
                  <div className="mt-2 space-y-1">
                    <div className="h-1.5 rounded-full bg-[var(--bg-control)] overflow-hidden">
                      <div
                        className={`h-full transition-all duration-200 rounded-full ${
                          job.status === 'paused'
                            ? 'bg-[var(--signal-amber-fill)]'
                            : 'bg-[var(--accent)]'
                        }`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-[var(--label-tertiary)] font-mono tabular-nums">
                      <span className="truncate max-w-[210px]">
                        {job.current_file ?? job.status}
                      </span>
                      <span>
                        {pct.toFixed(1)}%
                        {job.speed_bytes_per_sec > 0 &&
                          ` · ${formatBytes(job.speed_bytes_per_sec)}/s · ETA ${formatEta(job.eta_seconds)}`}
                      </span>
                    </div>
                  </div>
                )}

                {job.error && (
                  <div className="mt-1.5 text-[11px] text-[var(--signal-red)] bg-[var(--signal-red-soft)] border border-[var(--signal-red-border)] rounded-lg px-2.5 py-1.5 break-words">
                    {job.error}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      <div className="px-4 py-2.5 border-t border-[var(--border-hairline)] flex items-center justify-end bg-[var(--bg-surface-inset)]">
        {activePeer ? (
          <span className="text-[11px] text-[var(--label-tertiary)] font-mono truncate">
            Remote target: {activePeer.device_name} ({activePeer.tailscale_ip})
          </span>
        ) : (
          <button
            type="button"
            onClick={() => api.revealInExplorer(modelsDir)}
            className="text-[12px] text-[var(--label-secondary)] hover:text-[var(--label-primary)] flex items-center gap-1.5 cursor-pointer"
          >
            <span>Open Models Directory</span>
            <FolderOpen className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </div>
  );
};
