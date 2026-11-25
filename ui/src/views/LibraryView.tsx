import React, { useState } from 'react';
import { open } from '@tauri-apps/plugin-dialog';
import {
  Brain,
  Check,
  Copy,
  Eye,
  FolderOpen,
  FolderPlus,
  Layers,
  Library,
  RefreshCw,
  Search,
  Sliders,
  Trash2,
  Wrench,
  Zap,
} from 'lucide-react';
import type { AppSettings, LoadedModelInfo, LocalModelEntry } from '../types';
import { api, formatBytes, formatParams } from '../api';
import { PageHeader } from '../components/PageHeader';

interface LibraryViewProps {
  models: LocalModelEntry[];
  loadedModels: LoadedModelInfo[];
  settings: AppSettings;
  scanning: boolean;
  onRescan: () => void;
  onSaveSettings: (next: AppSettings) => void;
  onConfigureLoad: (model: LocalModelEntry) => void;
}

export const LibraryView: React.FC<LibraryViewProps> = ({
  models,
  loadedModels,
  settings,
  scanning,
  onRescan,
  onSaveSettings,
  onConfigureLoad,
}) => {
  const [query, setQuery] = useState('');
  const [capFilter, setCapFilter] = useState<'all' | 'moe' | 'reasoning' | 'vision' | 'embedding'>(
    'all',
  );
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<LocalModelEntry | null>(null);

  const handleAddFolder = async () => {
    const selected = await open({ directory: true, multiple: false, title: 'Add GGUF Models Folder' });
    if (typeof selected === 'string' && !settings.models_dirs.includes(selected)) {
      await onSaveSettings({
        ...settings,
        models_dirs: [...settings.models_dirs, selected],
      });
      onRescan();
    }
  };

  const handleDelete = async (m: LocalModelEntry) => {
    await api.deleteLocalModelFiles(m.path);
    setConfirmDelete(null);
    onRescan();
  };

  const filtered = models.filter((m) => {
    if (capFilter === 'moe' && !m.is_moe) return false;
    if (capFilter === 'reasoning' && !m.supports_reasoning) return false;
    if (capFilter === 'vision' && !m.supports_vision) return false;
    if (capFilter === 'embedding' && !m.is_embedding) return false;
    const q = query.trim().toLowerCase();
    if (!q) return true;
    return (
      m.display_name.toLowerCase().includes(q) ||
      m.id.toLowerCase().includes(q) ||
      (m.architecture ?? '').toLowerCase().includes(q) ||
      (m.quantization ?? '').toLowerCase().includes(q)
    );
  });

  const totalSize = models.reduce((acc, m) => acc + m.size_bytes, 0);

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Unified Header Toolbar */}
      <PageHeader
        icon={<Library className="w-4 h-4" />}
        title={`Local GGUF Library (${models.length})`}
        subtitle={
          <>
            Native Rust GGUF header inspection · {formatBytes(totalSize)} across{' '}
            {settings.models_dirs.length} director{settings.models_dirs.length === 1 ? 'y' : 'ies'}
          </>
        }
        actions={
          <>
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-[var(--label-tertiary)] absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Filter local models…"
                className="bibulm-input w-56 pl-8"
              />
            </div>
            <button onClick={handleAddFolder} className="bibulm-btn-secondary">
              <FolderPlus className="w-3.5 h-3.5" />
              <span>Add Folder</span>
            </button>
            <button onClick={onRescan} className="bibulm-btn-secondary">
              <RefreshCw className={`w-3.5 h-3.5 ${scanning ? 'animate-spin' : ''}`} />
              <span>Rescan</span>
            </button>
          </>
        }
      />

      {/* Capability Filter Bar */}
      <div className="px-6 py-2.5 border-b border-[var(--border-hairline)] bg-[var(--bg-canvas)] flex items-center shrink-0">
        <div className="apple-segmented">
          {(
            [
              { id: 'all', label: 'All Models' },
              { id: 'moe', label: 'MoE' },
              { id: 'reasoning', label: 'Reasoning' },
              { id: 'vision', label: 'Vision (mmproj)' },
              { id: 'embedding', label: 'Embedding' },
            ] as const
          ).map((f) => (
            <button
              key={f.id}
              type="button"
              onClick={() => setCapFilter(f.id)}
              className={`apple-segmented-item ${
                capFilter === f.id ? 'apple-segmented-item-active' : ''
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {/* Model List */}
      <div className="flex-1 overflow-y-auto p-6 space-y-2.5">
        {filtered.length === 0 ? (
          <div className="bibulm-card border-dashed py-16 px-4 text-center space-y-2">
            <div className="text-[14px] font-medium text-[var(--label-primary)]">
              No GGUF models found
            </div>
            <p className="text-[12px] text-[var(--label-secondary)] max-w-md mx-auto">
              Download models from the <strong>Model Search</strong> tab, or click{' '}
              <strong>Add Folder</strong> to link any local directory containing GGUF model files.
            </p>
          </div>
        ) : (
          filtered.map((m) => {
            const isLoaded = loadedModels.some((lm) => lm.model_id === m.id);
            return (
              <div
                key={m.id}
                className={`bibulm-card p-4 flex flex-wrap items-center justify-between gap-4 transition-colors ${
                  isLoaded
                    ? 'border-[var(--signal-green-border)] bg-[var(--signal-green-soft)]'
                    : 'hover:border-[var(--border-strong)]'
                }`}
              >
                <div className="min-w-0 flex-1 space-y-1.5">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[14px] font-semibold text-[var(--label-primary)]">
                      {m.display_name}
                    </span>
                    {m.source && (
                      <span className="apple-badge apple-badge-accent text-[10px]">
                        {m.source}
                      </span>
                    )}
                    {m.quantization && (
                      <span className="apple-badge font-mono text-[10px]">
                        {m.quantization}
                      </span>
                    )}
                    {isLoaded && (
                      <span className="apple-badge apple-badge-green text-[10px] font-semibold">
                        LOADED
                      </span>
                    )}
                    {m.is_moe && (
                      <span className="apple-badge apple-badge-purple text-[10px]">
                        <Layers className="w-3 h-3" />
                        MoE {m.expert_count ? `${m.expert_count}E` : ''}
                        {m.expert_used_count ? `/${m.expert_used_count}A` : ''}
                      </span>
                    )}
                    {m.supports_reasoning && (
                      <span className="apple-badge apple-badge-green text-[10px]">
                        <Brain className="w-3 h-3" /> Reasoning
                      </span>
                    )}
                    {m.supports_vision && (
                      <span className="apple-badge apple-badge-amber text-[10px]">
                        <Eye className="w-3 h-3" /> Vision
                      </span>
                    )}
                    {m.has_mtp && (
                      <span className="apple-badge apple-badge-purple text-[10px]">
                        <Zap className="w-3 h-3" /> MTP
                      </span>
                    )}
                    {m.supports_tools && (
                      <span className="apple-badge apple-badge-teal text-[10px]">
                        <Wrench className="w-3 h-3" /> Tools
                      </span>
                    )}
                  </div>

                  <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-[var(--label-secondary)] font-mono tabular-nums">
                    <span>{formatBytes(m.size_bytes)}</span>
                    {m.shard_count > 1 && <span>{m.shard_count} shards</span>}
                    {m.architecture && <span>arch: {m.architecture}</span>}
                    {m.parameter_count && <span>params: {formatParams(m.parameter_count)}</span>}
                    {m.layer_count && <span>{m.layer_count} layers</span>}
                    {m.head_count && (
                      <span>
                        heads: {m.head_count}
                        {m.head_count_kv ? `/${m.head_count_kv} KV` : ''}
                      </span>
                    )}
                    {m.max_context_length && (
                      <span>max ctx: {m.max_context_length.toLocaleString()}</span>
                    )}
                  </div>

                  <div className="text-[11px] text-[var(--label-tertiary)] truncate font-mono">
                    {m.id}
                  </div>
                </div>

                <div className="flex items-center gap-1.5 shrink-0">
                  <button
                    type="button"
                    onClick={() => onConfigureLoad(m)}
                    className="bibulm-btn-primary px-3.5 font-semibold"
                    title="Configure load parameters & load model"
                  >
                    <Sliders className="w-3.5 h-3.5" />
                    <span>Load</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      navigator.clipboard.writeText(m.id);
                      setCopiedId(m.id);
                      setTimeout(() => setCopiedId(null), 1500);
                    }}
                    className="apple-btn-ghost w-8 h-8"
                    title="Copy model ID"
                    aria-label="Copy model ID"
                  >
                    {copiedId === m.id ? (
                      <Check className="w-3.5 h-3.5 text-[var(--signal-green)]" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() => api.revealInExplorer(m.path)}
                    className="apple-btn-ghost w-8 h-8"
                    title="Reveal in Explorer"
                    aria-label="Reveal in Explorer"
                  >
                    <FolderOpen className="w-3.5 h-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={() => setConfirmDelete(m)}
                    className="apple-btn-ghost w-8 h-8 hover:text-[var(--signal-red)]"
                    title="Delete from disk"
                    aria-label="Delete from disk"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Delete Confirmation Modal (macOS Alert Sheet) */}
      {confirmDelete && (
        <div className="apple-modal-backdrop">
          <div className="w-[420px] max-w-full apple-sheet p-5 space-y-4">
            <div className="text-[15px] font-semibold text-[var(--label-primary)]">
              Delete Model from Disk?
            </div>
            <p className="text-[12px] text-[var(--label-secondary)]">
              This will permanently remove{' '}
              <strong className="text-[var(--label-primary)]">{confirmDelete.display_name}</strong>{' '}
              ({formatBytes(confirmDelete.size_bytes)}, {confirmDelete.shard_count} file
              {confirmDelete.shard_count > 1 ? 's' : ''}).
            </p>
            <p className="text-[11px] font-mono text-[var(--label-secondary)] break-all apple-card-inset px-3 py-2">
              {confirmDelete.path}
            </p>
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setConfirmDelete(null)}
                className="bibulm-btn-secondary"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => handleDelete(confirmDelete)}
                className="apple-btn-destructive"
              >
                Delete Permanently
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
