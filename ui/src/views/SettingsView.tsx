import React, { useEffect, useState } from 'react';
import { open } from '@tauri-apps/plugin-dialog';
import { openUrl } from '@tauri-apps/plugin-opener';
import {
  AlertTriangle,
  ArrowDownToLine,
  Check,
  CheckCircle2,
  ExternalLink,
  FolderOpen,
  FolderPlus,
  KeyRound,
  Keyboard,
  RefreshCw,
  Save,
  Settings,
  ShieldCheck,
  Trash2,
} from 'lucide-react';
import type {
  AppSettings,
  AppUpdateInfo,
  AppUpdateProgress,
  ExternalModelSource,
} from '../types';
import { api, formatBytes, formatRelativeDate } from '../api';
import { Toggle } from '../components/Toggle';
import { PageHeader } from '../components/PageHeader';

interface SettingsViewProps {
  settings: AppSettings;
  onSaveSettings: (next: AppSettings) => void;
  onRescanLibrary: () => void;
  appVersion: string;
  updateInfo: AppUpdateInfo | null;
  updateProgress: AppUpdateProgress | null;
  checkingUpdate: boolean;
  installingUpdate: boolean;
  updateError: string | null;
  onCheckUpdate: () => void;
  onInstallUpdate: (info: AppUpdateInfo) => void;
}

export const SettingsView: React.FC<SettingsViewProps> = ({
  settings,
  onSaveSettings,
  onRescanLibrary,
  appVersion,
  updateInfo,
  updateProgress,
  checkingUpdate,
  installingUpdate,
  updateError,
  onCheckUpdate,
  onInstallUpdate,
}) => {
  const [hfToken, setHfToken] = useState(settings.hf_token ?? '');
  const [ghToken, setGhToken] = useState(settings.github_token ?? '');
  const [savedNotice, setSavedNotice] = useState(false);
  const [discoveredSources, setDiscoveredSources] = useState<ExternalModelSource[]>([]);

  useEffect(() => {
    api
      .listDiscoveredModelSources()
      .then(setDiscoveredSources)
      .catch(() => setDiscoveredSources([]));
  }, []);

  const handleSaveTokens = () => {
    onSaveSettings({
      ...settings,
      hf_token: hfToken.trim() || null,
      github_token: ghToken.trim() || null,
    });
    setSavedNotice(true);
    setTimeout(() => setSavedNotice(false), 1500);
  };

  const handleAddModelsDir = async () => {
    const picked = await open({ directory: true, multiple: false, title: 'Select Models Directory' });
    if (typeof picked === 'string' && !settings.models_dirs.includes(picked)) {
      await onSaveSettings({ ...settings, models_dirs: [...settings.models_dirs, picked] });
      onRescanLibrary();
    }
  };

  const handlePickRuntimesDir = async () => {
    const picked = await open({ directory: true, multiple: false, title: 'Select Runtimes Directory' });
    if (typeof picked === 'string') {
      onSaveSettings({ ...settings, runtimes_dir: picked });
    }
  };

  return (
    <div className="h-full flex flex-col overflow-hidden text-[12px]">
      <PageHeader
        icon={<Settings className="w-4 h-4" />}
        title="Settings"
        subtitle="App updates, storage directories, desktop tray lifecycle, and HuggingFace / GitHub access tokens."
        badge={<span className="apple-badge font-mono text-[10px]">v{appVersion}</span>}
      />

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto space-y-5">
      {/* Application Updates */}
      <section className="bibulm-card p-5 space-y-3.5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <ArrowDownToLine className="w-4 h-4 text-[var(--accent-text)]" />
            <span className="text-[13px] font-semibold text-[var(--label-primary)]">
              Application Updates
            </span>
            <span className="apple-badge font-mono text-[10px]">v{appVersion}</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => openUrl('https://github.com/Capybaraclette/bibulm/releases')}
              className="bibulm-btn-secondary"
            >
              <ExternalLink className="w-3.5 h-3.5" /> Releases
            </button>
            <button
              type="button"
              onClick={onCheckUpdate}
              disabled={checkingUpdate || installingUpdate}
              className="bibulm-btn-primary"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${checkingUpdate ? 'animate-spin' : ''}`} />
              <span>{checkingUpdate ? 'Checking…' : 'Check for Updates'}</span>
            </button>
          </div>
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-[var(--border-hairline)]">
          <div>
            <div className="font-medium text-[var(--label-primary)]">
              Check for Bibulm updates on launch
            </div>
            <div className="text-[11px] text-[var(--label-secondary)]">
              Queries GitHub Releases silently at startup and notifies you when a new installer is ready.
            </div>
          </div>
          <Toggle
            checked={settings.auto_check_app_updates}
            onChange={(v) => onSaveSettings({ ...settings, auto_check_app_updates: v })}
          />
        </div>

        {updateError && (
          <div className="flex items-start gap-2 rounded-xl bg-[var(--signal-red-soft)] border border-[var(--signal-red-border)] px-3 py-2 text-[11px] text-[var(--signal-red)]">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            <span>{updateError}</span>
          </div>
        )}

        {updateInfo && !updateInfo.update_available && !updateError && (
          <div className="flex items-center gap-2 rounded-xl bg-[var(--signal-green-soft)] border border-[var(--signal-green-border)] px-3 py-2 text-[11px] text-[var(--signal-green)]">
            <CheckCircle2 className="w-4 h-4 shrink-0" />
            <span>
              Bibulm is up to date (<strong>v{updateInfo.current_version}</strong>).
            </span>
          </div>
        )}

        {updateInfo?.update_available && (
          <div className="rounded-xl bg-[var(--accent-soft)] border border-[var(--accent-border)] p-3.5 space-y-2.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="space-y-0.5">
                <div className="font-semibold text-[var(--label-primary)] flex items-center gap-2">
                  <span>
                    {updateInfo.release_name || `Bibulm v${updateInfo.latest_version}`}
                  </span>
                  {updateInfo.sha256 && (
                    <span className="apple-badge apple-badge-green text-[10px] font-normal">
                      <ShieldCheck className="w-3 h-3" /> SHA-256 verified
                    </span>
                  )}
                </div>
                <div className="text-[11px] text-[var(--label-secondary)] font-mono tabular-nums">
                  v{updateInfo.current_version} → v{updateInfo.latest_version}
                  {updateInfo.published_at && ` • ${formatRelativeDate(updateInfo.published_at)}`}
                  {updateInfo.asset_name &&
                    ` • ${updateInfo.asset_name} (${formatBytes(updateInfo.asset_size_bytes)})`}
                </div>
              </div>
              <div className="flex items-center gap-2">
                {updateInfo.asset_url ? (
                  <button
                    type="button"
                    onClick={() => onInstallUpdate(updateInfo)}
                    disabled={installingUpdate}
                    className="bibulm-btn-primary"
                  >
                    <ArrowDownToLine className="w-3.5 h-3.5" />
                    <span>{installingUpdate ? 'Installing…' : 'Download & Install Update'}</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => openUrl(updateInfo.html_url)}
                    className="bibulm-btn-secondary"
                  >
                    <ExternalLink className="w-3.5 h-3.5" /> Open Release Page
                  </button>
                )}
              </div>
            </div>

            {installingUpdate && updateProgress && (
              <div className="space-y-1">
                <div className="flex items-center justify-between text-[11px] font-mono text-[var(--accent-text)] tabular-nums">
                  <span>{updateProgress.message}</span>
                  <span>
                    {updateProgress.total_bytes > 0
                      ? `${formatBytes(updateProgress.downloaded_bytes)} / ${formatBytes(updateProgress.total_bytes)} (${updateProgress.percent.toFixed(0)}%)`
                      : `${updateProgress.percent.toFixed(0)}%`}
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-[var(--bg-control)] overflow-hidden">
                  <div
                    className="h-full bg-[var(--accent)] transition-all duration-200 rounded-full"
                    style={{ width: `${Math.min(100, Math.max(0, updateProgress.percent))}%` }}
                  />
                </div>
              </div>
            )}

            {updateInfo.release_notes.trim() && (
              <pre className="max-h-36 overflow-y-auto whitespace-pre-wrap rounded-lg bg-[var(--bg-terminal)] border border-[var(--border-hairline)] p-2.5 text-[11px] font-mono text-[#e5e5ea]">
                {updateInfo.release_notes.trim()}
              </pre>
            )}
          </div>
        )}
      </section>

      {/* Storage Directories */}
      <section className="bibulm-card p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div className="text-[13px] font-semibold text-[var(--label-primary)]">
            Model Storage Directories
          </div>
          <button type="button" onClick={handleAddModelsDir} className="bibulm-btn-secondary">
            <FolderPlus className="w-3.5 h-3.5" /> Add Folder
          </button>
        </div>

        <div className="space-y-1.5">
          {settings.models_dirs.map((dir, i) => (
            <div
              key={dir}
              className="apple-card-inset flex items-center justify-between gap-2 px-3 py-2"
            >
              <div className="min-w-0 flex items-center gap-2">
                {i === 0 && (
                  <span className="apple-badge apple-badge-accent text-[10px]">Primary</span>
                )}
                <span className="font-mono text-[var(--label-primary)] truncate">{dir}</span>
              </div>
              <div className="flex items-center gap-1 shrink-0">
                <button
                  type="button"
                  onClick={() => api.revealInExplorer(dir)}
                  className="apple-btn-ghost w-7 h-7"
                  title="Open in Explorer"
                  aria-label="Open directory in Explorer"
                >
                  <FolderOpen className="w-3.5 h-3.5" />
                </button>
                {settings.models_dirs.length > 1 && (
                  <button
                    type="button"
                    onClick={async () => {
                      await onSaveSettings({
                        ...settings,
                        models_dirs: settings.models_dirs.filter((d) => d !== dir),
                      });
                      onRescanLibrary();
                    }}
                    className="apple-btn-ghost w-7 h-7 hover:text-[var(--signal-red)]"
                    title="Remove directory from library"
                    aria-label="Remove directory"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="pt-3 border-t border-[var(--border-hairline)] space-y-2.5">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="font-medium text-[var(--label-primary)]">
                Auto-discover models from external apps
              </div>
              <div className="text-[11px] text-[var(--label-secondary)]">
                Automatically scans Ollama, LM Studio, Jan, GPT4All, llama.cpp, HuggingFace Hub,
                AnythingLLM, Msty, Backyard AI, LocalAI, Nexa, and legacy rustStudio folders.
              </div>
            </div>
            <Toggle
              checked={settings.auto_discover_external_models}
              onChange={async (v) => {
                await onSaveSettings({ ...settings, auto_discover_external_models: v });
                onRescanLibrary();
              }}
            />
          </div>

          {settings.auto_discover_external_models && discoveredSources.length > 0 && (
            <div className="space-y-1.5">
              <div className="text-[11px] text-[var(--label-secondary)] font-medium">
                Detected External Model Folders ({discoveredSources.length})
              </div>
              {discoveredSources.map((src) => (
                <div
                  key={`${src.app_name}:${src.path}`}
                  className="apple-card-inset flex items-center justify-between gap-2 px-3 py-1.5"
                >
                  <div className="min-w-0 flex items-center gap-2">
                    <span className="apple-badge apple-badge-accent text-[10px] font-medium shrink-0">
                      {src.app_name}
                    </span>
                    <span className="font-mono text-[var(--label-primary)] truncate">{src.path}</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => api.revealInExplorer(src.path)}
                    className="apple-btn-ghost w-7 h-7 shrink-0"
                    title="Open in Explorer"
                    aria-label="Open external source in Explorer"
                  >
                    <FolderOpen className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-3 border-t border-[var(--border-hairline)]">
          <div className="space-y-1">
            <span className="text-[var(--label-secondary)] font-medium">Runtimes Directory</span>
            <div className="flex gap-1.5">
              <input
                readOnly
                value={settings.runtimes_dir}
                className="flex-1 bibulm-input font-mono text-[var(--label-secondary)]"
              />
              <button
                type="button"
                onClick={handlePickRuntimesDir}
                className="bibulm-btn-secondary"
              >
                Change
              </button>
            </div>
          </div>
          <div className="space-y-1">
            <span className="text-[var(--label-secondary)] font-medium">
              Rotating Server Logs Directory
            </span>
            <div className="flex gap-1.5">
              <input
                readOnly
                value={settings.logs_dir}
                className="flex-1 bibulm-input font-mono text-[var(--label-secondary)]"
              />
              <button
                type="button"
                onClick={() => api.revealInExplorer(settings.logs_dir)}
                className="bibulm-btn-secondary"
              >
                Open
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Tokens */}
      <section className="bibulm-card p-5 space-y-3.5">
        <div className="flex items-center justify-between">
          <div className="text-[13px] font-semibold text-[var(--label-primary)] flex items-center gap-1.5">
            <KeyRound className="w-4 h-4 text-[var(--accent-text)]" /> Access Tokens (Optional)
          </div>
          <button type="button" onClick={handleSaveTokens} className="bibulm-btn-primary">
            {savedNotice ? (
              <>
                <Check className="w-3.5 h-3.5" /> Saved
              </>
            ) : (
              <>
                <Save className="w-3.5 h-3.5" /> Save Tokens
              </>
            )}
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="space-y-1">
            <label className="text-[var(--label-secondary)]">
              HuggingFace Access Token (for gated GGUFs)
            </label>
            <input
              type="password"
              value={hfToken}
              onChange={(e) => setHfToken(e.target.value)}
              placeholder="hf_..."
              className="w-full bibulm-input font-mono"
            />
          </div>
          <div className="space-y-1">
            <label className="text-[var(--label-secondary)]">
              GitHub Personal Access Token (higher API rate limit)
            </label>
            <input
              type="password"
              value={ghToken}
              onChange={(e) => setGhToken(e.target.value)}
              placeholder="ghp_..."
              className="w-full bibulm-input font-mono"
            />
          </div>
        </div>
      </section>

      {/* Desktop & System Tray */}
      <section className="bibulm-card p-5 divide-y divide-[var(--border-hairline)]">
        <div className="text-[13px] font-semibold text-[var(--label-primary)] pb-3">
          Desktop &amp; Background Behavior
        </div>

        <div className="flex items-center justify-between py-3">
          <div>
            <div className="font-medium text-[var(--label-primary)]">
              Auto-start OpenAI API Server on launch
            </div>
            <div className="text-[11px] text-[var(--label-secondary)]">
              Starts the local gateway immediately when Bibulm opens.
            </div>
          </div>
          <Toggle
            checked={settings.auto_start_server}
            onChange={(v) => onSaveSettings({ ...settings, auto_start_server: v })}
          />
        </div>

        <div className="flex items-center justify-between py-3">
          <div>
            <div className="font-medium text-[var(--label-primary)]">
              Minimize to System Tray on close
            </div>
            <div className="text-[11px] text-[var(--label-secondary)]">
              Stops WebView processes to free RAM/VRAM while keeping the API server, Link, and loaded
              models running in the tray; recreates the UI when reopened.
            </div>
          </div>
          <Toggle
            checked={settings.minimize_to_tray}
            onChange={(v) => onSaveSettings({ ...settings, minimize_to_tray: v })}
          />
        </div>

        <div className="flex items-center justify-between pt-3">
          <div>
            <div className="font-medium text-[var(--label-primary)]">Start minimized to tray</div>
            <div className="text-[11px] text-[var(--label-secondary)]">
              Launch silently in the background without popping up the window.
            </div>
          </div>
          <Toggle
            checked={settings.start_minimized}
            onChange={(v) => onSaveSettings({ ...settings, start_minimized: v })}
          />
        </div>
      </section>

      {/* Keyboard Shortcuts Reference */}
      <section className="bibulm-card p-5 space-y-3">
        <div className="text-[13px] font-semibold text-[var(--label-primary)] flex items-center gap-1.5">
          <Keyboard className="w-4 h-4 text-[var(--label-tertiary)]" /> Keyboard Shortcuts
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[12px]">
          {[
            ['Ctrl + L', 'Quick Load / Switch Model'],
            ['Ctrl + R', 'Reload Last Used Model'],
            ['Ctrl + Shift + R', 'Open Runtime Manager'],
            ['Ctrl + Shift + H', 'Open Hardware & Telemetry'],
            ['Ctrl + Enter', 'Confirm Load in Load Config Modal'],
          ].map(([key, label]) => (
            <div
              key={key}
              className="apple-card-inset flex items-center justify-between px-3 py-2"
            >
              <span className="text-[var(--label-primary)]">{label}</span>
              <kbd className="apple-kbd">{key}</kbd>
            </div>
          ))}
        </div>
      </section>
        </div>
      </div>
    </div>
  );
};
