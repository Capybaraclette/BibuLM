import React, { useEffect, useState } from 'react';
import { open } from '@tauri-apps/plugin-dialog';
import { Brain, FileCode2, FolderOpen, X } from 'lucide-react';
import type { LoadConfig, LocalModelEntry, ReasoningEffort, ReasoningFormat } from '../../types';
import { api } from '../../api';
import { Toggle } from '../Toggle';
import { fileNameOnly } from './FieldPrimitives';

interface ReasoningSectionProps {
  model: LocalModelEntry;
  config: LoadConfig;
  isRemote: boolean;
  onPatch: (p: Partial<LoadConfig>) => void;
}

export const ReasoningSection: React.FC<ReasoningSectionProps> = ({
  model,
  config,
  isRemote,
  onPatch,
}) => {
  const [templatePreviewOpen, setTemplatePreviewOpen] = useState(false);
  const [templateFilePreview, setTemplateFilePreview] = useState<string | null>(null);
  const [templatePreviewError, setTemplatePreviewError] = useState<string | null>(null);

  useEffect(() => {
    if (!templatePreviewOpen || isRemote) return;
    if (!config.chat_template_file) {
      setTemplateFilePreview(null);
      setTemplatePreviewError(null);
      return;
    }
    let cancelled = false;
    api
      .readTextFilePreview(config.chat_template_file)
      .then((txt) => {
        if (!cancelled) {
          setTemplateFilePreview(txt);
          setTemplatePreviewError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setTemplateFilePreview(null);
          setTemplatePreviewError(String(e));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [templatePreviewOpen, config.chat_template_file, isRemote]);

  const handleBrowseChatTemplateFile = async () => {
    const selected = await open({
      multiple: false,
      title: 'Select Chat Template File (.jinja / .tmpl / .txt)',
      filters: [
        { name: 'Chat Template', extensions: ['jinja', 'j2', 'tmpl', 'txt', 'json'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });
    if (typeof selected === 'string' && selected.trim()) {
      onPatch({
        chat_template_file: selected,
        chat_template_override: null,
        jinja: true,
      });
    }
  };

  return (
    <>
      {/* Chat Template */}
      <div className="apple-card-inset p-4 space-y-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <FileCode2 className="w-4 h-4 text-[var(--accent-text)]" />
            <span className="font-semibold text-[var(--label-primary)]">Chat Template</span>
            <span className="apple-badge text-[10px]">
              {config.chat_template_file
                ? 'Custom File'
                : config.chat_template_override
                  ? 'Built-in Override'
                  : model.has_chat_template
                    ? 'GGUF Embedded'
                    : 'Default'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[var(--label-secondary)]">Jinja (--jinja)</span>
            <Toggle size="sm" checked={config.jinja} onChange={(v) => onPatch({ jinja: v })} />
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {!isRemote && (
            <button
              type="button"
              onClick={handleBrowseChatTemplateFile}
              className="bibulm-btn-secondary h-8"
            >
              <FolderOpen className="w-3.5 h-3.5" />
              <span>Load Template File…</span>
            </button>
          )}

          {config.chat_template_file ? (
            <div className="flex-1 min-w-0 flex items-center justify-between gap-2 h-8 px-2.5 rounded-lg bg-[var(--bg-surface)] border border-[var(--accent-border)]">
              <span
                className="font-mono text-[11px] text-[var(--accent-text)] truncate"
                title={config.chat_template_file}
              >
                {fileNameOnly(config.chat_template_file)}
              </span>
              <div className="flex items-center gap-1 shrink-0">
                {!isRemote && (
                  <button
                    type="button"
                    onClick={() => setTemplatePreviewOpen((v) => !v)}
                    className="bibulm-btn-secondary h-6 px-2 text-[10px]"
                  >
                    {templatePreviewOpen ? 'Hide' : 'Preview'}
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => {
                    onPatch({ chat_template_file: null });
                    setTemplatePreviewOpen(false);
                  }}
                  className="apple-btn-ghost w-6 h-6 hover:text-[var(--signal-red)]"
                  title="Clear template file"
                  aria-label="Clear template file"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          ) : (
            <div className="flex-1 min-w-[200px] flex items-center gap-2">
              <input
                value={config.chat_template_override ?? ''}
                onChange={(e) => onPatch({ chat_template_override: e.target.value || null })}
                placeholder="Optional built-in name (e.g. chatml, llama3) or leave empty for GGUF template"
                className="flex-1 bibulm-input h-8"
              />
              {model.chat_template_preview && (
                <button
                  type="button"
                  onClick={() => setTemplatePreviewOpen((v) => !v)}
                  className="bibulm-btn-secondary h-8 shrink-0"
                >
                  {templatePreviewOpen ? 'Hide' : 'Inspect GGUF'}
                </button>
              )}
            </div>
          )}
        </div>

        {templatePreviewOpen && (
          <div className="rounded-lg bg-[var(--bg-terminal)] border border-[var(--border-hairline)] p-2.5">
            {templatePreviewError ? (
              <div className="text-[11px] text-[#ff6961]">{templatePreviewError}</div>
            ) : (
              <pre className="text-[10px] font-mono text-[#e5e5ea] whitespace-pre-wrap break-all max-h-36 overflow-y-auto">
                {config.chat_template_file
                  ? (templateFilePreview ?? 'Loading file…')
                  : (model.chat_template_preview ?? 'No embedded template found.')}
              </pre>
            )}
          </div>
        )}
      </div>

      {/* Thinking / Reasoning Controls */}
      <div className="apple-card-inset p-4 space-y-3">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-[var(--signal-green)]" />
            <span className="font-semibold text-[var(--label-primary)]">
              Thinking / Reasoning Controls
            </span>
          </div>
          <div className="flex gap-1.5 text-[10px]">
            {model.supports_reasoning_effort && (
              <span className="apple-badge apple-badge-green text-[10px]">
                reasoning_effort detected
              </span>
            )}
            {model.supports_enable_thinking && (
              <span className="apple-badge apple-badge-teal text-[10px]">
                enable_thinking detected
              </span>
            )}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[var(--label-secondary)]">Reasoning Effort</span>
            <select
              value={config.reasoning.effort}
              onChange={(e) =>
                onPatch({
                  reasoning: {
                    ...config.reasoning,
                    effort: e.target.value as ReasoningEffort,
                  },
                })
              }
              className="apple-select h-7"
            >
              <option value="default">Model Default</option>
              <option value="off">Off (none)</option>
              <option value="minimal">Minimal</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="xhigh">XHigh</option>
              <option value="max">Max</option>
            </select>
          </div>
          <div className="flex items-center justify-between gap-2">
            <span className="text-[var(--label-secondary)]">Reasoning Format</span>
            <select
              value={config.reasoning.format}
              onChange={(e) =>
                onPatch({
                  reasoning: {
                    ...config.reasoning,
                    format: e.target.value as ReasoningFormat,
                  },
                })
              }
              className="apple-select h-7"
            >
              <option value="auto">Auto</option>
              <option value="deepseek">DeepSeek (&lt;think&gt;)</option>
              <option value="deepseek_legacy">DeepSeek Legacy</option>
              <option value="none">None (Raw)</option>
            </select>
          </div>
          <div className="flex items-center justify-between gap-2">
            <span className="text-[var(--label-secondary)]">Thinking Toggle (Qwen3)</span>
            <select
              value={
                config.reasoning.enable_thinking === null
                  ? 'auto'
                  : config.reasoning.enable_thinking
                    ? 'true'
                    : 'false'
              }
              onChange={(e) =>
                onPatch({
                  reasoning: {
                    ...config.reasoning,
                    enable_thinking:
                      e.target.value === 'auto' ? null : e.target.value === 'true',
                  },
                })
              }
              className="apple-select h-7"
            >
              <option value="auto">Template Default</option>
              <option value="true">Force Thinking ON</option>
              <option value="false">Force Thinking OFF</option>
            </select>
          </div>
          <div className="flex items-center justify-between gap-2">
            <span className="text-[var(--label-secondary)]">Reasoning Budget</span>
            <input
              type="number"
              placeholder="-1 unlimited / 0 off"
              value={config.reasoning.budget ?? ''}
              onChange={(e) =>
                onPatch({
                  reasoning: {
                    ...config.reasoning,
                    budget: e.target.value === '' ? null : parseInt(e.target.value, 10),
                  },
                })
              }
              className="w-36 bibulm-input h-7 text-right font-mono tabular-nums"
            />
          </div>
        </div>

        <div className="space-y-1">
          <span className="text-[var(--label-secondary)]">Reasoning Budget Cutoff Message</span>
          <input
            value={config.reasoning.budget_message ?? ''}
            onChange={(e) =>
              onPatch({
                reasoning: {
                  ...config.reasoning,
                  budget_message: e.target.value || null,
                },
              })
            }
            placeholder="Optional message injected when reasoning budget is reached…"
            className="w-full bibulm-input h-7"
          />
        </div>
      </div>
    </>
  );
};
