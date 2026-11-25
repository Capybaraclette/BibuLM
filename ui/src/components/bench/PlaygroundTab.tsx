import React, { useEffect, useMemo, useRef, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import {
  Brain,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ImagePlus,
  Loader2,
  Send,
  Sparkles,
  Trash2,
  Wrench,
  X,
  XCircle,
} from 'lucide-react';
import type {
  ActiveRemoteModel,
  LoadedModelInfo,
  PlaygroundMessage,
  PlaygroundStreamChunk,
  PlaygroundTurnResult,
  ReasoningEffort,
  SamplingParams,
} from '../../types';
import { api } from '../../api';

interface PlaygroundTabProps {
  activeModel: LoadedModelInfo | null;
  activeRemoteModel: ActiveRemoteModel | null;
  sampling: SamplingParams;
  reasoningEffort: ReasoningEffort;
  onOpenQuickLoad: () => void;
}

interface AttachedImage {
  name: string;
  dataUrl: string;
}

interface PlaygroundEntry {
  id: string;
  role: 'user' | 'assistant' | 'tool';
  text: string;
  images?: AttachedImage[];
  reasoning?: string;
  tool_calls?: Record<string, unknown>[];
  tool_call_id?: string;
  metrics?: PlaygroundTurnResult;
  jsonValid?: boolean | null;
}

const SAMPLE_TOOLS_JSON = JSON.stringify(
  [
    {
      type: 'function',
      function: {
        name: 'get_system_metric',
        description: 'Inspect a hardware or runtime metric on the rig',
        parameters: {
          type: 'object',
          properties: {
            metric: {
              type: 'string',
              enum: ['gpu_temp', 'vram_used', 'ram_used', 'active_slots'],
            },
          },
          required: ['metric'],
        },
      },
    },
  ],
  null,
  2,
);

export const PlaygroundTab: React.FC<PlaygroundTabProps> = ({
  activeModel,
  activeRemoteModel,
  sampling,
  reasoningEffort,
  onOpenQuickLoad,
}) => {
  const [entries, setEntries] = useState<PlaygroundEntry[]>([]);
  const [input, setInput] = useState('');
  const [attachedImages, setAttachedImages] = useState<AttachedImage[]>([]);
  const [toolsOpen, setToolsOpen] = useState(false);
  const [toolsJson, setToolsJson] = useState('');
  const [running, setRunning] = useState(false);
  const [streamReasoning, setStreamReasoning] = useState('');
  const [streamOutput, setStreamOutput] = useState('');
  const [streamTokens, setStreamTokens] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [toolReplyId, setToolReplyId] = useState<string | null>(null);
  const [toolReplyContent, setToolReplyContent] = useState('{"value": "ok"}');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const targetModelId = activeModel?.model_id ?? activeRemoteModel?.qualifiedId ?? null;
  const maxTokens = sampling.max_tokens ?? 1024;

  const toolsValidation = useMemo(() => {
    const trimmed = toolsJson.trim();
    if (!trimmed) return { valid: true, parsed: null as string | null, err: null as string | null };
    try {
      const parsed = JSON.parse(trimmed);
      if (!Array.isArray(parsed)) {
        return { valid: false, parsed: null, err: 'Tools schema must be a JSON array' };
      }
      return { valid: true, parsed: JSON.stringify(parsed), err: null };
    } catch (e) {
      return { valid: false, parsed: null, err: String(e) };
    }
  }, [toolsJson]);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    listen<PlaygroundStreamChunk>('playground-stream-chunk', (ev) => {
      const c = ev.payload;
      if (c.reasoning_delta) setStreamReasoning((prev) => prev + c.reasoning_delta);
      if (c.content_delta) setStreamOutput((prev) => prev + c.content_delta);
      setStreamTokens(c.approx_tokens);
    }).then((fn) => {
      unlisten = fn;
    });
    return () => {
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [entries.length, streamOutput, streamReasoning]);

  const handleImagePick = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const loaded: AttachedImage[] = [];
    for (const file of Array.from(files)) {
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result ?? ''));
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      });
      if (dataUrl) {
        loaded.push({ name: file.name, dataUrl });
      }
    }
    setAttachedImages((prev) => [...prev, ...loaded]);
  };

  const buildApiMessages = (nextEntries: PlaygroundEntry[]): PlaygroundMessage[] => {
    const msgs: PlaygroundMessage[] = [];
    if (sampling.system_prompt.trim()) {
      msgs.push({ role: 'system', content: sampling.system_prompt.trim() });
    }
    for (const e of nextEntries) {
      if (e.role === 'user') {
        if (e.images && e.images.length > 0) {
          const parts: unknown[] = [];
          if (e.text.trim()) {
            parts.push({ type: 'text', text: e.text });
          }
          for (const img of e.images) {
            parts.push({ type: 'image_url', image_url: { url: img.dataUrl } });
          }
          msgs.push({ role: 'user', content: parts });
        } else {
          msgs.push({ role: 'user', content: e.text });
        }
      } else if (e.role === 'assistant') {
        msgs.push({
          role: 'assistant',
          content: e.text,
          tool_calls: e.tool_calls && e.tool_calls.length > 0 ? e.tool_calls : undefined,
        });
      } else if (e.role === 'tool') {
        msgs.push({
          role: 'tool',
          content: e.text,
          tool_call_id: e.tool_call_id ?? 'call_0',
        });
      }
    }
    return msgs;
  };

  const sendEntriesToModel = async (nextEntries: PlaygroundEntry[]) => {
    if (!targetModelId || running) return;
    if (!toolsValidation.valid) {
      setError(toolsValidation.err ?? 'Invalid tools JSON');
      return;
    }

    setRunning(true);
    setError(null);
    setStreamReasoning('');
    setStreamOutput('');
    setStreamTokens(0);

    try {
      const res = await api.runPlaygroundTurn(targetModelId, {
        messages: buildApiMessages(nextEntries),
        max_tokens: maxTokens,
        reasoning_effort: reasoningEffort === 'default' ? null : reasoningEffort,
        sampling,
        tools_json: toolsValidation.parsed,
      });

      let jsonValid: boolean | null = null;
      if (sampling.structured_output_enabled && sampling.json_schema) {
        try {
          JSON.parse(res.output_text);
          jsonValid = true;
        } catch {
          jsonValid = false;
        }
      }

      const assistantEntry: PlaygroundEntry = {
        id: res.turn_id || `turn-${Date.now()}`,
        role: 'assistant',
        text: res.output_text,
        reasoning: res.reasoning_text || undefined,
        tool_calls: res.tool_calls && res.tool_calls.length > 0 ? res.tool_calls : undefined,
        metrics: res,
        jsonValid,
      };
      setEntries([...nextEntries, assistantEntry]);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  };

  const handleSendUserTurn = async () => {
    if ((!input.trim() && attachedImages.length === 0) || !targetModelId || running) return;
    const userEntry: PlaygroundEntry = {
      id: `user-${Date.now()}`,
      role: 'user',
      text: input.trim(),
      images: attachedImages.length > 0 ? [...attachedImages] : undefined,
    };
    const nextEntries = [...entries, userEntry];
    setEntries(nextEntries);
    setInput('');
    setAttachedImages([]);
    await sendEntriesToModel(nextEntries);
  };

  const handleSendToolResult = async () => {
    if (!toolReplyId || !targetModelId || running) return;
    const toolEntry: PlaygroundEntry = {
      id: `tool-${Date.now()}`,
      role: 'tool',
      text: toolReplyContent,
      tool_call_id: toolReplyId,
    };
    const nextEntries = [...entries, toolEntry];
    setEntries(nextEntries);
    setToolReplyId(null);
    await sendEntriesToModel(nextEntries);
  };

  return (
    <div className="space-y-4">
      {/* Top Toolbar: System Prompt Status + Tools Toggle + Clear */}
      <div className="bibulm-card p-3.5 space-y-2.5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-[12px] text-[var(--label-primary)] flex-wrap">
            <Sparkles className="w-3.5 h-3.5 text-[var(--accent-text)]" />
            <span className="font-semibold">API Playground</span>
            {sampling.system_prompt.trim() && (
              <span
                className="apple-badge apple-badge-accent text-[10px]"
                title={sampling.system_prompt}
              >
                System prompt active
              </span>
            )}
            {sampling.structured_output_enabled && sampling.json_schema && (
              <span className="apple-badge apple-badge-green font-mono text-[10px]">
                JSON Schema Active
              </span>
            )}
            {activeModel?.load_config.enable_vision && (
              <span className="apple-badge apple-badge-teal text-[10px]">
                Vision Ready
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setToolsOpen((v) => !v)}
              className={`h-7 px-2.5 rounded-lg border text-[11px] font-medium flex items-center gap-1 transition-colors cursor-pointer ${
                toolsValidation.parsed
                  ? 'bg-[var(--signal-amber-soft)] border-[var(--signal-amber-border)] text-[var(--signal-amber)]'
                  : 'bg-[var(--bg-surface)] hover:bg-[var(--bg-control-hover)] border-[var(--border-hairline)] text-[var(--label-primary)]'
              }`}
            >
              <Wrench className="w-3 h-3" />
              <span>Tools JSON</span>
              {toolsOpen ? (
                <ChevronDown className="w-3 h-3" />
              ) : (
                <ChevronRight className="w-3 h-3" />
              )}
            </button>
            {entries.length > 0 && (
              <button
                type="button"
                onClick={() => {
                  setEntries([]);
                  setStreamOutput('');
                  setStreamReasoning('');
                  setError(null);
                }}
                className="bibulm-btn-secondary h-7 px-2.5 text-[11px]"
              >
                <Trash2 className="w-3 h-3" /> Reset Chat
              </button>
            )}
          </div>
        </div>

        {toolsOpen && (
          <div className="pt-2.5 border-t border-[var(--border-hairline)] space-y-2">
            <div className="flex items-center justify-between gap-2 text-[11px]">
              <span className="text-[var(--label-secondary)]">
                OpenAI <code className="text-[var(--label-primary)]">tools</code> array sent with
                every turn:
              </span>
              <div className="flex items-center gap-2">
                {!toolsValidation.valid && (
                  <span className="text-[var(--signal-red)] text-[10px] font-mono">
                    {toolsValidation.err}
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => setToolsJson(SAMPLE_TOOLS_JSON)}
                  className="bibulm-btn-secondary h-6 px-2 text-[10px]"
                >
                  Insert Sample Tool
                </button>
                {toolsJson && (
                  <button
                    type="button"
                    onClick={() => setToolsJson('')}
                    className="bibulm-btn-secondary h-6 px-2 text-[10px]"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
            <textarea
              rows={5}
              value={toolsJson}
              onChange={(e) => setToolsJson(e.target.value)}
              placeholder='[{"type": "function", "function": {"name": "...", "parameters": {...}}}]'
              className="w-full p-2.5 rounded-lg bg-[var(--bg-terminal)] border border-[var(--border-hairline)] text-[12px] font-mono text-[#e5e5ea] focus:outline-none focus:border-[var(--accent)]"
            />
          </div>
        )}
      </div>

      {!targetModelId && (
        <div className="bibulm-card p-6 text-center space-y-2.5">
          <div className="text-[13px] text-[var(--label-primary)] font-medium">
            Load a local model or select a remote peer model to start testing turns.
          </div>
          <button type="button" onClick={onOpenQuickLoad} className="bibulm-btn-primary">
            Quick Load Model
          </button>
        </div>
      )}

      {/* Conversation Transcript */}
      {entries.length > 0 && (
        <div className="space-y-3">
          {entries.map((entry) => {
            if (entry.role === 'user') {
              return (
                <div key={entry.id} className="flex justify-end">
                  <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-[var(--accent)] px-4 py-2.5 text-[13px] text-white space-y-2 shadow-sm">
                    {entry.images && entry.images.length > 0 && (
                      <div className="flex flex-wrap gap-2">
                        {entry.images.map((img, i) => (
                          <img
                            key={i}
                            src={img.dataUrl}
                            alt={img.name}
                            className="max-h-36 rounded-lg border border-white/25 object-contain"
                          />
                        ))}
                      </div>
                    )}
                    {entry.text && <div className="whitespace-pre-wrap">{entry.text}</div>}
                  </div>
                </div>
              );
            }

            if (entry.role === 'tool') {
              return (
                <div key={entry.id} className="flex justify-end">
                  <div className="max-w-[85%] rounded-xl bg-[var(--signal-amber-soft)] border border-[var(--signal-amber-border)] px-3 py-2 text-[11px] font-mono text-[var(--label-primary)]">
                    <div className="text-[10px] text-[var(--signal-amber)] font-semibold mb-0.5">
                      tool result ({entry.tool_call_id})
                    </div>
                    <div className="whitespace-pre-wrap">{entry.text}</div>
                  </div>
                </div>
              );
            }

            const cachePct =
              entry.metrics && entry.metrics.prompt_tokens > 0
                ? Math.round((entry.metrics.cached_tokens / entry.metrics.prompt_tokens) * 100)
                : 0;

            return (
              <div key={entry.id} className="bibulm-card p-4 space-y-2.5 text-[13px]">
                {entry.reasoning && (
                  <div className="rounded-lg bg-[var(--signal-purple-soft)] border border-[var(--signal-purple-border)] p-3">
                    <div className="flex items-center gap-1.5 text-[11px] text-[var(--signal-purple)] font-semibold mb-1">
                      <Brain className="w-3.5 h-3.5" /> Reasoning
                    </div>
                    <pre className="text-[11px] font-mono text-[var(--label-secondary)] whitespace-pre-wrap leading-relaxed max-h-48 overflow-y-auto">
                      {entry.reasoning}
                    </pre>
                  </div>
                )}

                {entry.text && (
                  <pre className="text-[13px] font-sans text-[var(--label-primary)] whitespace-pre-wrap leading-relaxed">
                    {entry.text}
                  </pre>
                )}

                {entry.tool_calls && entry.tool_calls.length > 0 && (
                  <div className="rounded-lg bg-[var(--signal-amber-soft)] border border-[var(--signal-amber-border)] p-3 space-y-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[11px] font-semibold text-[var(--signal-amber)] flex items-center gap-1">
                        <Wrench className="w-3 h-3" /> Tool Calls ({entry.tool_calls.length})
                      </span>
                      {typeof entry.tool_calls[0]?.id === 'string' && (
                        <button
                          type="button"
                          onClick={() => setToolReplyId(String(entry.tool_calls?.[0]?.id ?? 'call_0'))}
                          className="bibulm-btn-secondary h-6 px-2 text-[10px]"
                        >
                          Reply with tool output
                        </button>
                      )}
                    </div>
                    <pre className="text-[11px] font-mono text-[var(--label-primary)] whitespace-pre-wrap overflow-x-auto">
                      {JSON.stringify(entry.tool_calls, null, 2)}
                    </pre>
                  </div>
                )}

                {(entry.metrics || entry.jsonValid !== null) && (
                  <div className="pt-2.5 border-t border-[var(--border-hairline)] flex flex-wrap items-center gap-1.5 text-[10px] font-mono tabular-nums text-[var(--label-secondary)]">
                    {entry.metrics && (
                      <>
                        <span className="apple-badge apple-badge-teal">
                          TTFT {entry.metrics.ttft_ms.toFixed(0)} ms
                        </span>
                        <span className="apple-badge apple-badge-accent">
                          pp {entry.metrics.prompt_eval_tok_s.toFixed(1)} tok/s
                        </span>
                        <span className="apple-badge apple-badge-green">
                          tg {entry.metrics.generation_tok_s.toFixed(1)} tok/s
                        </span>
                        <span className="apple-badge">
                          Prompt {entry.metrics.prompt_tokens} · Gen {entry.metrics.completion_tokens}
                        </span>
                        <span
                          className={`apple-badge ${
                            entry.metrics.cached_tokens > 0 ? 'apple-badge-green' : ''
                          }`}
                        >
                          KV Cache hit {entry.metrics.cached_tokens} ({cachePct}%)
                        </span>
                      </>
                    )}
                    {entry.jsonValid === true && (
                      <span className="apple-badge apple-badge-green">
                        <CheckCircle2 className="w-3 h-3" /> Valid JSON
                      </span>
                    )}
                    {entry.jsonValid === false && (
                      <span className="apple-badge apple-badge-red">
                        <XCircle className="w-3 h-3" /> Invalid JSON
                      </span>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Streaming live assistant preview */}
      {running && (
        <div className="bibulm-card p-4 space-y-2 border-[var(--accent-border)]">
          <div className="flex items-center justify-between text-[11px] text-[var(--accent-text)] font-mono tabular-nums">
            <span className="flex items-center gap-1.5">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> Streaming turn…
            </span>
            <span>{streamTokens} tokens</span>
          </div>
          {streamReasoning && (
            <pre className="text-[11px] font-mono text-[var(--signal-purple)] whitespace-pre-wrap max-h-40 overflow-y-auto">
              {streamReasoning}
            </pre>
          )}
          {streamOutput && (
            <pre className="text-[13px] font-sans text-[var(--label-primary)] whitespace-pre-wrap">
              {streamOutput}
            </pre>
          )}
        </div>
      )}

      {error && (
        <div className="p-3 rounded-xl bg-[var(--signal-red-soft)] border border-[var(--signal-red-border)] text-[12px] text-[var(--signal-red)] font-mono">
          {error}
        </div>
      )}

      {/* Tool Call Reply Composer */}
      {toolReplyId && (
        <div className="bibulm-card p-3.5 border-[var(--signal-amber-border)] space-y-2">
          <div className="flex items-center justify-between text-[12px] text-[var(--signal-amber)] font-medium">
            <span>
              Submit tool output for call ID: <code>{toolReplyId}</code>
            </span>
            <button
              type="button"
              onClick={() => setToolReplyId(null)}
              className="apple-btn-ghost w-6 h-6"
              aria-label="Cancel tool reply"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
          <div className="flex gap-2">
            <input
              value={toolReplyContent}
              onChange={(e) => setToolReplyContent(e.target.value)}
              className="flex-1 bibulm-input font-mono"
            />
            <button
              type="button"
              onClick={handleSendToolResult}
              disabled={running}
              className="bibulm-btn-primary"
            >
              Send Tool Output
            </button>
          </div>
        </div>
      )}

      {/* Prompt Composer */}
      <div className="bibulm-card p-3 space-y-2">
        {attachedImages.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {attachedImages.map((img, idx) => (
              <div
                key={idx}
                className="relative group rounded-lg border border-[var(--border-hairline)] overflow-hidden bg-[var(--bg-surface-inset)]"
              >
                <img src={img.dataUrl} alt={img.name} className="h-16 w-16 object-cover" />
                <button
                  type="button"
                  onClick={() => setAttachedImages((prev) => prev.filter((_, i) => i !== idx))}
                  className="absolute top-0.5 right-0.5 p-0.5 rounded bg-black/75 text-white hover:text-[#ff453a] cursor-pointer"
                  aria-label="Remove image"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        )}
        {attachedImages.length > 0 && activeModel && !activeModel.load_config.enable_vision && (
          <div className="text-[11px] text-[var(--signal-amber)]">
            Note: this model was loaded without <code className="font-mono">enable_vision</code> /
            mmproj. Reload with Vision enabled if the backend rejects image parts.
          </div>
        )}

        <div className="flex items-end gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(e) => {
              handleImagePick(e.target.files);
              e.target.value = '';
            }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={!targetModelId || running}
            className="bibulm-btn-secondary h-9 px-2.5 shrink-0"
            title="Attach image for Vision models"
            aria-label="Attach image"
          >
            <ImagePlus className="w-4 h-4" />
          </button>
          <textarea
            rows={2}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendUserTurn();
              }
            }}
            placeholder={
              targetModelId
                ? 'Type a prompt (Enter to send, Shift+Enter for newline)…'
                : 'Load a model first…'
            }
            className="flex-1 px-3 py-2 rounded-lg bg-[var(--bg-surface-inset)] border border-[var(--border-hairline)] text-[13px] text-[var(--label-primary)] placeholder-[var(--label-quaternary)] resize-y min-h-[38px] focus:outline-none focus:border-[var(--accent)]"
          />
          <button
            type="button"
            onClick={handleSendUserTurn}
            disabled={!targetModelId || running || (!input.trim() && attachedImages.length === 0)}
            className="bibulm-btn-primary h-9 px-3.5 font-semibold shrink-0"
          >
            {running ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
            <span>Send</span>
          </button>
        </div>
      </div>

      <div ref={chatEndRef} />
    </div>
  );
};
