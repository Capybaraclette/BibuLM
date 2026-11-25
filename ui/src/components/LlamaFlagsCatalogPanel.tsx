import React, { useMemo, useState } from 'react';
import { open } from '@tauri-apps/plugin-dialog';
import { FolderOpen, Plus, RotateCcw, Search, SlidersHorizontal, X } from 'lucide-react';
import type { CliFlagValue, DiscoveredCliFlag } from '../types';
import {
  LLAMA_FLAG_CATEGORIES,
  LLAMA_FLAGS_CATALOG,
  type LlamaFlagCategoryId,
  type LlamaFlagSpec,
} from '../data/llamaFlagsCatalog';
import { Toggle } from './Toggle';

interface LlamaFlagsCatalogPanelProps {
  cliFlags: Record<string, CliFlagValue>;
  discoveredFlags?: DiscoveredCliFlag[];
  onChange: (next: Record<string, CliFlagValue>) => void;
}

function isSpecModified(spec: LlamaFlagSpec, cliFlags: Record<string, CliFlagValue>): boolean {
  if (spec.control === 'tri_bool') {
    const pos = cliFlags[spec.flag];
    const neg = spec.negFlag ? cliFlags[spec.negFlag] : undefined;
    return Boolean(
      (pos && pos.kind === 'flag' && pos.value) || (neg && neg.kind === 'flag' && neg.value),
    );
  }
  const val = cliFlags[spec.flag];
  if (!val) return false;
  if (val.kind === 'flag') return val.value;
  if (val.kind === 'value') return val.value.trim().length > 0;
  return val.value.some((s) => s.trim().length > 0);
}

export const LlamaFlagsCatalogPanel: React.FC<LlamaFlagsCatalogPanelProps> = ({
  cliFlags,
  discoveredFlags,
  onChange,
}) => {
  const [search, setSearch] = useState('');
  const [activeCategory, setActiveCategory] = useState<LlamaFlagCategoryId | 'all'>('all');
  const [modifiedOnly, setModifiedOnly] = useState(false);

  const discoveredSet = useMemo(() => {
    if (!discoveredFlags || discoveredFlags.length === 0) return null;
    const set = new Set<string>();
    for (const f of discoveredFlags) {
      set.add(f.flag);
      for (const a of f.aliases) set.add(a);
    }
    return set;
  }, [discoveredFlags]);

  const uncatalogedFlags = useMemo(() => {
    if (!discoveredFlags || discoveredFlags.length === 0) return [];
    const known = new Set<string>();
    for (const s of LLAMA_FLAGS_CATALOG) {
      known.add(s.flag);
      if (s.negFlag) known.add(s.negFlag);
      for (const a of s.aliases) known.add(a);
    }
    const q = search.trim().toLowerCase();
    return discoveredFlags.filter((d) => {
      if (known.has(d.flag) || d.aliases.some((a) => known.has(a))) return false;
      if (!q) return true;
      return (
        d.flag.toLowerCase().includes(q) ||
        d.description.toLowerCase().includes(q)
      );
    });
  }, [discoveredFlags, search]);

  const modifiedCount = useMemo(
    () => LLAMA_FLAGS_CATALOG.filter((spec) => isSpecModified(spec, cliFlags)).length,
    [cliFlags],
  );

  const categoryCounts = useMemo(() => {
    const total: Record<string, number> = {};
    const modified: Record<string, number> = {};
    for (const spec of LLAMA_FLAGS_CATALOG) {
      total[spec.category] = (total[spec.category] ?? 0) + 1;
      if (isSpecModified(spec, cliFlags)) {
        modified[spec.category] = (modified[spec.category] ?? 0) + 1;
      }
    }
    return { total, modified };
  }, [cliFlags]);

  const filteredSpecs = useMemo(() => {
    const q = search.trim().toLowerCase();
    return LLAMA_FLAGS_CATALOG.filter((spec) => {
      if (activeCategory !== 'all' && spec.category !== activeCategory) return false;
      if (modifiedOnly && !isSpecModified(spec, cliFlags)) return false;
      if (!q) return true;
      return (
        spec.flag.toLowerCase().includes(q) ||
        (spec.negFlag?.toLowerCase().includes(q) ?? false) ||
        spec.label.toLowerCase().includes(q) ||
        spec.description.toLowerCase().includes(q) ||
        (spec.envVar?.toLowerCase().includes(q) ?? false) ||
        spec.aliases.some((a) => a.toLowerCase().includes(q))
      );
    });
  }, [search, activeCategory, modifiedOnly, cliFlags]);

  const setFlagEntry = (key: string, value: CliFlagValue | null, alsoDelete?: string) => {
    const next = { ...cliFlags };
    if (alsoDelete) delete next[alsoDelete];
    if (value === null) {
      delete next[key];
    } else {
      next[key] = value;
    }
    onChange(next);
  };

  const clearSpec = (spec: LlamaFlagSpec) => {
    const next = { ...cliFlags };
    delete next[spec.flag];
    if (spec.negFlag) delete next[spec.negFlag];
    onChange(next);
  };

  const handleBrowsePath = async (spec: LlamaFlagSpec) => {
    const isDir =
      spec.placeholder?.toUpperCase().includes('DIR') ||
      spec.flag.endsWith('-dir') ||
      spec.flag.endsWith('-path');
    const selected = await open({
      multiple: false,
      directory: Boolean(isDir && !spec.flag.includes('file') && !spec.flag.includes('cache')),
      title: `Select ${spec.label} (${spec.flag})`,
    });
    if (typeof selected === 'string' && selected.trim()) {
      setFlagEntry(spec.flag, { kind: 'value', value: selected });
    }
  };

  return (
    <div className="space-y-3 pt-3 border-t border-[var(--border-hairline)]">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="w-4 h-4 text-[var(--accent-text)]" />
          <span className="font-semibold text-[var(--label-primary)]">
            All llama.cpp Flags ({LLAMA_FLAGS_CATALOG.length})
          </span>
          {discoveredFlags && discoveredFlags.length > 0 && (
            <span className="apple-badge apple-badge-green font-mono text-[10px]">
              {discoveredFlags.length} detected via --help
            </span>
          )}
          {modifiedCount > 0 && (
            <span className="apple-badge apple-badge-accent font-mono text-[10px]">
              {modifiedCount} active
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setModifiedOnly((v) => !v)}
            className={`h-7 px-2.5 rounded-lg border text-[11px] font-medium transition-colors cursor-pointer ${
              modifiedOnly
                ? 'bg-[var(--accent-soft)] border-[var(--accent-border)] text-[var(--accent-text)]'
                : 'bg-[var(--bg-surface)] border-[var(--border-hairline)] text-[var(--label-secondary)] hover:text-[var(--label-primary)]'
            }`}
          >
            Modified only ({modifiedCount})
          </button>
          {modifiedCount > 0 && (
            <button
              type="button"
              onClick={() => onChange({})}
              className="bibulm-btn-secondary h-7 px-2.5 text-[11px]"
              title="Reset all catalog flag overrides"
            >
              <RotateCcw className="w-3 h-3" />
              <span>Reset</span>
            </button>
          )}
        </div>
      </div>

      {/* Search bar */}
      <div className="relative">
        <Search className="w-3.5 h-3.5 text-[var(--label-tertiary)] absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search all llama-server flags by name, --flag, alias, env var, or description…"
          className="w-full bibulm-input pl-8 pr-7"
        />
        {search && (
          <button
            type="button"
            onClick={() => setSearch('')}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-[var(--label-tertiary)] hover:text-[var(--label-primary)] cursor-pointer"
            aria-label="Clear flag search"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {/* Category pills */}
      <div className="flex flex-wrap gap-1">
        <button
          type="button"
          onClick={() => setActiveCategory('all')}
          className={`px-2 py-1 rounded-md text-[10px] font-medium border transition-colors cursor-pointer ${
            activeCategory === 'all'
              ? 'bg-[var(--accent-soft)] border-[var(--accent-border)] text-[var(--accent-text)]'
              : 'bg-[var(--bg-surface)] border-[var(--border-hairline)] text-[var(--label-secondary)] hover:text-[var(--label-primary)]'
          }`}
        >
          All ({LLAMA_FLAGS_CATALOG.length})
        </button>
        {LLAMA_FLAG_CATEGORIES.map((cat) => {
          const count = categoryCounts.total[cat.id] ?? 0;
          const mod = categoryCounts.modified[cat.id] ?? 0;
          if (count === 0) return null;
          return (
            <button
              key={cat.id}
              type="button"
              onClick={() => setActiveCategory(cat.id)}
              className={`px-2 py-1 rounded-md text-[10px] font-medium border transition-colors flex items-center gap-1 cursor-pointer ${
                activeCategory === cat.id
                  ? 'bg-[var(--accent-soft)] border-[var(--accent-border)] text-[var(--accent-text)]'
                  : 'bg-[var(--bg-surface)] border-[var(--border-hairline)] text-[var(--label-secondary)] hover:text-[var(--label-primary)]'
              }`}
            >
              <span>{cat.label}</span>
              <span className="text-[var(--label-tertiary)] font-mono tabular-nums">{count}</span>
              {mod > 0 && (
                <span className="px-1 rounded bg-[var(--accent-soft)] text-[var(--accent-text)] font-mono tabular-nums">
                  {mod}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Flag rows */}
      <div className="max-h-80 overflow-y-auto rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-surface)] divide-y divide-[var(--border-hairline)]">
        {filteredSpecs.length === 0 ? (
          <div className="p-4 text-center text-[var(--label-tertiary)] text-[11px]">
            No llama.cpp flags match your current filter.
          </div>
        ) : (
          filteredSpecs.map((spec) => {
            const modified = isSpecModified(spec, cliFlags);
            const current = cliFlags[spec.flag];
            const inCurrentBuild =
              discoveredSet !== null &&
              discoveredSet.size > 0 &&
              (discoveredSet.has(spec.flag) ||
                (spec.negFlag ? discoveredSet.has(spec.negFlag) : false) ||
                spec.aliases.some((a) => discoveredSet.has(a)));
            return (
              <div
                key={spec.flag}
                className={`px-3 py-2 flex flex-col gap-1.5 ${
                  modified ? 'bg-[var(--accent-soft)]' : ''
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="font-medium text-[var(--label-primary)] text-[11px]">
                        {spec.label}
                      </span>
                      <code className="apple-badge apple-badge-accent font-mono text-[10px] py-0 px-1.5">
                        {spec.flag}
                      </code>
                      {inCurrentBuild && (
                        <span
                          className="apple-badge apple-badge-green font-mono text-[9px] py-0 px-1"
                          title="Verified present in the installed llama-server --help output"
                        >
                          in build
                        </span>
                      )}
                      {spec.negFlag && (
                        <code className="apple-badge font-mono text-[10px] py-0 px-1">
                          {spec.negFlag}
                        </code>
                      )}
                      {spec.aliases
                        .filter((a) => a !== spec.flag && a !== spec.negFlag)
                        .slice(0, 2)
                        .map((alias) => (
                          <code
                            key={alias}
                            className="text-[var(--label-tertiary)] font-mono text-[10px]"
                            title={`Alias: ${alias}`}
                          >
                            {alias}
                          </code>
                        ))}
                    </div>
                    <p className="text-[10px] text-[var(--label-secondary)] leading-snug mt-0.5">
                      {spec.description}
                    </p>
                  </div>

                  {/* Control */}
                  <div className="flex items-center gap-1.5 shrink-0">
                    {spec.control === 'bool' && (
                      <Toggle
                        size="sm"
                        checked={current?.kind === 'flag' ? current.value : false}
                        onChange={(v) =>
                          setFlagEntry(spec.flag, v ? { kind: 'flag', value: true } : null)
                        }
                      />
                    )}

                    {spec.control === 'tri_bool' && (
                      <select
                        value={
                          cliFlags[spec.flag]?.kind === 'flag' && cliFlags[spec.flag].value
                            ? 'on'
                            : spec.negFlag &&
                                cliFlags[spec.negFlag]?.kind === 'flag' &&
                                cliFlags[spec.negFlag].value
                              ? 'off'
                              : 'default'
                        }
                        onChange={(e) => {
                          const mode = e.target.value;
                          if (mode === 'on') {
                            setFlagEntry(spec.flag, { kind: 'flag', value: true }, spec.negFlag);
                          } else if (mode === 'off' && spec.negFlag) {
                            setFlagEntry(spec.negFlag, { kind: 'flag', value: true }, spec.flag);
                          } else {
                            clearSpec(spec);
                          }
                        }}
                        className="apple-select h-7 text-[11px]"
                      >
                        <option value="default">Default</option>
                        <option value="on">On ({spec.flag})</option>
                        {spec.negFlag && <option value="off">Off ({spec.negFlag})</option>}
                      </select>
                    )}

                    {spec.control === 'enum' && (
                      <select
                        value={current?.kind === 'value' ? current.value : ''}
                        onChange={(e) =>
                          setFlagEntry(
                            spec.flag,
                            e.target.value ? { kind: 'value', value: e.target.value } : null,
                          )
                        }
                        className="apple-select h-7 text-[11px] font-mono"
                      >
                        <option value="">Default</option>
                        {(spec.options ?? []).map((opt) => (
                          <option key={opt} value={opt}>
                            {opt}
                          </option>
                        ))}
                      </select>
                    )}

                    {(spec.control === 'int' || spec.control === 'float') && (
                      <input
                        type="number"
                        step={spec.control === 'float' ? 'any' : 1}
                        placeholder={spec.placeholder ?? 'Default'}
                        value={current?.kind === 'value' ? current.value : ''}
                        onChange={(e) =>
                          setFlagEntry(
                            spec.flag,
                            e.target.value === ''
                              ? null
                              : { kind: 'value', value: e.target.value },
                          )
                        }
                        className="w-28 bibulm-input h-7 text-right font-mono text-[11px] tabular-nums"
                      />
                    )}

                    {spec.control === 'string' && (
                      <input
                        type="text"
                        placeholder={spec.placeholder ?? 'Default'}
                        value={current?.kind === 'value' ? current.value : ''}
                        onChange={(e) =>
                          setFlagEntry(
                            spec.flag,
                            e.target.value === ''
                              ? null
                              : { kind: 'value', value: e.target.value },
                          )
                        }
                        className="w-44 bibulm-input h-7 font-mono text-[11px]"
                      />
                    )}

                    {spec.control === 'path' && (
                      <div className="flex items-center gap-1">
                        <input
                          type="text"
                          placeholder={spec.placeholder ?? 'Path'}
                          value={current?.kind === 'value' ? current.value : ''}
                          onChange={(e) =>
                            setFlagEntry(
                              spec.flag,
                              e.target.value === ''
                                ? null
                                : { kind: 'value', value: e.target.value },
                            )
                          }
                          className="w-36 bibulm-input h-7 font-mono text-[11px]"
                        />
                        <button
                          type="button"
                          onClick={() => handleBrowsePath(spec)}
                          className="bibulm-btn-secondary h-7 px-2"
                          title="Browse…"
                          aria-label="Browse path"
                        >
                          <FolderOpen className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    )}

                    {spec.control === 'multi_arg' && (
                      <div className="flex items-center gap-1">
                        {(spec.multiArgLabels ?? ['ARG1', 'ARG2']).map((lbl, idx) => {
                          const vals =
                            current?.kind === 'multi_arg'
                              ? current.value
                              : Array((spec.multiArgLabels ?? ['ARG1', 'ARG2']).length).fill('');
                          return (
                            <input
                              key={lbl}
                              type="text"
                              placeholder={lbl}
                              value={vals[idx] ?? ''}
                              onChange={(e) => {
                                const nextVals = [...vals];
                                nextVals[idx] = e.target.value;
                                if (nextVals.every((s) => s.trim() === '')) {
                                  setFlagEntry(spec.flag, null);
                                } else {
                                  setFlagEntry(spec.flag, { kind: 'multi_arg', value: nextVals });
                                }
                              }}
                              className="w-16 bibulm-input h-7 font-mono text-[11px] text-right tabular-nums"
                            />
                          );
                        })}
                      </div>
                    )}

                    {spec.control === 'repeatable' && (
                      <button
                        type="button"
                        onClick={() => {
                          const prev = current?.kind === 'repeatable' ? current.value : [];
                          setFlagEntry(spec.flag, { kind: 'repeatable', value: [...prev, ''] });
                        }}
                        className="bibulm-btn-secondary h-7 px-2 text-[11px]"
                      >
                        <Plus className="w-3 h-3" />
                        <span>Add</span>
                      </button>
                    )}

                    {modified && (
                      <button
                        type="button"
                        onClick={() => clearSpec(spec)}
                        className="apple-btn-ghost w-7 h-7 hover:text-[var(--signal-red)]"
                        title="Reset to default"
                        aria-label="Reset flag to default"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>

                {/* Repeatable items editor */}
                {spec.control === 'repeatable' &&
                  current?.kind === 'repeatable' &&
                  current.value.length > 0 && (
                    <div className="space-y-1 pl-2 pt-0.5">
                      {current.value.map((item, idx) => (
                        <div key={idx} className="flex items-center gap-1.5">
                          <input
                            type="text"
                            value={item}
                            placeholder={spec.placeholder ?? 'value'}
                            onChange={(e) => {
                              const nextVals = [...current.value];
                              nextVals[idx] = e.target.value;
                              setFlagEntry(spec.flag, { kind: 'repeatable', value: nextVals });
                            }}
                            className="flex-1 bibulm-input h-7 font-mono text-[11px]"
                          />
                          <button
                            type="button"
                            onClick={() => {
                              const nextVals = current.value.filter((_, i) => i !== idx);
                              setFlagEntry(
                                spec.flag,
                                nextVals.length > 0
                                  ? { kind: 'repeatable', value: nextVals }
                                  : null,
                              );
                            }}
                            className="apple-btn-ghost w-7 h-7 hover:text-[var(--signal-red)]"
                            aria-label="Remove item"
                          >
                            <X className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
              </div>
            );
          })
        )}
      </div>

      {uncatalogedFlags.length > 0 && !modifiedOnly && (
        <div className="apple-card-inset p-3 space-y-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[11px] font-medium text-[var(--label-primary)]">
              Additional flags detected in installed{' '}
              <code className="text-[var(--signal-green)]">llama-server --help</code> (
              {uncatalogedFlags.length})
            </span>
            <span className="text-[10px] text-[var(--label-tertiary)]">
              Not yet in static catalog
            </span>
          </div>
          <div className="max-h-44 overflow-y-auto divide-y divide-[var(--border-hairline)] rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-surface)]">
            {uncatalogedFlags.map((df) => {
              const entry = cliFlags[df.flag];
              const isEnabled = Boolean(entry);
              return (
                <div
                  key={df.flag}
                  className="px-2.5 py-1.5 flex items-center justify-between gap-2"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <code className="apple-badge apple-badge-green font-mono text-[10px] py-0 px-1.5">
                        {df.flag}
                      </code>
                      {df.value_hint && (
                        <span className="text-[10px] font-mono text-[var(--label-secondary)]">
                          {df.value_hint}
                        </span>
                      )}
                      {df.aliases.slice(0, 2).map((alias) => (
                        <code
                          key={alias}
                          className="text-[10px] font-mono text-[var(--label-tertiary)]"
                        >
                          {alias}
                        </code>
                      ))}
                    </div>
                    {df.description && (
                      <p
                        className="text-[10px] text-[var(--label-tertiary)] truncate mt-0.5"
                        title={df.description}
                      >
                        {df.description}
                      </p>
                    )}
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    {df.value_hint ? (
                      <input
                        type="text"
                        placeholder={df.value_hint ?? 'value'}
                        value={entry?.kind === 'value' ? entry.value : ''}
                        onChange={(e) =>
                          setFlagEntry(
                            df.flag,
                            e.target.value === ''
                              ? null
                              : { kind: 'value', value: e.target.value },
                          )
                        }
                        className="w-28 bibulm-input h-6 font-mono text-[10px]"
                      />
                    ) : (
                      <Toggle
                        size="sm"
                        checked={entry?.kind === 'flag' ? entry.value : false}
                        onChange={(v) =>
                          setFlagEntry(df.flag, v ? { kind: 'flag', value: true } : null)
                        }
                      />
                    )}
                    {isEnabled && (
                      <button
                        type="button"
                        onClick={() => setFlagEntry(df.flag, null)}
                        className="apple-btn-ghost w-6 h-6 hover:text-[var(--signal-red)]"
                        title="Remove override"
                        aria-label="Remove override"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
