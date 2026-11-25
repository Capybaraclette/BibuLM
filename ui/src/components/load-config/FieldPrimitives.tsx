import React from 'react';
import type { KvQuantType } from '../../types';

export const KV_QUANT_OPTIONS: { value: KvQuantType | ''; label: string }[] = [
  { value: '', label: 'Default (F16)' },
  { value: 'f16', label: 'F16' },
  { value: 'bf16', label: 'BF16' },
  { value: 'f8_e4m3', label: 'F8_E4M3 (FP8)' },
  { value: 'q8_0', label: 'Q8_0 (Recommended)' },
  { value: 'q5_1', label: 'Q5_1' },
  { value: 'q5_0', label: 'Q5_0' },
  { value: 'q4_1', label: 'Q4_1' },
  { value: 'q4_0', label: 'Q4_0 (Low VRAM)' },
  { value: 'iq4_nl', label: 'IQ4_NL' },
  { value: 'f32', label: 'F32' },
];

export function fileNameOnly(path: string | null | undefined): string {
  if (!path) return '';
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

export const NumberField: React.FC<{
  label: string;
  value: number | null;
  placeholder?: string;
  onChange: (v: number | null) => void;
}> = ({ label, value, placeholder, onChange }) => (
  <div className="flex items-center justify-between gap-2">
    <span className="text-[var(--label-secondary)] truncate">{label}</span>
    <input
      type="number"
      placeholder={placeholder}
      value={value ?? ''}
      onChange={(e) => onChange(e.target.value === '' ? null : parseInt(e.target.value, 10))}
      className="w-24 bibulm-input h-7 text-right font-mono tabular-nums"
    />
  </div>
);

export const FloatField: React.FC<{
  label: string;
  value: number | null;
  placeholder?: string;
  onChange: (v: number | null) => void;
}> = ({ label, value, placeholder, onChange }) => (
  <div className="flex items-center justify-between gap-2">
    <span className="text-[var(--label-secondary)] truncate">{label}</span>
    <input
      type="number"
      step="any"
      placeholder={placeholder}
      value={value ?? ''}
      onChange={(e) => onChange(e.target.value === '' ? null : parseFloat(e.target.value))}
      className="w-24 bibulm-input h-7 text-right font-mono tabular-nums"
    />
  </div>
);
