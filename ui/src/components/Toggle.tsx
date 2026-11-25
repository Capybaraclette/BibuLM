import React from 'react';

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  color?: 'indigo' | 'blue' | 'green';
  disabled?: boolean;
  size?: 'sm' | 'md';
}

export const Toggle: React.FC<ToggleProps> = ({
  checked,
  onChange,
  color = 'blue',
  disabled = false,
  size = 'md',
}) => {
  const activeBg =
    color === 'green'
      ? 'bg-[var(--signal-green-fill)]'
      : 'bg-[var(--accent)]';
  const trackCls = size === 'sm' ? 'w-[34px] h-[19px]' : 'w-[42px] h-[23px]';
  const knobCls = size === 'sm' ? 'w-[15px] h-[15px]' : 'w-[19px] h-[19px]';
  const shiftCls = size === 'sm' ? 'translate-x-[15px]' : 'translate-x-[19px]';

  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => !disabled && onChange(!checked)}
      className={`relative inline-flex items-center shrink-0 rounded-full p-[2px] transition-colors duration-150 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${trackCls} ${
        checked ? activeBg : 'bg-[var(--border-strong)]'
      }`}
    >
      <span
        className={`inline-block rounded-full bg-white shadow-sm transform transition-transform duration-150 ${knobCls} ${
          checked ? shiftCls : 'translate-x-0'
        }`}
      />
    </button>
  );
};
