import React, { useEffect, useState } from 'react';
import { getCurrentWindow } from '@tauri-apps/api/window';

export function useWindowFocus(): boolean {
  const [focused, setFocused] = useState(true);

  useEffect(() => {
    let mounted = true;
    const unsubs: (() => void)[] = [];

    const onWinFocus = () => setFocused(true);
    const onWinBlur = () => setFocused(false);
    window.addEventListener('focus', onWinFocus);
    window.addEventListener('blur', onWinBlur);

    try {
      const appWindow = getCurrentWindow();
      appWindow
        .onFocusChanged(({ payload }) => {
          if (mounted) setFocused(payload);
        })
        .then((fn) => unsubs.push(fn))
        .catch(() => {});
    } catch {
      // Running in plain browser preview without Tauri runtime
    }

    return () => {
      mounted = false;
      window.removeEventListener('focus', onWinFocus);
      window.removeEventListener('blur', onWinBlur);
      unsubs.forEach((u) => u());
    };
  }, []);

  return focused;
}

export function startWindowDrag(e: React.MouseEvent<HTMLElement>): void {
  if (e.button === 0 && e.detail === 1) {
    try {
      void getCurrentWindow().startDragging();
    } catch {
      // ignore outside Tauri
    }
  }
}

export function toggleWindowMaximize(): void {
  try {
    void getCurrentWindow().toggleMaximize();
  } catch {
    // ignore outside Tauri
  }
}

/**
 * Unobtrusive desktop window controls integrated into the unified toolbar.
 */
export const WindowControls: React.FC<{ focused?: boolean }> = ({ focused = true }) => {
  const [maximized, setMaximized] = useState(false);

  useEffect(() => {
    let mounted = true;
    const unsubs: (() => void)[] = [];

    try {
      const appWindow = getCurrentWindow();
      appWindow
        .isMaximized()
        .then((val) => {
          if (mounted) setMaximized(val);
        })
        .catch(() => {});

      appWindow
        .onResized(async () => {
          try {
            const val = await appWindow.isMaximized();
            if (mounted) setMaximized(val);
          } catch {
            // ignore outside Tauri
          }
        })
        .then((fn) => unsubs.push(fn))
        .catch(() => {});
    } catch {
      // Running in plain browser preview without Tauri runtime
    }

    return () => {
      mounted = false;
      unsubs.forEach((u) => u());
    };
  }, []);

  const handleMinimize = () => {
    try {
      void getCurrentWindow().minimize();
    } catch {
      // ignore outside Tauri
    }
  };

  const handleClose = () => {
    try {
      void getCurrentWindow().close();
    } catch {
      // ignore outside Tauri
    }
  };

  return (
    <div
      className={`flex items-center gap-0.5 transition-opacity duration-150 ${
        focused ? 'opacity-100' : 'opacity-60'
      }`}
      onMouseDown={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
    >
      <button
        type="button"
        onClick={handleMinimize}
        aria-label="Minimize"
        title="Minimize"
        className="w-8 h-7 rounded-md flex items-center justify-center text-[var(--label-secondary)] hover:text-[var(--label-primary)] hover:bg-[var(--bg-control-hover)] transition-colors cursor-pointer"
      >
        <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
          <path d="M1.5 5h7" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        </svg>
      </button>

      <button
        type="button"
        onClick={toggleWindowMaximize}
        aria-label={maximized ? 'Restore' : 'Maximize'}
        title={maximized ? 'Restore' : 'Maximize'}
        className="w-8 h-7 rounded-md flex items-center justify-center text-[var(--label-secondary)] hover:text-[var(--label-primary)] hover:bg-[var(--bg-control-hover)] transition-colors cursor-pointer"
      >
        {maximized ? (
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">
            <path d="M3 1.5h5.5V7" stroke="currentColor" strokeWidth="1.15" strokeLinecap="round" />
            <rect
              x="1.5"
              y="3"
              width="5.5"
              height="5.5"
              rx="1"
              stroke="currentColor"
              strokeWidth="1.15"
            />
          </svg>
        ) : (
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">
            <rect
              x="1.5"
              y="1.5"
              width="7"
              height="7"
              rx="1.2"
              stroke="currentColor"
              strokeWidth="1.15"
            />
          </svg>
        )}
      </button>

      <button
        type="button"
        onClick={handleClose}
        aria-label="Close"
        title="Close"
        className="w-8 h-7 rounded-md flex items-center justify-center text-[var(--label-secondary)] hover:bg-[#e81123] hover:text-white active:bg-[#c50f1f] transition-colors cursor-pointer"
      >
        <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
          <path
            d="M2 2l6 6M8 2l-6 6"
            stroke="currentColor"
            strokeWidth="1.2"
            strokeLinecap="round"
          />
        </svg>
      </button>
    </div>
  );
};
