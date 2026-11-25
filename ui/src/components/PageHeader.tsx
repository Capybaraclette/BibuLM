import React from 'react';

interface PageHeaderProps {
  icon: React.ReactNode;
  title: string;
  subtitle?: React.ReactNode;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  children?: React.ReactNode;
}

/**
 * Clean macOS view header shared across all BibuLM workspace views.
 */
export const PageHeader: React.FC<PageHeaderProps> = ({
  icon,
  title,
  subtitle,
  badge,
  actions,
  children,
}) => (
  <div className="px-6 py-4 border-b border-[var(--border-hairline)] bg-[var(--bg-canvas)] shrink-0 relative">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex items-center gap-3 min-w-0">
        <div className="w-8 h-8 rounded-[9px] bg-[var(--accent-soft)] border border-[var(--accent-border)] flex items-center justify-center text-[var(--accent-text)] shrink-0">
          {icon}
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="text-[18px] leading-tight font-semibold tracking-[-0.02em] text-[var(--label-primary)] truncate">
              {title}
            </h1>
            {badge}
          </div>
          {subtitle && (
            <p className="text-[12px] text-[var(--label-secondary)] truncate mt-0.5">{subtitle}</p>
          )}
        </div>
      </div>
      {actions && <div className="flex items-center gap-2 flex-wrap">{actions}</div>}
    </div>
    {children}
  </div>
);
