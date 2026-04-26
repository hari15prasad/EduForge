export const theme = {
  colors: {
    background: {
      base: "#0a0e1a",
      surface: "#0f1629",
      elevated: "#151e38",
      overlay: "#1a2540",
    },
    slate: {
      50: "#f0f4ff",
      100: "#dde6ff",
      200: "#bcceff",
      300: "#8aaeff",
      400: "#567aee",
      500: "#3a56d4",
      600: "#2b3fab",
      700: "#1f2e7a",
      800: "#162055",
      900: "#0d1533",
    },
    indigo: {
      50: "#eef2ff",
      100: "#e0e7ff",
      200: "#c7d2fe",
      300: "#a5b4fc",
      400: "#818cf8",
      500: "#6366f1",
      600: "#4f46e5",
      700: "#4338ca",
      800: "#3730a3",
      900: "#312e81",
    },
    accent: {
      cyan: "#22d3ee",
      violet: "#8b5cf6",
      emerald: "#10b981",
      amber: "#f59e0b",
      rose: "#f43f5e",
    },
    status: {
      online: "#10b981",
      training: "#f59e0b",
      evaluating: "#6366f1",
      error: "#f43f5e",
    },
    text: {
      primary: "#e8eeff",
      secondary: "#8899cc",
      muted: "#4a5580",
      inverse: "#0a0e1a",
    },
    border: {
      default: "#1e2d50",
      subtle: "#131d38",
      accent: "#4338ca",
      glow: "rgba(99,102,241,0.4)",
    },
  },

  typography: {
    fonts: {
      display: "'Space Mono', 'Courier New', monospace",
      body: "'DM Sans', system-ui, sans-serif",
      mono: "'JetBrains Mono', 'Fira Code', monospace",
    },
    sizes: {
      xs: "0.75rem",
      sm: "0.875rem",
      base: "1rem",
      lg: "1.125rem",
      xl: "1.25rem",
      "2xl": "1.5rem",
      "3xl": "1.875rem",
      "4xl": "2.25rem",
    },
  },

  effects: {
    glowIndigo: "0 0 20px rgba(99,102,241,0.35), 0 0 40px rgba(99,102,241,0.15)",
    glowCyan: "0 0 20px rgba(34,211,238,0.35), 0 0 40px rgba(34,211,238,0.15)",
    glowEmerald: "0 0 16px rgba(16,185,129,0.4)",
    cardShadow: "0 4px 32px rgba(0,0,0,0.5), 0 1px 0 rgba(255,255,255,0.04) inset",
    borderGlow: "0 0 0 1px rgba(99,102,241,0.4)",
  },

  spacing: {
    sidebar: {
      expanded: "280px",
      collapsed: "64px",
    },
    header: "56px",
  },

  animation: {
    fast: "120ms ease",
    base: "220ms ease",
    slow: "400ms cubic-bezier(0.4,0,0.2,1)",
    pulse: "2s cubic-bezier(0.4,0,0.6,1) infinite",
  },
} as const;

export type Theme = typeof theme;

// CSS variable map — inject into :root
export const cssVars = `
  --bg-base: ${theme.colors.background.base};
  --bg-surface: ${theme.colors.background.surface};
  --bg-elevated: ${theme.colors.background.elevated};
  --bg-overlay: ${theme.colors.background.overlay};

  --indigo-400: ${theme.colors.indigo[400]};
  --indigo-500: ${theme.colors.indigo[500]};
  --indigo-600: ${theme.colors.indigo[600]};

  --accent-cyan: ${theme.colors.accent.cyan};
  --accent-violet: ${theme.colors.accent.violet};
  --accent-emerald: ${theme.colors.accent.emerald};
  --accent-amber: ${theme.colors.accent.amber};
  --accent-rose: ${theme.colors.accent.rose};

  --status-online: ${theme.colors.status.online};
  --status-training: ${theme.colors.status.training};
  --status-evaluating: ${theme.colors.status.evaluating};

  --text-primary: ${theme.colors.text.primary};
  --text-secondary: ${theme.colors.text.secondary};
  --text-muted: ${theme.colors.text.muted};

  --border-default: ${theme.colors.border.default};
  --border-subtle: ${theme.colors.border.subtle};
  --border-accent: ${theme.colors.border.accent};

  --glow-indigo: ${theme.effects.glowIndigo};
  --glow-cyan: ${theme.effects.glowCyan};
  --glow-emerald: ${theme.effects.glowEmerald};

  --font-display: ${theme.typography.fonts.display};
  --font-body: ${theme.typography.fonts.body};
  --font-mono: ${theme.typography.fonts.mono};

  --sidebar-expanded: ${theme.spacing.sidebar.expanded};
  --sidebar-collapsed: ${theme.spacing.sidebar.collapsed};
  --header-height: ${theme.spacing.header};
`;
