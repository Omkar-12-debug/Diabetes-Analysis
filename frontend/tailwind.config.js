/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        clinical: {
          dark: "#0a0f1d",
          panel: "#131b2e",
          card: "#1a243b",
          border: "#263554",
          low: {
            DEFAULT: "#10b981",
            bg: "rgba(16, 185, 129, 0.12)",
            glow: "rgba(16, 185, 129, 0.3)",
          },
          mod: {
            DEFAULT: "#f59e0b",
            bg: "rgba(245, 158, 11, 0.12)",
            glow: "rgba(245, 158, 11, 0.3)",
          },
          high: {
            DEFAULT: "#f43f5e",
            bg: "rgba(244, 63, 94, 0.12)",
            glow: "rgba(244, 63, 94, 0.3)",
          },
          cyan: "#06b6d4",
          indigo: "#6366f1",
        },
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "monospace"],
      },
    },
  },
  plugins: [],
};
