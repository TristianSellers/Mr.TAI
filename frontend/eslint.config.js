// eslint.config.js
import js from "@eslint/js";
import globals from "globals";
import react from "eslint-plugin-react";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";

export default [
  // Ignore build artifacts
  { ignores: ["dist/**", "node_modules/**"] },

  // Base JS rules
  js.configs.recommended,

  // Frontend (browser) files
  {
    files: ["**/*.{js,jsx}"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: { ...globals.browser },
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
    },
    plugins: {
      react,
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      // React/JSX usage detection so ESLint doesn’t flag <App /> as unused
      "react/jsx-uses-react": "error",
      "react/jsx-uses-vars": "error",

      // Modern React doesn’t need React in scope
      "react/react-in-jsx-scope": "off",

      // Hooks best practices
      ...reactHooks.configs.recommended.rules,

      // Helpful in dev with Vite + React
      "react-refresh/only-export-components": "warn",
    },
    settings: {
      react: { version: "detect" },
    },
  },

  // Node-specific files (Vite config, etc.)
  {
    files: ["vite.config.js", "eslint.config.js"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: { ...globals.node },
    },
  },
];
