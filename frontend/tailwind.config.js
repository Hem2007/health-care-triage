/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        triage: {
          red: '#dc2626',
          yellow: '#eab308',
          green: '#16a34a',
          black: '#111827',
        }
      }
    },
  },
  plugins: [],
}
