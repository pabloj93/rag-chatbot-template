import typography from "@tailwindcss/typography";

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: {} },
  // @tailwindcss/typography provides the `prose` class — gives sensible
  // defaults for rendered markdown (headings, lists, code, links).
  plugins: [typography],
};
