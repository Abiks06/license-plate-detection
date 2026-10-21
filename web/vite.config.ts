import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { readFileSync } from 'node:fs'
import { defineConfig } from 'vite'

// Cross-origin isolation unlocks SharedArrayBuffer, which multi-threaded
// ONNX Runtime WebAssembly needs. vercel.json sends the same headers in production.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

// Same policy as vercel.json, so `vite preview` catches anything the CSP would block in production.
const vercel = JSON.parse(readFileSync(new URL('./vercel.json', import.meta.url), 'utf8')) as {
  headers: { source: string; headers: { key: string; value: string }[] }[]
}
const csp = vercel.headers[0].headers.find((h) => h.key === 'Content-Security-Policy')!.value

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { headers: isolation },
  preview: { headers: { ...isolation, 'Content-Security-Policy': csp } },
  worker: { format: 'es' },
  // ORT ships its own wasm loader; pre-bundling it breaks the asset URLs.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: {
    target: 'es2022',
    // Never inline fonts as data: URLs; the CSP only allows fonts from our own origin.
    assetsInlineLimit: (file) => (file.endsWith('.woff2') ? false : undefined),
  },
})
