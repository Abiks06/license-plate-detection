import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { defineConfig, type Plugin } from 'vite'

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

/**
 * Adds <link rel="preload"> tags for the WebAssembly runtime and the first model to the
 * built index.html, so both start downloading while the HTML is parsed instead of after
 * the app and worker boot. Runs after the build because the worker's hashed .wasm name
 * only exists once the worker bundle has been written.
 */
function preloadDetector(): Plugin {
  return {
    name: 'preload-detector',
    apply: 'build',
    closeBundle() {
      const dist = new URL('./dist/', import.meta.url)
      const wasm = readdirSync(new URL('assets/', dist)).find((f) => /^ort-wasm-simd-threaded.*\.wasm$/.test(f))
      const manifest = JSON.parse(readFileSync(new URL('model/models.json', dist), 'utf8')) as { variants: Record<string, { file: string }> }
      const first = (manifest.variants.fast ?? manifest.variants.accurate)?.file
      const links = [
        wasm && `<link rel="preload" href="/assets/${wasm}" as="fetch" type="application/wasm" crossorigin="anonymous" />`,
        first && `<link rel="preload" href="/model/${first}" as="fetch" crossorigin="anonymous" />`,
      ].filter(Boolean)
      const htmlPath = new URL('index.html', dist)
      writeFileSync(htmlPath, readFileSync(htmlPath, 'utf8').replace('</head>', `    ${links.join('\n    ')}\n  </head>`))
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), preloadDetector()],
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
