import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { readFileSync, writeFileSync } from 'node:fs'
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
 * Adds a <link rel="preload"> for the first model (named in models.json) to the built
 * index.html, so it starts downloading while the HTML is parsed. The runtime isn't
 * preloaded: which build to fetch (WebGPU or CPU) depends on the device, so
 * src/lib/runtime.ts starts that download as soon as the app's JavaScript runs.
 */
function preloadDetector(): Plugin {
  return {
    name: 'preload-detector',
    apply: 'build',
    closeBundle() {
      const dist = new URL('./dist/', import.meta.url)
      const manifest = JSON.parse(readFileSync(new URL('model/models.json', dist), 'utf8')) as { variants: Record<string, { file: string }> }
      const first = (manifest.variants.fast ?? manifest.variants.accurate)?.file
      if (!first) return
      const link = `<link rel="preload" href="/model/${first}" as="fetch" crossorigin="anonymous" fetchpriority="low" />`
      const htmlPath = new URL('index.html', dist)
      writeFileSync(htmlPath, readFileSync(htmlPath, 'utf8').replace('</head>', `    ${link}\n  </head>`))
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
