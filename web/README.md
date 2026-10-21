# PlateSight (web)

The in-browser front end: React + Vite + Tailwind + Motion, with the detector running in a
Web Worker on ONNX Runtime Web (multi-threaded WebAssembly).

```bash
npm install
npm run dev       # http://localhost:5173
npm run build && npm run preview   # production build, served with the same COOP/COEP + CSP headers as Vercel
```

- `public/model/` holds the ONNX models and `models.json`, written by `training/scripts/export.py`
  and `optimize_model.py`.
- `public/samples/` holds test-split images (never trained on) with their ground-truth boxes.
- `vercel.json` sends cross-origin isolation headers (needed for WebAssembly threads), a strict
  Content-Security-Policy, and cache rules.
