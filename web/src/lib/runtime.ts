import cpuWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import gpuWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';

export type Backend = 'webgpu' | 'wasm';
export type Runtime = { backend: Backend; wasm: ArrayBuffer };

type GpuAdapter = { isFallbackAdapter?: boolean; info?: { isFallbackAdapter?: boolean } } | null;
type GpuNavigator = { gpu?: { requestAdapter(): Promise<GpuAdapter> } };

/** True only for a hardware WebGPU adapter: software fallbacks are slower than WebAssembly. */
async function hasHardwareGpu() {
  try {
    // isFallbackAdapter is newer than the bundled WebGPU typings, hence the local type.
    const adapter = await (navigator as unknown as GpuNavigator).gpu?.requestAdapter();
    return !!adapter && !adapter.isFallbackAdapter && !adapter.info?.isFallbackAdapter;
  } catch {
    return false;
  }
}

async function fetchRuntime(backend: Backend): Promise<Runtime> {
  const res = await fetch(backend === 'webgpu' ? gpuWasmUrl : cpuWasmUrl);
  if (!res.ok) throw new Error(`runtime: ${res.status}`);
  return { backend, wasm: await res.arrayBuffer() };
}

const forced = new URLSearchParams(location.search).get('backend');

let gpuRuntime: Promise<Runtime | null> | null = null;

/**
 * The runtimes:
 * - cpu: the small CPU-only build, fetched as soon as this module is imported. It serves first.
 * - gpu(): the WebGPU build, only on devices with a hardware GPU. Fetched on demand (after the
 *   CPU worker is ready) so it never competes for bandwidth with what the first detection needs.
 *   Its first run compiles shaders (seconds on a first visit, cached by the browser afterwards),
 *   so it warms up in the background and takes over when ready.
 * `?backend=wasm` or `?backend=webgpu` forces a single runtime (for comparisons).
 */
export const runtimes = {
  cpu: forced === 'webgpu' ? null : fetchRuntime('wasm'),
  gpu(): Promise<Runtime | null> {
    gpuRuntime ??=
      forced === 'wasm'
        ? Promise.resolve(null)
        : forced === 'webgpu'
          ? fetchRuntime('webgpu') // its built-in CPU fallback covers devices without a GPU
          : hasHardwareGpu().then((ok) => (ok ? fetchRuntime('webgpu') : null));
    return gpuRuntime;
  },
};
