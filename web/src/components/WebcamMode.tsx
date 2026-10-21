import { m } from "motion/react";
import { Camera, CameraOff } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { Box, Timing } from '../lib/types';
import type { Detection } from '../lib/useDetector';
import { Overlay } from './Overlay';
import { Results } from './Results';

type VideoWithFrames = HTMLVideoElement & { requestVideoFrameCallback?: (cb: () => void) => number };

export function WebcamMode({
  ready,
  threshold,
  detect,
}: {
  ready: string | null;
  threshold: number;
  detect: (image: ImageBitmap, threshold: number) => Promise<Detection>;
}) {
  const video = useRef<VideoWithFrames>(null);
  const stream = useRef<MediaStream | null>(null);
  const running = useRef(false);
  const thresholdRef = useRef(threshold);
  const [on, setOn] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [size, setSize] = useState({ width: 1280, height: 720 });
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [timing, setTiming] = useState<Timing | null>(null);
  const [fps, setFps] = useState(0);

  useEffect(() => {
    thresholdRef.current = threshold;
  }, [threshold]);

  const stop = useCallback(() => {
    running.current = false;
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    setOn(false);
    setBoxes([]);
  }, []);

  // One detection at a time: the next frame is grabbed only after the previous result lands.
  const loop = useCallback(async () => {
    const v = video.current;
    let last = performance.now();
    let ema = 0;
    while (running.current && v) {
      if (v.readyState >= 2 && v.videoWidth) {
        try {
          const frame = await createImageBitmap(v);
          const res = await detect(frame, thresholdRef.current);
          if (!running.current) break;
          setBoxes(res.boxes);
          setTiming(res.timing);
          const now = performance.now();
          ema = ema ? ema * 0.85 + (1000 / (now - last)) * 0.15 : 1000 / (now - last);
          last = now;
          setFps(ema);
        } catch (e) {
          setError(e instanceof Error ? e.message : String(e));
          break;
        }
      }
      // Yield to the next video frame so we never spin faster than the camera.
      await new Promise<void>((r) => (v.requestVideoFrameCallback ? v.requestVideoFrameCallback(() => r()) : requestAnimationFrame(() => r())));
    }
  }, [detect]);

  const start = async () => {
    setError(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      stream.current = s;
      const v = video.current!;
      v.srcObject = s;
      await v.play();
      setSize({ width: v.videoWidth, height: v.videoHeight });
      setOn(true);
      running.current = true;
      loop();
    } catch (e) {
      setError(e instanceof DOMException && e.name === 'NotAllowedError' ? 'Camera access was blocked. Allow it in your browser’s site settings to use live mode.' : 'No camera is available on this device.');
    }
  };

  useEffect(() => stop, [stop]);

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
      <div>
        <div className="relative overflow-hidden rounded-2xl border border-line bg-black" style={{ aspectRatio: `${size.width} / ${size.height}`, maxHeight: '70vh' }}>
          <video ref={video} className="h-full w-full object-contain" playsInline muted />
          {on && <Overlay width={size.width} height={size.height} boxes={boxes} live />}
          {!on && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-6 text-center">
              <span className="grid size-14 place-items-center rounded-2xl border border-white/15 bg-white/5 text-accent">
                <Camera className="size-6" aria-hidden="true" />
              </span>
              <p className="max-w-sm text-sm text-white/70">Point your camera at a car. Frames are analysed on this device and never leave it.</p>
              <button type="button" onClick={start} disabled={!ready} className="btn btn-primary">
                <Camera className="size-4" aria-hidden="true" /> {ready ? 'Start camera' : 'Loading model…'}
              </button>
            </div>
          )}
          {on && (
            <m.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-black/60 px-3 py-1.5 font-mono text-xs text-white backdrop-blur">
              <span className="relative flex size-2">
                <span className="absolute inline-flex size-full animate-ping rounded-full bg-red-500 opacity-75 motion-reduce:hidden" />
                <span className="relative inline-flex size-2 rounded-full bg-red-500" />
              </span>
              LIVE · <span className="tabular-nums">{fps.toFixed(1)}</span> fps
            </m.div>
          )}
        </div>
        {error && <p className="mt-3 text-sm text-danger" role="alert">{error}</p>}
        {on && (
          <div className="mt-4">
            <button type="button" onClick={stop} className="btn">
              <CameraOff className="size-4" aria-hidden="true" /> Stop camera
            </button>
          </div>
        )}
      </div>
      <aside className="rounded-2xl border border-line bg-surface p-5">
        {on ? (
          <Results boxes={boxes} crops={[]} timing={timing} live />
        ) : (
          <div className="flex h-full min-h-48 flex-col justify-center text-sm text-muted">
            <span className="font-mono text-[11px] uppercase tracking-wider text-faint">Live mode</span>
            <p className="mt-2">Use the Fast model for the smoothest frame rate.</p>
          </div>
        )}
      </aside>
    </div>
  );
}
