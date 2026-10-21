import '@fontsource-variable/inter'
import '@fontsource-variable/jetbrains-mono'
import { LazyMotion, MotionConfig } from 'motion/react'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'

// Motion's animation engine arrives as its own chunk instead of in the main bundle.
const features = () => import('./motion-features').then((mod) => mod.default)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {/* strict: a stray `motion.*` component (which bundles the full engine) throws in development. */}
    <LazyMotion features={features} strict>
      {/* "user": every animation honours the visitor's reduced-motion setting. */}
      <MotionConfig reducedMotion="user" transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}>
        <App />
      </MotionConfig>
    </LazyMotion>
  </StrictMode>,
)
