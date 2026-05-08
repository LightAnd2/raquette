import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, ChevronDown } from 'lucide-react'
import { api } from '../api'
import SampleAnalysis from '../components/SampleAnalysis'

const DEFAULT_NAMES = ['', '', '', '']

export default function Landing() {
  const [dragging, setDragging]     = useState(false)
  const [uploading, setUploading]   = useState(false)
  const [error, setError]           = useState(null)
  const [mode, setMode]             = useState('singles')
  const [playerNames, setPlayerNames] = useState(DEFAULT_NAMES)
  const navigate = useNavigate()

  const nPlayers = mode === 'doubles' ? 4 : 2

  const setName = (i, val) => {
    setPlayerNames((prev) => {
      const next = [...prev]
      next[i] = val
      return next
    })
  }

  const handleFile = useCallback(async (file) => {
    if (!file) return
    if (!file.type.startsWith('video/')) {
      setError('Please upload a video file.')
      return
    }
    setError(null)
    setUploading(true)

    const names = playerNames.slice(0, nPlayers)

    try {
      const res = await api.upload(file, mode, names)
      if (!res.ok) throw new Error('Upload failed')
      const { job_id } = await res.json()
      navigate(`/analysis/${job_id}`)
    } catch {
      setError('Upload failed — make sure the backend is running.')
      setUploading(false)
    }
  }, [navigate, mode, playerNames, nPlayers])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [handleFile])

  const useSampleVideo = useCallback(async () => {
    setError(null)
    setUploading(true)
    try {
      const res  = await fetch('/demo.mp4')
      const blob = await res.blob()
      await handleFile(new File([blob], 'demo.mp4', { type: 'video/mp4' }))
    } catch {
      setError('Could not load sample video.')
      setUploading(false)
    }
  }, [handleFile])

  return (
    <div className="min-h-screen bg-[#FAFAF7] flex flex-col">
      {/* Nav */}
      <nav className="flex items-center justify-between px-10 py-6">
        <span
          className="text-[#1B4332] text-xl tracking-widest uppercase font-light"
          style={{ fontFamily: '"DM Sans", system-ui, sans-serif', letterSpacing: '0.3em' }}
        >
          Raquette
        </span>
        <div className="flex gap-8 text-sm text-[#888880] font-light tracking-wide whitespace-nowrap">
          <a href="#how" className="hover:text-[#1B4332] transition-colors duration-300">How it works</a>
          <a href="#sample" className="hover:text-[#1B4332] transition-colors duration-300">Demo</a>
        </div>
      </nav>

      {/* Hero */}
      <section className="flex flex-col items-center justify-center flex-1 px-6 pt-16 pb-24 text-center">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        >
          <h1
            className="text-[#1B4332] leading-none mb-6"
            style={{
              fontFamily: '"Playfair Display", Georgia, serif',
              fontSize: 'clamp(3.5rem, 8vw, 7rem)',
              fontWeight: 600,
              letterSpacing: '-0.02em',
            }}
          >
            Raquette
          </h1>
          <p
            className="text-[#888880] font-light mb-16 tracking-wide"
            style={{ fontFamily: '"DM Sans", system-ui, sans-serif', fontSize: '1.1rem', letterSpacing: '0.08em' }}
          >
            Every shot. Identified.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
          className="w-full max-w-xl"
        >
          {/* Mode toggle */}
          <div className="flex gap-px mb-6 border border-[#1B4332]/20">
            {['singles', 'doubles'].map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className="flex-1 py-2.5 text-xs font-light tracking-widest uppercase transition-all duration-300"
                style={{
                  backgroundColor: mode === m ? '#1B4332' : 'transparent',
                  color: mode === m ? '#FAFAF7' : '#888880',
                }}
              >
                {m}
              </button>
            ))}
          </div>

          {/* Player name inputs */}
          <div className={`grid gap-3 mb-6 ${nPlayers === 4 ? 'grid-cols-2' : 'grid-cols-2'}`}>
            {Array.from({ length: nPlayers }).map((_, i) => (
              <input
                key={i}
                type="text"
                maxLength={20}
                placeholder={`Player ${i + 1}`}
                value={playerNames[i]}
                onChange={(e) => setName(i, e.target.value)}
                className="bg-transparent border-b border-[#1B4332]/20 py-2 px-0 text-sm text-[#1B4332] font-light placeholder-[#888880]/50 focus:outline-none focus:border-[#1B4332]/60 transition-colors"
              />
            ))}
          </div>

          {/* Upload zone */}
          <label
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={`
              flex flex-col items-center gap-4 px-12 py-14 cursor-pointer
              border transition-all duration-500
              ${dragging
                ? 'border-[#1B4332] bg-[#1B4332]/5'
                : 'border-[#1B4332]/30 hover:border-[#1B4332]/60 hover:bg-[#1B4332]/[0.02]'
              }
            `}
          >
            <input type="file" accept="video/*" className="hidden" onChange={(e) => handleFile(e.target.files[0])} disabled={uploading} />
            <AnimatePresence mode="wait">
              {uploading ? (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center gap-3"
                >
                  <div className="w-5 h-5 border border-[#1B4332] border-t-transparent rounded-full animate-spin" />
                  <span className="text-sm text-[#1B4332] font-light tracking-widest uppercase">Uploading</span>
                </motion.div>
              ) : (
                <motion.div
                  key="idle"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center gap-3"
                >
                  <Upload size={20} strokeWidth={1.25} className="text-[#1B4332]/60" />
                  <div className="text-center">
                    <p className="text-sm text-[#1B4332] font-light tracking-widest uppercase mb-1">Upload match footage</p>
                    <p className="text-xs text-[#888880] font-light">MP4, MOV, AVI — drag or click</p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </label>

          <div className="mt-4 text-center">
            <span className="text-xs text-[#888880] font-light">No video? </span>
            <button
              onClick={useSampleVideo}
              disabled={uploading}
              className="text-xs text-[#1B4332] font-light underline underline-offset-2 hover:text-[#2D6A4F] transition-colors disabled:opacity-40"
            >
              Try the sample clip
            </button>
          </div>

          {error && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-3 text-xs text-[#C1440E] font-light tracking-wide text-center"
            >
              {error}
            </motion.p>
          )}
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2, duration: 0.8 }}
          className="mt-20 flex flex-col items-center gap-2 text-[#888880]"
        >
          <span className="text-xs font-light tracking-widest uppercase">See it in action</span>
          <ChevronDown size={14} strokeWidth={1} className="animate-bounce" />
        </motion.div>
      </section>

      <div className="w-full h-px bg-[#1B4332]/10" />

      {/* Demo */}
      <section id="sample" className="py-24 px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          className="text-center mb-14"
        >
          <p className="text-xs text-[#888880] font-light tracking-widest uppercase mb-3">Live demo</p>
          <h2
            className="text-[#1B4332]"
            style={{ fontFamily: '"Playfair Display", Georgia, serif', fontSize: '2rem', fontWeight: 600 }}
          >
            A rally, fully identified
          </h2>
        </motion.div>
        <SampleAnalysis />
      </section>

      <div className="w-full h-px bg-[#1B4332]/10" />

      {/* How it works */}
      <section id="how" className="py-24 px-6 max-w-5xl mx-auto w-full">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          className="text-center mb-16"
        >
          <p className="text-xs text-[#888880] font-light tracking-widest uppercase mb-3">The pipeline</p>
          <h2
            className="text-[#1B4332]"
            style={{ fontFamily: '"Playfair Display", Georgia, serif', fontSize: '2rem', fontWeight: 600 }}
          >
            Three models. One pipeline.
          </h2>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-[#1B4332]/10">
          {[
            {
              step: '01',
              title: 'Player Detection',
              model: 'YOLOv8',
              body: 'Players are detected and tracked consistently across every frame using centroid re-identification — ball boys and spectators are filtered out automatically.',
            },
            {
              step: '02',
              title: 'Pose Estimation',
              model: 'MediaPipe',
              body: '33 body landmarks are extracted per player at each frame — shoulder rotation, hip alignment, wrist angle — the full biomechanical signature of every swing.',
            },
            {
              step: '03',
              title: 'Shot Classification',
              model: 'Temporal CNN',
              body: 'A 1D convolutional network slides over the pose sequence to distinguish forehand from backhand, serve from volley, slice from smash.',
            },
          ].map((item) => (
            <motion.div
              key={item.step}
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
              className="bg-[#FAFAF7] p-10"
            >
              <div className="flex items-start gap-6">
                <span
                  className="text-[#1B4332]/20 shrink-0"
                  style={{ fontFamily: '"Playfair Display", Georgia, serif', fontSize: '2.5rem', lineHeight: 1 }}
                >
                  {item.step}
                </span>
                <div>
                  <p className="text-xs text-[#888880] font-light tracking-widest uppercase mb-1">{item.model}</p>
                  <h3
                    className="text-[#1B4332] mb-3"
                    style={{ fontFamily: '"Playfair Display", Georgia, serif', fontSize: '1.15rem' }}
                  >
                    {item.title}
                  </h3>
                  <p className="text-sm text-[#888880] font-light leading-relaxed">{item.body}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      <div className="w-full h-px bg-[#1B4332]/10" />
      <footer className="py-8 px-10 flex items-center justify-between">
        <span className="text-[#1B4332]/60 text-sm font-light tracking-widest uppercase" style={{ letterSpacing: '0.25em' }}>
          Raquette
        </span>
        <p className="text-xs text-[#888880] font-light">Open source · Zero cost · Built for the love of tennis</p>
      </footer>
    </div>
  )
}
