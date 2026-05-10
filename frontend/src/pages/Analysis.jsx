import { useEffect, useState, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { api } from '../api'

const SHOT_COLORS = {
  Forehand: '#C8E000',
  Backhand: '#C1440E',
  Serve:    '#1B4332',
  Volley:   '#2D6A4F',
  Smash:    '#888880',
  Return:   '#2D6A4F',
}

function formatTime(t) {
  const m = Math.floor(t / 60)
  const s = (t % 60).toFixed(1).padStart(4, '0')
  return `${m}:${s}`
}

export default function Analysis() {
  const { jobId }   = useParams()
  const navigate    = useNavigate()
  const [status, setStatus]     = useState('processing')
  const [progress, setProgress] = useState(0)
  const [liveShots, setLiveShots] = useState([])
  const [tally, setTally]       = useState({})
  const pollRef = useRef(null)

  // Keep HF Space alive
  useEffect(() => {
    const ping = setInterval(() => fetch((import.meta.env.VITE_API_URL ?? '') + '/').catch(() => {}), 25000)
    return () => clearInterval(ping)
  }, [])

  useEffect(() => {
    pollRef.current = setInterval(async () => {
      try {
        const res  = await api.job(jobId)
        const data = await res.json()
        setProgress(data.progress ?? 0)
        setStatus(data.status)

        if (data.shots) {
          setLiveShots(data.shots.slice(-8))
          const t = {}
          data.shots.forEach((s) => { t[s.type] = (t[s.type] ?? 0) + 1 })
          setTally(t)
        }

        if (data.status === 'complete') {
          clearInterval(pollRef.current)
          setTimeout(() => navigate(`/results/${jobId}`), 800)
        }
      } catch {
        // silently retry
      }
    }, 1000)

    return () => clearInterval(pollRef.current)
  }, [jobId, navigate])

  return (
    <div className="min-h-screen bg-[#FAFAF7] flex flex-col">
      <nav className="flex items-center justify-between px-10 py-6 border-b border-[#1B4332]/10">
        <a
          href="/"
          className="text-[#1B4332]/60 text-sm font-light tracking-widest uppercase hover:text-[#1B4332] transition-colors duration-300"
          style={{ letterSpacing: '0.3em' }}
        >
          Raquette
        </a>
        <span className="text-xs text-[#888880] font-light tracking-wide">
          {status === 'complete' ? 'Analysis complete' : 'Identifying shots…'}
        </span>
      </nav>

      <div className="flex-1 max-w-2xl mx-auto w-full px-8 py-16 flex flex-col gap-12">

        {/* Progress */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <p className="text-xs text-[#888880] font-light tracking-widest uppercase">Progress</p>
            <span className="text-xs text-[#1B4332] font-light tabular-nums">{progress}%</span>
          </div>
          <div className="w-full h-px bg-[#1B4332]/10 relative">
            <motion.div
              className="absolute top-1/2 left-0 -translate-y-1/2 h-0.5 bg-[#1B4332]"
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
          <p className="mt-2 text-xs text-[#888880]/60 font-light">
            {progress < 2 ? 'Loading models…' : progress < 5 ? 'Models ready, scanning frames…' : 'Running YOLOv8 · MediaPipe · Classifier'}
          </p>
        </div>

        {/* Live shot feed */}
        <div>
          <p className="text-xs text-[#888880] font-light tracking-widest uppercase mb-4">Live shots</p>
          <div className="space-y-2 min-h-[200px]">
            <AnimatePresence>
              {liveShots.map((shot, i) => (
                <motion.div
                  key={`${shot.type}-${shot.timestamp}-${i}`}
                  initial={{ opacity: 0, x: 12 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.4 }}
                  className="flex items-center justify-between py-2 border-b border-[#1B4332]/8"
                >
                  <div className="flex items-center gap-3">
                    <div
                      className="w-1.5 h-1.5 rounded-full shrink-0"
                      style={{ backgroundColor: SHOT_COLORS[shot.type] ?? '#888880' }}
                    />
                    <span className="text-sm text-[#1B4332] font-light">{shot.type}</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-xs text-[#1B4332] font-light">{shot.player_name}</span>
                    <span className="text-xs text-[#888880] font-light tabular-nums">{formatTime(shot.timestamp)}</span>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            {liveShots.length === 0 && (
              <p className="text-xs text-[#888880]/60 font-light">Waiting for first shot…</p>
            )}
          </div>
        </div>

        {/* Running tally */}
        {Object.keys(tally).length > 0 && (
          <div className="pt-6 border-t border-[#1B4332]/10 space-y-3">
            <p className="text-xs text-[#888880] font-light tracking-widest uppercase mb-4">Shot tally</p>
            {Object.entries(tally).map(([type, count]) => (
              <div key={type} className="flex items-baseline justify-between">
                <span className="text-xs text-[#888880] font-light tracking-wide">{type}</span>
                <span className="text-sm text-[#1B4332] font-light tabular-nums">{count}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
