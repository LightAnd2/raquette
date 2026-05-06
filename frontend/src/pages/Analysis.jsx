import { useEffect, useState, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { api } from '../api'

const SHOT_COLORS = {
  Forehand: '#C8E000',
  Backhand: '#C1440E',
  Serve: '#1B4332',
  Volley: '#2D6A4F',
  Smash: '#1B4332',
  Slice: '#888880',
  Return: '#2D6A4F',
}

export default function Analysis() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const [status, setStatus] = useState('processing')
  const [progress, setProgress] = useState(0)
  const [currentFrame, setCurrentFrame] = useState(null)
  const [liveShots, setLiveShots] = useState([])
  const [stats, setStats] = useState({ rallyLength: 0, ballSpeed: 0, shotTally: {} })
  const pollRef = useRef(null)

  // Keep HF Space alive — ping every 25s so it doesn't sleep mid-job
  useEffect(() => {
    const ping = setInterval(() => api.job('ping').catch(() => {}), 25000)
    return () => clearInterval(ping)
  }, [])

  useEffect(() => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await api.job(jobId)
        const data = await res.json()
        setProgress(data.progress ?? 0)
        setStatus(data.status)

        if (data.current_frame) setCurrentFrame(data.current_frame)
        if (data.shots) {
          setLiveShots(data.shots.slice(-8))
          const tally = {}
          data.shots.forEach((s) => { tally[s.type] = (tally[s.type] ?? 0) + 1 })
          setStats({
            rallyLength: data.shots.length,
            ballSpeed: data.shots.at(-1)?.speed ?? 0,
            shotTally: tally,
          })
        }

        if (data.status === 'complete') {
          clearInterval(pollRef.current)
          setTimeout(() => navigate(`/results/${jobId}`), 800)
        }
      } catch (e) {
        // silently retry
      }
    }, 1000)

    return () => clearInterval(pollRef.current)
  }, [jobId, navigate])

  return (
    <div className="min-h-screen bg-[#FAFAF7] flex flex-col">
      {/* Nav */}
      <nav className="flex items-center justify-between px-10 py-6 border-b border-[#1B4332]/10">
        <a
          href="/"
          className="text-[#1B4332]/60 text-sm font-light tracking-widest uppercase hover:text-[#1B4332] transition-colors duration-300"
          style={{ letterSpacing: '0.3em' }}
        >
          Raquette
        </a>
        <span className="text-xs text-[#888880] font-light tracking-wide">
          {status === 'complete' ? 'Analysis complete' : 'Analysing footage…'}
        </span>
      </nav>

      <div className="flex-1 grid grid-cols-1 md:grid-cols-[1fr_360px] gap-px bg-[#1B4332]/10 m-8">
        {/* Video / frame panel */}
        <div className="bg-[#FAFAF7] p-8 flex flex-col">
          <FrameViewer frame={currentFrame} progress={progress} status={status} />
        </div>

        {/* Stats panel */}
        <div className="bg-[#FAFAF7] p-8 flex flex-col gap-8">
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
          </div>

          {/* Live shot feed */}
          <div>
            <p className="text-xs text-[#888880] font-light tracking-widest uppercase mb-4">Live shots</p>
            <div className="space-y-2 min-h-[200px]">
              <AnimatePresence>
                {liveShots.map((shot, i) => (
                  <motion.div
                    key={`${shot.type}-${i}`}
                    initial={{ opacity: 0, x: 12 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.4 }}
                    className="flex items-center justify-between py-2 border-b border-[#1B4332]/8"
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className="w-1.5 h-1.5 rounded-full"
                        style={{ backgroundColor: SHOT_COLORS[shot.type] ?? '#888880' }}
                      />
                      <span className="text-sm text-[#1B4332] font-light">{shot.type}</span>
                    </div>
                    <span className="text-xs text-[#888880] font-light tabular-nums">{shot.speed} km/h</span>
                  </motion.div>
                ))}
              </AnimatePresence>
              {liveShots.length === 0 && (
                <p className="text-xs text-[#888880]/60 font-light">Waiting for first shot…</p>
              )}
            </div>
          </div>

          {/* Running stats */}
          <div className="mt-auto pt-6 border-t border-[#1B4332]/10 space-y-4">
            <LiveStat label="Shots detected" value={stats.rallyLength} />
            <LiveStat label="Last ball speed" value={stats.ballSpeed ? `${stats.ballSpeed} km/h` : '—'} highlight />
            {Object.entries(stats.shotTally).map(([type, count]) => (
              <LiveStat key={type} label={type} value={count} />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function LiveStat({ label, value, highlight }) {
  return (
    <div className="flex items-baseline justify-between">
      <span className="text-xs text-[#888880] font-light tracking-wide">{label}</span>
      <motion.span
        key={String(value)}
        initial={{ opacity: 0.4 }}
        animate={{ opacity: 1 }}
        className="text-sm font-light tabular-nums"
        style={{ color: highlight ? '#C8E000' : '#1B4332' }}
      >
        {value}
      </motion.span>
    </div>
  )
}

function FrameViewer({ frame, progress, status }) {
  return (
    <div className="flex-1 flex flex-col">
      <div
        className="flex-1 border border-[#1B4332]/20 bg-[#E8EFE8]/30 flex items-center justify-center relative overflow-hidden"
        style={{ minHeight: 320 }}
      >
        {frame ? (
          <img src={`data:image/jpeg;base64,${frame}`} className="w-full h-full object-contain" alt="Analysis frame" />
        ) : (
          <div className="flex flex-col items-center gap-4 text-center p-8">
            <div className="w-6 h-6 border border-[#1B4332]/40 border-t-[#1B4332] rounded-full animate-spin" />
            <div>
              <p className="text-sm text-[#1B4332] font-light mb-1">Processing frames</p>
              <p className="text-xs text-[#888880] font-light">
                Running YOLOv8 · TrackNet · MediaPipe
              </p>
            </div>
          </div>
        )}

        {/* Overlay labels when frame present */}
        {frame && (
          <div className="absolute bottom-3 left-3 flex gap-2">
            {['YOLOv8', 'TrackNet', 'MediaPipe'].map((label) => (
              <span
                key={label}
                className="text-[9px] font-light tracking-widest uppercase px-2 py-1 bg-[#1B4332]/80 text-[#FAFAF7]"
              >
                {label}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="mt-4 text-xs text-[#888880] font-light tracking-wide">
        {status === 'complete'
          ? 'Analysis complete — redirecting to results…'
          : `Processing · ${progress}% complete`}
      </div>
    </div>
  )
}
