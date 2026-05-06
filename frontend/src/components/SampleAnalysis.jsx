import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

const SHOTS = [
  { time: 0.5, type: 'Serve', player: 'P1', speed: 182, side: 'right' },
  { time: 2.1, type: 'Return', player: 'P2', speed: 94, side: 'left' },
  { time: 3.4, type: 'Forehand', player: 'P1', speed: 112, side: 'right' },
  { time: 4.8, type: 'Backhand', player: 'P2', speed: 87, side: 'left' },
  { time: 6.0, type: 'Forehand', player: 'P1', speed: 128, side: 'right' },
  { time: 7.2, type: 'Volley', player: 'P2', speed: 71, side: 'left' },
  { time: 8.1, type: 'Smash', player: 'P1', speed: 156, side: 'right' },
]

const SHOT_COLORS = {
  Serve: '#1B4332',
  Return: '#2D6A4F',
  Forehand: '#C8E000',
  Backhand: '#C1440E',
  Volley: '#888880',
  Smash: '#1B4332',
  Slice: '#C8E000',
}

export default function SampleAnalysis() {
  const [activeIdx, setActiveIdx] = useState(0)
  const [playing, setPlaying] = useState(true)

  useEffect(() => {
    if (!playing) return
    const interval = setInterval(() => {
      setActiveIdx((i) => (i + 1) % SHOTS.length)
    }, 1400)
    return () => clearInterval(interval)
  }, [playing])

  const active = SHOTS[activeIdx]

  return (
    <div className="max-w-4xl mx-auto">
      <div className="grid grid-cols-1 md:grid-cols-[1fr_320px] gap-px bg-[#1B4332]/10">
        {/* Court visualisation */}
        <div className="bg-[#FAFAF7] p-8 flex flex-col items-center">
          <CourtDiagram shots={SHOTS} activeIdx={activeIdx} />

          {/* Timeline */}
          <div className="mt-8 w-full flex items-center gap-2">
            {SHOTS.map((shot, i) => (
              <button
                key={i}
                onClick={() => { setActiveIdx(i); setPlaying(false) }}
                className="flex-1 flex flex-col items-center gap-1.5 group"
              >
                <motion.div
                  className="h-0.5 w-full rounded-full transition-colors duration-300"
                  style={{ backgroundColor: i === activeIdx ? SHOT_COLORS[shot.type] : '#1B4332' }}
                  animate={{ scaleX: i === activeIdx ? 1 : 0.6, opacity: i === activeIdx ? 1 : 0.25 }}
                  transition={{ duration: 0.4 }}
                />
                <span
                  className="text-[9px] font-light tracking-widest uppercase transition-colors duration-300"
                  style={{ color: i === activeIdx ? '#1B4332' : '#888880' }}
                >
                  {shot.type}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Stats panel */}
        <div className="bg-[#FAFAF7] p-8 flex flex-col justify-between">
          <div>
            <p className="text-xs text-[#888880] font-light tracking-widest uppercase mb-1">Current shot</p>
            <motion.h3
              key={active.type}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              style={{ fontFamily: '"Playfair Display", Georgia, serif', fontSize: '1.75rem' }}
              className="text-[#1B4332] mb-6"
            >
              {active.type}
            </motion.h3>

            <div className="space-y-5">
              <StatRow label="Player" value={active.player === 'P1' ? 'Player 1' : 'Player 2'} />
              <StatRow label="Ball speed" value={`${active.speed} km/h`} highlight />
              <StatRow label="Shot in rally" value={`${activeIdx + 1} of ${SHOTS.length}`} />
              <StatRow label="Rally length" value={`${SHOTS.length} shots`} />
            </div>
          </div>

          {/* Shot distribution mini */}
          <div className="mt-8 pt-6 border-t border-[#1B4332]/10">
            <p className="text-xs text-[#888880] font-light tracking-widest uppercase mb-4">Shot breakdown</p>
            {['Forehand', 'Backhand', 'Serve', 'Volley', 'Smash'].map((type) => {
              const count = SHOTS.filter((s) => s.type === type).length
              const pct = Math.round((count / SHOTS.length) * 100)
              return (
                <div key={type} className="flex items-center gap-3 mb-2.5">
                  <span className="text-xs text-[#888880] font-light w-20 shrink-0">{type}</span>
                  <div className="flex-1 h-px bg-[#1B4332]/10 relative">
                    <motion.div
                      className="absolute top-1/2 left-0 -translate-y-1/2 h-0.5"
                      style={{ backgroundColor: SHOT_COLORS[type] }}
                      initial={{ width: 0 }}
                      animate={{ width: `${pct}%` }}
                      transition={{ duration: 0.8, ease: 'easeOut' }}
                    />
                  </div>
                  <span className="text-xs text-[#888880] font-light w-6 text-right">{count}</span>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      <div className="mt-4 text-center">
        <button
          onClick={() => setPlaying((p) => !p)}
          className="text-xs text-[#888880] font-light tracking-widest uppercase hover:text-[#1B4332] transition-colors duration-300"
        >
          {playing ? 'Pause' : 'Play'} demo
        </button>
      </div>
    </div>
  )
}

function StatRow({ label, value, highlight }) {
  return (
    <div className="flex items-baseline justify-between border-b border-[#1B4332]/8 pb-3">
      <span className="text-xs text-[#888880] font-light tracking-wide">{label}</span>
      <span
        className="text-sm font-light"
        style={{ color: highlight ? '#C8E000' : '#1B4332', fontVariantNumeric: 'tabular-nums' }}
      >
        {value}
      </span>
    </div>
  )
}

function CourtDiagram({ shots, activeIdx }) {
  const W = 340
  const H = 200
  const MARGIN = { left: 40, right: 40, top: 20, bottom: 20 }

  const courtW = W - MARGIN.left - MARGIN.right
  const courtH = H - MARGIN.top - MARGIN.bottom

  // Ball positions across the rally — alternating sides
  const ballPositions = shots.map((shot, i) => ({
    x: MARGIN.left + (shot.side === 'right' ? courtW * 0.72 : courtW * 0.28),
    y: MARGIN.top + courtH * 0.2 + (i % 3) * (courtH * 0.2),
    ...shot,
  }))

  const active = ballPositions[activeIdx]

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="w-full max-w-sm">
      {/* Court background */}
      <rect x={MARGIN.left} y={MARGIN.top} width={courtW} height={courtH}
        fill="#E8EFE8" stroke="#1B4332" strokeWidth="1" strokeOpacity="0.3" />

      {/* Centre line */}
      <line x1={W / 2} y1={MARGIN.top} x2={W / 2} y2={MARGIN.top + courtH}
        stroke="#1B4332" strokeWidth="0.5" strokeOpacity="0.3" />

      {/* Service boxes */}
      <line x1={MARGIN.left} y1={MARGIN.top + courtH / 2} x2={MARGIN.left + courtW} y2={MARGIN.top + courtH / 2}
        stroke="#1B4332" strokeWidth="0.5" strokeOpacity="0.3" />
      <line x1={MARGIN.left + courtW * 0.12} y1={MARGIN.top} x2={MARGIN.left + courtW * 0.12} y2={MARGIN.top + courtH}
        stroke="#1B4332" strokeWidth="0.5" strokeOpacity="0.2" />
      <line x1={MARGIN.left + courtW * 0.88} y1={MARGIN.top} x2={MARGIN.left + courtW * 0.88} y2={MARGIN.top + courtH}
        stroke="#1B4332" strokeWidth="0.5" strokeOpacity="0.2" />

      {/* Net */}
      <line x1={W / 2 - 2} y1={MARGIN.top - 2} x2={W / 2 - 2} y2={MARGIN.top + courtH + 2}
        stroke="#1B4332" strokeWidth="2" strokeOpacity="0.6" />

      {/* Ball trajectory arc */}
      {activeIdx > 0 && (() => {
        const prev = ballPositions[activeIdx - 1]
        const curr = active
        const midX = (prev.x + curr.x) / 2
        const midY = Math.min(prev.y, curr.y) - 28
        return (
          <path
            d={`M ${prev.x} ${prev.y} Q ${midX} ${midY} ${curr.x} ${curr.y}`}
            fill="none"
            stroke="#C8E000"
            strokeWidth="1.5"
            strokeOpacity="0.7"
            strokeDasharray="3 3"
          />
        )
      })()}

      {/* Past ball positions (faded) */}
      {ballPositions.slice(0, activeIdx).map((pos, i) => (
        <circle key={i} cx={pos.x} cy={pos.y} r={3} fill="#1B4332" fillOpacity={0.15} />
      ))}

      {/* Active ball */}
      <motion.circle
        key={activeIdx}
        cx={active.x}
        cy={active.y}
        r={5}
        fill="#C8E000"
        stroke="#1B4332"
        strokeWidth="1"
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.3 }}
      />

      {/* Players */}
      <circle cx={MARGIN.left + courtW * 0.18} cy={MARGIN.top + courtH * 0.5} r={6}
        fill="#1B4332" fillOpacity="0.8" />
      <circle cx={MARGIN.left + courtW * 0.82} cy={MARGIN.top + courtH * 0.5} r={6}
        fill="#2D6A4F" fillOpacity="0.8" />

      {/* Player labels */}
      <text x={MARGIN.left + courtW * 0.18} y={MARGIN.top + courtH * 0.5 + 16}
        textAnchor="middle" fontSize="7" fill="#888880" fontFamily="DM Sans, sans-serif">P1</text>
      <text x={MARGIN.left + courtW * 0.82} y={MARGIN.top + courtH * 0.5 + 16}
        textAnchor="middle" fontSize="7" fill="#888880" fontFamily="DM Sans, sans-serif">P2</text>

      {/* Shot label badge */}
      <rect x={active.x - 22} y={active.y - 22} width={44} height={14} rx={2}
        fill="#1B4332" fillOpacity="0.85" />
      <text x={active.x} y={active.y - 12} textAnchor="middle"
        fontSize="6.5" fill="#FAFAF7" fontFamily="DM Sans, sans-serif" letterSpacing="0.05em">
        {active.type.toUpperCase()}
      </text>
    </svg>
  )
}
