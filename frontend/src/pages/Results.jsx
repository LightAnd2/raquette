import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { api } from '../api'
import { motion } from 'framer-motion'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts'
import CourtHeatmap from '../components/CourtHeatmap'
import RallyTimeline from '../components/RallyTimeline'

const COLORS = {
  Forehand: '#C8E000',
  Backhand: '#C1440E',
  Serve: '#1B4332',
  Volley: '#2D6A4F',
  Smash: '#888880',
  Slice: '#A3B18A',
  Return: '#2D6A4F',
}

const MOCK_RESULTS = {
  shots: [
    { type: 'Serve', speed: 182, player: 'P1', time: 0.5, court_x: 0.72, court_y: 0.15 },
    { type: 'Return', speed: 94, player: 'P2', time: 2.1, court_x: 0.28, court_y: 0.75 },
    { type: 'Forehand', speed: 112, player: 'P1', time: 3.4, court_x: 0.65, court_y: 0.4 },
    { type: 'Backhand', speed: 87, player: 'P2', time: 4.8, court_x: 0.35, court_y: 0.6 },
    { type: 'Forehand', speed: 128, player: 'P1', time: 6.0, court_x: 0.7, court_y: 0.3 },
    { type: 'Volley', speed: 71, player: 'P2', time: 7.2, court_x: 0.42, court_y: 0.5 },
    { type: 'Smash', speed: 156, player: 'P1', time: 8.1, court_x: 0.55, court_y: 0.25 },
  ],
  rally_length: 7,
  winner: 'P1',
  avg_speed: 118,
  max_speed: 182,
}

export default function Results() {
  const { jobId } = useParams()
  const [data, setData] = useState(null)
  const [activeShot, setActiveShot] = useState(null)

  useEffect(() => {
    api.results(jobId)
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData(MOCK_RESULTS))
  }, [jobId])

  const results = data ?? MOCK_RESULTS

  const shotTally = results.shots.reduce((acc, s) => {
    acc[s.type] = (acc[s.type] ?? 0) + 1
    return acc
  }, {})

  const pieData = Object.entries(shotTally).map(([name, value]) => ({ name, value }))

  return (
    <div className="min-h-screen bg-[#FAFAF7]">
      {/* Nav */}
      <nav className="flex items-center justify-between px-10 py-6 border-b border-[#1B4332]/10">
        <a
          href="/"
          className="text-[#1B4332]/60 text-sm font-light tracking-widest uppercase hover:text-[#1B4332] transition-colors duration-300"
          style={{ letterSpacing: '0.3em' }}
        >
          Raquette
        </a>
        <button
          onClick={() => window.print()}
          className="text-xs text-[#888880] font-light tracking-widest uppercase hover:text-[#1B4332] transition-colors duration-300"
        >
          Download report
        </button>
      </nav>

      <div className="max-w-6xl mx-auto px-8 py-16 space-y-20">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center"
        >
          <p className="text-xs text-[#888880] font-light tracking-widest uppercase mb-3">Match analysis</p>
          <h1
            className="text-[#1B4332] mb-4"
            style={{
              fontFamily: '"Playfair Display", Georgia, serif',
              fontSize: 'clamp(2rem, 5vw, 3.5rem)',
              fontWeight: 600,
            }}
          >
            Rally Report
          </h1>
          <p className="text-sm text-[#888880] font-light tracking-wide">
            {results.rally_length} shots · {results.avg_speed} km/h average · Won by {results.winner}
          </p>
        </motion.div>

        {/* Summary cards */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-px bg-[#1B4332]/10"
        >
          {[
            { label: 'Rally length', value: `${results.rally_length} shots` },
            { label: 'Avg ball speed', value: `${results.avg_speed} km/h`, highlight: true },
            { label: 'Peak speed', value: `${results.max_speed} km/h` },
            { label: 'Winner', value: results.winner },
          ].map((card) => (
            <div key={card.label} className="bg-[#FAFAF7] p-8 text-center">
              <p className="text-xs text-[#888880] font-light tracking-widest uppercase mb-2">{card.label}</p>
              <p
                className="text-2xl font-light"
                style={{
                  fontFamily: '"Playfair Display", Georgia, serif',
                  color: card.highlight ? '#C8E000' : '#1B4332',
                }}
              >
                {card.value}
              </p>
            </div>
          ))}
        </motion.div>

        {/* Shot distribution + heatmap */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="grid grid-cols-1 md:grid-cols-2 gap-px bg-[#1B4332]/10"
        >
          {/* Donut chart */}
          <div className="bg-[#FAFAF7] p-10">
            <p className="text-xs text-[#888880] font-light tracking-widest uppercase mb-1">Shot distribution</p>
            <h2
              className="text-[#1B4332] mb-8"
              style={{ fontFamily: '"Playfair Display", Georgia, serif', fontSize: '1.5rem' }}
            >
              By type
            </h2>
            <div className="flex items-center gap-8">
              <ResponsiveContainer width={160} height={160}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={48}
                    outerRadius={72}
                    strokeWidth={0}
                    dataKey="value"
                  >
                    {pieData.map((entry) => (
                      <Cell key={entry.name} fill={COLORS[entry.name] ?? '#888880'} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: '#FAFAF7',
                      border: '1px solid rgba(27,67,50,0.2)',
                      borderRadius: 0,
                      fontSize: 11,
                      fontFamily: '"DM Sans", system-ui',
                      color: '#1B4332',
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="space-y-2 flex-1">
                {pieData.map((entry) => (
                  <div key={entry.name} className="flex items-center gap-2">
                    <div className="w-2 h-2 shrink-0" style={{ backgroundColor: COLORS[entry.name] }} />
                    <span className="text-xs text-[#888880] font-light flex-1">{entry.name}</span>
                    <span className="text-xs text-[#1B4332] font-light tabular-nums">{entry.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Court heatmap */}
          <div className="bg-[#FAFAF7] p-10">
            <p className="text-xs text-[#888880] font-light tracking-widest uppercase mb-1">Shot placement</p>
            <h2
              className="text-[#1B4332] mb-8"
              style={{ fontFamily: '"Playfair Display", Georgia, serif', fontSize: '1.5rem' }}
            >
              Court heatmap
            </h2>
            <CourtHeatmap shots={results.shots} />
          </div>
        </motion.div>

        {/* Rally timeline */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="bg-[#FAFAF7] border border-[#1B4332]/10 p-10"
        >
          <p className="text-xs text-[#888880] font-light tracking-widest uppercase mb-1">Shot by shot</p>
          <h2
            className="text-[#1B4332] mb-8"
            style={{ fontFamily: '"Playfair Display", Georgia, serif', fontSize: '1.5rem' }}
          >
            Rally timeline
          </h2>
          <RallyTimeline shots={results.shots} activeShot={activeShot} onSelect={setActiveShot} />
        </motion.div>

        {/* Export */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="text-center pb-8"
        >
          <button
            onClick={() => window.print()}
            className="inline-flex items-center gap-3 px-10 py-4 border border-[#1B4332] text-[#1B4332] text-sm font-light tracking-widest uppercase hover:bg-[#1B4332] hover:text-[#FAFAF7] transition-all duration-500"
          >
            Download report
          </button>
        </motion.div>
      </div>
    </div>
  )
}
