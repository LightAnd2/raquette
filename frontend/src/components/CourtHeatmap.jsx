export default function CourtHeatmap({ shots }) {
  const W = 300
  const H = 180
  const M = { left: 20, right: 20, top: 16, bottom: 16 }
  const cW = W - M.left - M.right
  const cH = H - M.top - M.bottom

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="w-full max-w-xs">
      {/* Court */}
      <rect x={M.left} y={M.top} width={cW} height={cH}
        fill="#E8EFE8" stroke="#1B4332" strokeWidth="1" strokeOpacity="0.3" />
      <line x1={W / 2} y1={M.top} x2={W / 2} y2={M.top + cH}
        stroke="#1B4332" strokeWidth="0.5" strokeOpacity="0.3" />
      <line x1={M.left} y1={M.top + cH / 2} x2={M.left + cW} y2={M.top + cH / 2}
        stroke="#1B4332" strokeWidth="0.5" strokeOpacity="0.3" />
      <line x1={M.left + cW * 0.12} y1={M.top} x2={M.left + cW * 0.12} y2={M.top + cH}
        stroke="#1B4332" strokeWidth="0.5" strokeOpacity="0.2" />
      <line x1={M.left + cW * 0.88} y1={M.top} x2={M.left + cW * 0.88} y2={M.top + cH}
        stroke="#1B4332" strokeWidth="0.5" strokeOpacity="0.2" />
      {/* Net */}
      <line x1={W / 2 - 1} y1={M.top - 2} x2={W / 2 - 1} y2={M.top + cH + 2}
        stroke="#1B4332" strokeWidth="2" strokeOpacity="0.5" />

      {/* Shot landing spots */}
      {shots.map((shot, i) => (
        <circle
          key={i}
          cx={M.left + (shot.court_x ?? 0.5) * cW}
          cy={M.top + (shot.court_y ?? 0.5) * cH}
          r={5}
          fill={shot.player === 'P1' ? '#1B4332' : '#C1440E'}
          fillOpacity={0.5}
          stroke="none"
        />
      ))}

      {/* Legend */}
      <circle cx={M.left} cy={M.top + cH + 10} r={3} fill="#1B4332" fillOpacity="0.6" />
      <text x={M.left + 6} y={M.top + cH + 13} fontSize="7" fill="#888880" fontFamily="DM Sans, sans-serif">P1</text>
      <circle cx={M.left + 28} cy={M.top + cH + 10} r={3} fill="#C1440E" fillOpacity="0.6" />
      <text x={M.left + 34} y={M.top + cH + 13} fontSize="7" fill="#888880" fontFamily="DM Sans, sans-serif">P2</text>
    </svg>
  )
}
