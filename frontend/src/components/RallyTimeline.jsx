import { motion } from 'framer-motion'

const SHOT_COLORS = {
  Forehand: '#C8E000',
  Backhand: '#C1440E',
  Serve: '#1B4332',
  Volley: '#2D6A4F',
  Smash: '#1B4332',
  Slice: '#888880',
  Return: '#2D6A4F',
}

export default function RallyTimeline({ shots, activeShot, onSelect }) {
  return (
    <div className="overflow-x-auto">
      <div className="flex items-start gap-1 min-w-max pb-2">
        {shots.map((shot, i) => (
          <motion.button
            key={i}
            onClick={() => onSelect(i === activeShot ? null : i)}
            className="flex flex-col items-center gap-2 group"
            whileHover={{ scale: 1.05 }}
            transition={{ duration: 0.2 }}
          >
            {/* Connector line */}
            {i > 0 && (
              <div className="absolute h-px bg-[#1B4332]/10" />
            )}

            {/* Shot pill */}
            <div
              className="px-3 py-2 text-center transition-all duration-300"
              style={{
                backgroundColor: i === activeShot
                  ? (SHOT_COLORS[shot.type] ?? '#888880')
                  : 'transparent',
                border: `1px solid ${SHOT_COLORS[shot.type] ?? '#888880'}`,
                opacity: i === activeShot ? 1 : 0.6,
              }}
            >
              <p
                className="text-[9px] font-light tracking-widest uppercase whitespace-nowrap"
                style={{ color: i === activeShot ? '#FAFAF7' : (SHOT_COLORS[shot.type] ?? '#888880') }}
              >
                {shot.type}
              </p>
            </div>

            {/* Speed label */}
            <span className="text-[9px] text-[#888880] font-light tabular-nums">
              {shot.speed} km/h
            </span>

            {/* Player indicator */}
            <span className="text-[8px] text-[#888880]/60 font-light">{shot.player}</span>
          </motion.button>
        ))}
      </div>

      {/* Active shot detail */}
      {activeShot !== null && shots[activeShot] && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="mt-6 p-6 border border-[#1B4332]/10 grid grid-cols-3 gap-6"
        >
          <div>
            <p className="text-xs text-[#888880] font-light tracking-widest uppercase mb-1">Shot type</p>
            <p
              className="text-[#1B4332]"
              style={{ fontFamily: '"Playfair Display", Georgia, serif', fontSize: '1.25rem' }}
            >
              {shots[activeShot].type}
            </p>
          </div>
          <div>
            <p className="text-xs text-[#888880] font-light tracking-widest uppercase mb-1">Ball speed</p>
            <p
              className="text-[#C8E000]"
              style={{ fontFamily: '"Playfair Display", Georgia, serif', fontSize: '1.25rem' }}
            >
              {shots[activeShot].speed} km/h
            </p>
          </div>
          <div>
            <p className="text-xs text-[#888880] font-light tracking-widest uppercase mb-1">Player</p>
            <p
              className="text-[#1B4332]"
              style={{ fontFamily: '"Playfair Display", Georgia, serif', fontSize: '1.25rem' }}
            >
              {shots[activeShot].player === 'P1' ? 'Player 1' : 'Player 2'}
            </p>
          </div>
        </motion.div>
      )}
    </div>
  )
}
