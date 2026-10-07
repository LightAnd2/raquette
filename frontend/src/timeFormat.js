/**
 * Clip-relative timestamp for UI (first frame of upload = 0:00.00).
 * @param {number|null|undefined} sec seconds from start of video
 */
export function formatVideoTimestamp(sec) {
  if (sec == null || Number.isNaN(Number(sec))) return '-'
  const t = Math.max(0, Number(sec))
  const m = Math.floor(t / 60)
  const rem = t - m * 60
  return `${m}:${rem.toFixed(2).padStart(5, '0')}`
}
