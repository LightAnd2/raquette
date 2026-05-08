const BASE = import.meta.env.VITE_API_URL ?? ''

export const api = {
  upload: (file, mode, playerNames) => {
    const form = new FormData()
    form.append('file', file)
    form.append('mode', mode)
    form.append('player_names', JSON.stringify(playerNames))
    return fetch(`${BASE}/api/upload`, { method: 'POST', body: form })
  },
  job: (jobId) =>
    fetch(`${BASE}/api/jobs/${jobId}`),
  results: (jobId) =>
    fetch(`${BASE}/api/results/${jobId}`),
}
