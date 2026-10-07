const BASE = import.meta.env.VITE_API_URL ?? ''

export const mediaUrl = path => `${BASE}${path}`

export async function responseData(response) {
  let data
  try { data = await response.json() } catch { throw new Error('The server returned an invalid response.') }
  if (!response.ok || response.status === 202) throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed. Check the local server.')
  return data
}

export const api = {
  upload: (file, mode, playerNames, handedness = 'right') => {
    const form = new FormData()
    form.append('file', file)
    form.append('mode', mode)
    form.append('player_names', JSON.stringify(playerNames))
    form.append('handedness', handedness)
    return fetch(`${BASE}/api/upload`, { method: 'POST', body: form })
  },
  health: () =>
    fetch(`${BASE}/api/health`),
  job: (jobId) =>
    fetch(`${BASE}/api/jobs/${jobId}`),
  results: (jobId) =>
    fetch(`${BASE}/api/results/${jobId}`),
}
