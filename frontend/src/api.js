const BASE = import.meta.env.VITE_API_URL ?? ''

export const api = {
  upload: (formData) =>
    fetch(`${BASE}/api/upload`, { method: 'POST', body: formData }),
  job: (jobId) =>
    fetch(`${BASE}/api/jobs/${jobId}`),
  results: (jobId) =>
    fetch(`${BASE}/api/results/${jobId}`),
}
