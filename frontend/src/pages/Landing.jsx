import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowUpRight, Upload, Play } from 'lucide-react'
import { api, responseData } from '../api'

export default function Landing() {
  const navigate = useNavigate()
  const input = useRef(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [name, setName] = useState('')
  const [handedness, setHandedness] = useState('right')
  const [server, setServer] = useState({ max_duration_seconds: 60, demo: false })
  useEffect(() => { api.health().then(r => r.json()).then(setServer).catch(() => {}) }, [])

  async function upload(file) {
    if (!file || busy) return
    setBusy(true)
    setError('')
    try {
      const result = await responseData(await api.upload(file, 'singles', [name], handedness))
      navigate(`/analysis/${result.job_id}`)
    } catch (e) { setError(e.message); setBusy(false) }
  }

  async function sample() {
    if (busy) return
    setError('')
    setBusy(true)
    try {
      const response = await fetch('/demo.mp4')
      if (!response.ok) throw new Error('Sample video is unavailable.')
      const file = new File([await response.blob()], 'federer-nadal-ao2017.mp4', { type: 'video/mp4' })
      // Demo clip: Nadal is the near player and plays left-handed.
      const result = await responseData(await api.upload(file, 'singles', ['Nadal'], 'left'))
      navigate(`/analysis/${result.job_id}`)
    } catch (e) { setError(e.message); setBusy(false) }
  }

  return <div className="workspace">
    <header className="app-header"><a className="brand" href="/">Raquette</a><span className="muted">Video analytics</span></header>
    <main className="upload-layout">
      <section className="upload-form">
        <p className="eyebrow">New analysis</p><h1>Match footage</h1>
        <div className="name-fields single"><label>Near player <span className="field-hint">(back to the camera)</span>
          <input value={name} maxLength={40} placeholder="Near player" onChange={e => setName(e.target.value)} />
        </label></div>
        <fieldset className="handedness"><legend>Near player plays</legend>
          {['right', 'left'].map(hand => <label key={hand}>
            <input type="radio" name="handedness" value={hand} checked={handedness === hand} onChange={() => setHandedness(hand)} />
            {hand === 'right' ? 'Right-handed' : 'Left-handed'}
          </label>)}
        </fieldset>
        <input ref={input} type="file" accept=".mp4,.mov,.m4v,.webm" hidden disabled={busy} onChange={e => upload(e.target.files[0])} />
        <button className="upload-target" disabled={busy} onClick={() => input.current.click()}
          onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); upload(e.dataTransfer.files[0]) }}>
          <Upload size={24} /><strong>{busy ? 'Uploading...' : 'Upload video'}</strong><span>MP4, MOV, M4V, WebM</span>
        </button>
        <p className="muted small">Singles · Up to 256 MB · {server.max_duration_seconds} seconds</p>
        {server.demo && <p className="demo-note">Free demo server: expect about 1.5 minutes of processing per second of video. <a href="https://github.com/LightAnd2/raquette#run-it-locally">Run Raquette locally</a> for full speed and longer clips.</p>}
        {error && <p role="alert" className="error-message">{error}</p>}
      </section>
      <section className="sample-section"><div className="section-heading"><h2>Demo clip</h2><span className="muted small">Federer v Nadal · AO 2017 · 00:08</span></div>
        <video className="sample-video" src="/demo.mp4" controls playsInline preload="metadata" />
        <button className="primary" onClick={sample} disabled={busy}><Play size={16} />{busy ? 'Uploading...' : 'Analyze demo'}<ArrowUpRight size={16} /></button>
      </section>
    </main>
    <footer className="site-footer"><a href="/privacy.html">Privacy and terms</a></footer>
  </div>
}
