import { useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Download, Pause, Play, RotateCcw, SkipBack, SkipForward, Volume2, VolumeX } from 'lucide-react'
import { api, mediaUrl, responseData } from '../api'
import { formatVideoTimestamp } from '../timeFormat'

const PLAYER = '#C8E000'
const CALL_COLORS = { in: '#C8E000', out: '#C1440E', close: '#b9c4be' }
const SHOT_LETTER = { forehand: 'F', backhand: 'B', serve: 'S' }

function nearest(rows, time, tolerance) {
  const row = rows.reduce((best, item) => !best || Math.abs(item.time - time) < Math.abs(best.time - time) ? item : best, null)
  return row && Math.abs(row.time - time) <= tolerance ? row : null
}

function CourtMap({ data, players, shots, time, seek }) {
  const c = data.court_model
  const calibrated = nearest(data.calibrations, time, 1.1)?.calibration
  const w = c.width, h = c.height, inset = c.singlesInset, service = c.serviceInset
  return <section className="court-section">
    <div className="section-heading"><h2>Court map</h2><span className={`status ${calibrated ? '' : 'unavailable'}`}>{calibrated ? 'Estimated' : 'Unavailable'}</span></div>
    <svg className={`court-map ${calibrated ? '' : 'uncalibrated'}`} viewBox={`-4 -${c.runoff} ${w+8} ${h+c.runoff*2}`} role="img" aria-label="Court with space behind both baselines">
      <rect x="-4" y={-c.runoff} width={w+8} height={h+c.runoff*2} fill="#d5e6dc" />
      <rect width={w} height={h} fill="#2b7568" />
      <g fill="none" stroke="#eef5f1" strokeWidth=".09">
        <rect width={w} height={h} /><rect x={inset} width={w-2*inset} height={h} />
        <path d={`M ${inset} ${service} H ${w-inset} M ${inset} ${h-service} H ${w-inset} M ${w/2} ${service} V ${h-service} M ${w/2} 0 V .3 M ${w/2} ${h} V ${h-.3}`} />
        <path d={`M -.7 ${h/2} H ${w+.7}`} stroke="#172e29" strokeWidth=".15" />
      </g>
      {data.bounces.filter(b => b.court).map(b => <g key={b.id} role="button" tabIndex="0" aria-label={`Bounce ${b.call ?? ""} at ${formatVideoTimestamp(b.time)}`} onClick={() => seek(b.time)} onKeyDown={e => { if (e.key === 'Enter') seek(b.time) }} className="map-event">
        <circle cx={b.court[0]} cy={b.court[1]} r=".38" fill={CALL_COLORS[b.call] ?? '#eef5f1'} stroke="#172e29" strokeWidth=".06" />
      </g>)}
      {shots.filter(s => s.court).map(s => <g key={`s${s.hit_id}`} role="button" tabIndex="0" aria-label={`${s.type} at ${formatVideoTimestamp(s.time)}`} onClick={() => seek(s.time)} onKeyDown={e => { if (e.key === 'Enter') seek(s.time) }} className="map-event" transform={`translate(${s.court.join(' ')})`}>
        <circle r=".6" fill="#FAFAF7" stroke="#1B4332" strokeWidth=".08" /><text textAnchor="middle" dominantBaseline="central" fontSize=".5" fontWeight="600" fill="#1B4332">{SHOT_LETTER[s.type] ?? '?'}</text>
      </g>)}
      {calibrated && players.filter(p => p.court).map(p => <g key={p.id} transform={`translate(${p.court.join(' ')})`}>
        <circle r=".5" fill="#1B4332" stroke="#FAFAF7" strokeWidth=".1" />
      </g>)}
    </svg>
    {!calibrated && <p className="muted small">Court landmarks did not pass validation at this time.</p>}
    <div className="map-legend">
      {Object.entries(CALL_COLORS).map(([call, color]) => <span key={call}><i style={{ background: color }} />{call === 'close' ? 'Too close to call' : `Bounce ${call}`}</span>)}
      <span><i className="shot-key">F</i>Shot position</span>
    </div>
    <RallyStats data={data} shots={shots} />
  </section>
}

function RallyStats({ data, shots }) {
  const mix = shots.reduce((counts, s) => ({ ...counts, [s.type]: (counts[s.type] ?? 0) + 1 }), {})
  const calls = data.bounces.reduce((counts, b) => b.call ? { ...counts, [b.call]: (counts[b.call] ?? 0) + 1 } : counts, {})
  const called = (calls.in ?? 0) + (calls.out ?? 0)
  return <div className="rally-stats">
    <div><span className="muted">Shot mix</span>
      {shots.length ? Object.entries(mix).sort((a, b) => b[1] - a[1]).map(([type, n]) => <p key={type}><strong>{n}</strong> {type}{n > 1 && !type.endsWith('h') ? 's' : n > 1 ? 'es' : ''}</p>) : <p className="muted">No shots detected</p>}
    </div>
    <div><span className="muted">Bounces called</span>
      {called ? <><p><strong>{calls.in ?? 0}</strong> in · <strong>{calls.out ?? 0}</strong> out</p><p className="muted small">{Math.round(100 * (calls.in ?? 0) / called)}% in{calls.close ? ` · ${calls.close} too close to call` : ''}</p></> : <p className="muted">No mapped bounces</p>}
    </div>
  </div>
}

function Playback({ data }) {
  const video = useRef(null)
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [muted, setMuted] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [boxes, setBoxes] = useState(true)
  const [ball, setBall] = useState(true)
  const [playbackError, setPlaybackError] = useState('')
  useEffect(() => {
    const element = video.current
    let handle
    function frame(_now, metadata) {
      setTime(metadata.mediaTime)
      handle = element.requestVideoFrameCallback(frame)
    }
    if (element.requestVideoFrameCallback) handle = element.requestVideoFrameCallback(frame)
    // ?t=1.5 opens the results at that moment, e.g. to share a specific shot.
    const start = Number(new URLSearchParams(window.location.search).get('t'))
    if (start > 0) {
      element.currentTime = start
      setTime(start)
    }
    return () => { if (handle) element.cancelVideoFrameCallback(handle) }
  }, [])
  const players = nearest(data.player_track, time, .13)?.players || []
  const position = nearest(data.ball_track, time, .6/data.video.fps)
  const duration = data.video.duration
  const shots = (data.shots ?? []).map(s => ({ ...s, court: nearest(data.player_track, s.time, .3)?.players?.[0]?.court }))
  const trail = data.ball_track.filter(row => row.point && row.time <= time && row.time >= time - .5)
  const half = court => court[1] < data.court_model.height/2 ? 'Far court' : 'Near court'
  const CALLS = { in: 'in', out: 'out', close: 'too close to call' }
  const events = [
    ...shots.map(s => ({ key: `s${s.hit_id}`, time: s.time, label: s.type[0].toUpperCase() + s.type.slice(1), detail: data.player_names[0] })),
    ...data.bounces.map(b => ({ key: `b${b.id}`, time: b.time, label: 'Bounce', detail: b.court ? `${half(b.court)} · ${CALLS[b.call] ?? ''}` : 'Off the mapped court' })),
  ].sort((a, b) => a.time - b.time)
  function seek(next) {
    const value = Math.max(0, Math.min(next, duration-1/data.video.fps))
    video.current.pause()
    video.current.currentTime = value
    setTime(value)
  }
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a')
    link.href = url; link.download = 'raquette-analysis.json'; link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  return <>
    <div className="results-title"><div><p className="eyebrow">Analysis</p><h1>{data.filename || 'Video analysis'}</h1></div>
      <button className="icon-button" onClick={download} title="Download analysis JSON" aria-label="Download analysis JSON"><Download size={19} /></button></div>
    <div className="metrics">
      <div><span className="muted">Shots</span><strong>{shots.length}</strong></div>
      <div><span className="muted">Bounces</span><strong>{data.bounces.length}</strong></div>
      <div><span className="muted">Ball coverage</span><strong>{Math.round(data.quality.ball_coverage*100)}<small>%</small></strong></div>
      <div><span className="muted">Court found</span><strong>{data.quality.calibrated_samples}<small>/{data.quality.court_samples}</small></strong></div>
      <div><span className="muted">Clip length</span><strong>{formatVideoTimestamp(duration)}</strong></div>
    </div>
    {data.warnings.map(w => <p key={w} className="result-note">{w}</p>)}
    <div className="review-layout">
      <section className="video-section">
        <div className="section-heading"><h2>Video review</h2><div className="overlay-options"><label><input type="checkbox" checked={boxes} onChange={e => setBoxes(e.target.checked)} />Players</label><label><input type="checkbox" checked={ball} onChange={e => setBall(e.target.checked)} />Ball</label></div></div>
        <div className="video-well"><div className="video-stage" style={{ aspectRatio: `${data.video.width} / ${data.video.height}`, width: `min(100%, ${65*data.video.width/data.video.height}vh)` }}>
          <video ref={video} src={mediaUrl(data.video_url)} playsInline preload="auto" muted={muted}
            onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)}
            onTimeUpdate={e => setTime(e.currentTarget.currentTime)} onSeeked={e => setTime(e.currentTarget.currentTime)}
            onError={() => setPlaybackError('This video codec cannot be played by your browser. Export an H.264 MP4.')} />
          <svg className="video-overlay" viewBox={`0 0 ${data.coordinate_space?.width ?? 640} ${data.coordinate_space?.height ?? 360}`} preserveAspectRatio="none" aria-label="Tracked player boxes and ball">
            {boxes && players.map(p => <g key={p.id} stroke={PLAYER} fill="none" strokeWidth="1.2">
              <rect x={p.box[0]} y={p.box[1]} width={p.box[2]-p.box[0]} height={p.box[3]-p.box[1]} />
              <text x={p.box[0]} y={Math.max(9,p.box[1]-4)} fontSize="8" fill={PLAYER} stroke="none">{data.player_names[0]}</text>
            </g>)}
            {ball && trail.length > 1 && <polyline points={trail.map(row => row.point.join(',')).join(' ')} fill="none" stroke="#ebff36" strokeWidth="1.2" strokeOpacity=".55" strokeLinecap="round" strokeLinejoin="round" />}
            {ball && position?.point && <circle cx={position.point[0]} cy={position.point[1]} r="4" stroke="#ebff36" strokeWidth="1.5" fill="none" />}
          </svg>
        </div></div>
        <div className="playback-controls">
          <button className="icon-button" aria-label={playing ? 'Pause' : 'Play'} title={playing ? 'Pause' : 'Play'} onClick={() => playing ? video.current.pause() : video.current.play().catch(e => setPlaybackError(e.message))}>{playing ? <Pause size={19} /> : <Play size={19} />}</button>
          <button className="icon-button" aria-label="Restart" title="Restart" onClick={() => seek(0)}><RotateCcw size={17} /></button>
          <button className="icon-button" aria-label="Previous frame" title="Previous frame" onClick={() => seek(time-1/data.video.fps)}><SkipBack size={17} /></button>
          <button className="icon-button" aria-label="Next frame" title="Next frame" onClick={() => seek(time+1/data.video.fps)}><SkipForward size={17} /></button>
          <span className="tabular playback-time">{formatVideoTimestamp(time)}</span>
          <select aria-label="Playback speed" value={speed} onChange={e => { setSpeed(Number(e.target.value)); video.current.playbackRate = Number(e.target.value) }}>{[.25,.5,1,1.5,2].map(s => <option key={s} value={s}>{s}x</option>)}</select>
          <button className="icon-button" title={muted ? 'Unmute' : 'Mute'} aria-label={muted ? 'Unmute' : 'Mute'} onClick={() => setMuted(!muted)}>{muted ? <VolumeX size={17} /> : <Volume2 size={17} />}</button>
          <input className="scrubber" type="range" min="0" max={duration} step={1/data.video.fps} value={time} aria-label="Video position" onChange={e => seek(Number(e.target.value))} />
        </div>
        {playbackError && <p className="error-message" role="alert">{playbackError}</p>}
        <section className="events-section"><div className="section-heading"><h2>Rally timeline</h2><span className="muted small">Near-player shots and ball bounces</span></div>
          {events.length ? <table><thead><tr><th>Time</th><th>Event</th><th>Detail</th></tr></thead><tbody>{events.map(e => <tr key={e.key} className={Math.abs(e.time-time)<.1 ? 'active-event' : ''}><td><button className="timestamp-button" onClick={() => seek(e.time)}><Play size={12} />{formatVideoTimestamp(e.time)}</button></td><td>{e.label}</td><td>{e.detail}</td></tr>)}</tbody></table> : <p className="muted">No shots or bounces were detected in this clip.</p>}
        </section>
      </section>
      <CourtMap data={data} players={players} shots={shots} time={time} seek={seek} />
    </div>
  </>
}

export default function Results() {
  const { jobId } = useParams()
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let cancelled = false
    api.results(jobId).then(responseData).then(result => { if (!cancelled) setData(result) }).catch(e => { if (!cancelled) setError(e.message) })
    return () => { cancelled = true }
  }, [jobId])
  return <div className="workspace"><header className="app-header"><Link className="brand" to="/">Raquette</Link><Link className="text-link" to="/">New analysis</Link></header>
    <main className="results-layout">{error ? <div role="alert" className="error-message">{error} <Link to={`/analysis/${jobId}`}>Check analysis status</Link></div> : data ? <Playback data={data} /> : <p className="muted">Loading analysis...</p>}</main>
    <footer className="site-footer"><a href="/privacy.html">Privacy and terms</a></footer>
  </div>
}
