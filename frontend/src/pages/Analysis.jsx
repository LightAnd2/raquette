import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { LoaderCircle } from 'lucide-react'
import { api, responseData, mediaUrl } from '../api'

export default function Analysis() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const [job, setJob] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let stopped = false
    let timer
    async function poll() {
      try {
        const result = await responseData(await api.job(jobId))
        if (stopped) return
        setJob(result)
        setError('')
        if (result.status === 'complete') { navigate(`/results/${jobId}`, { replace: true }); return }
        if (result.status === 'error') { setError(result.error); return }
      } catch (e) { if (!stopped) setError(e.message) }
      if (!stopped) timer = setTimeout(poll, 1500)
    }
    poll()
    return () => { stopped = true; clearTimeout(timer) }
  }, [jobId, navigate])
  return <div className="workspace">
    <header className="app-header"><Link className="brand" to="/">Raquette</Link><span className="muted">Analysis</span></header>
    <main className="processing-layout">
      <h1>{job?.filename || 'Video analysis'}</h1>
      {job && <video src={mediaUrl(`/api/video/${jobId}`)} controls playsInline preload="metadata" />}
      <div className="section-heading"><h2>{job?.stage || 'Connecting...'}</h2><span className="tabular">{Math.round(job?.progress || 0)}%</span></div>
      <progress max="100" value={job?.progress || 0} aria-label="Analysis progress" />
      {job?.status !== 'error' && <p className="muted progress-detail"><LoaderCircle className="spinner" size={16} />
        {job?.processed_frames ? `${job.processed_frames} / ${job.total_frames} frames` : 'Preparing analysis'}
      </p>}
      {error && <p role="alert" className="error-message">{error}</p>}
      <Link className="text-link" to="/">New analysis</Link>
    </main>
    <footer className="site-footer"><a href="/privacy.html">Privacy and terms</a></footer>
  </div>
}
