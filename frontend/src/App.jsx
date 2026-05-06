import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Landing from './pages/Landing'
import Analysis from './pages/Analysis'
import Results from './pages/Results'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/analysis/:jobId" element={<Analysis />} />
        <Route path="/results/:jobId" element={<Results />} />
      </Routes>
    </BrowserRouter>
  )
}
