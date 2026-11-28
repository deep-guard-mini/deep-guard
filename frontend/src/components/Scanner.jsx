import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import ScanProgress from './ScanProgress'

export default function Scanner({ file, previewUrl, isScanning, result, onComplete, evidenceId }) {
  const [frameIndex, setFrameIndex] = useState(1)

  const isVideo = Boolean(
    file?.type?.startsWith('video/') ||
    ['mp4', 'mov', 'avi'].includes((file?.name || '').split('.').pop()?.toLowerCase())
  )

  useEffect(() => {
    // Only run frame counter animation for video files while actively scanning
    if (!isScanning || !isVideo) return undefined

    const ticker = setInterval(() => {
      setFrameIndex((previous) => (previous >= 120 ? 1 : previous + 1))
    }, 90)

    return () => clearInterval(ticker)
  }, [isScanning, isVideo])

  return (
    <div className="scanner-layout">
      <div className="scanner-shell">
        <div className="scanner-hud">
          <span>{isVideo ? 'VIDEO ANALYSIS' : 'IMAGE ANALYSIS'}</span>
          <span className="status-live">{isScanning ? 'LIVE' : 'RESULT'}</span>
        </div>

        <div className="scan-viewport">
          <div className="scan-grid" />
          <div className={`scan-beam ${isScanning ? 'active' : ''}`} />
          {isVideo ? (
            <video src={previewUrl} controls={false} autoPlay muted loop className="scanner-media video" />
          ) : (
            <motion.img
              src={previewUrl}
              alt={file?.name || 'Forensic scan preview'}
              className="scanner-media image"
              animate={{ rotateX: [0, 7, -3, 0], rotateY: [0, -10, 10, 0], y: [0, -8, 0] }}
              transition={{ duration: 5.5, repeat: Infinity, ease: 'easeInOut' }}
              style={{ transformStyle: 'preserve-3d' }}
            />
          )}

          <div className="scan-overlay left" />
          <div className="scan-overlay right" />
        </div>

        <div className="scanner-footer">
          <div>
            <span className="metric-label">{isVideo ? 'Frame' : 'Target'}</span>
            <strong>{isVideo ? `${String(frameIndex).padStart(3, '0')} / 120` : 'IMAGE 01'}</strong>
          </div>
          <div>
            <span className="metric-label">Status</span>
            <strong>{isScanning ? 'ANALYZING' : (result ? 'COMPLETE' : 'READY')}</strong>
          </div>
          <div>
            <span className="metric-label">Trace</span>
            <strong>{isVideo ? 'video stream' : 'image tensor'}</strong>
          </div>
        </div>
      </div>

      <ScanProgress isScanning={isScanning} result={result} evidenceId={evidenceId} isVideo={isVideo} />
      {onComplete && result && (<button type="button" className="secondary-button result-button" onClick={onComplete}>Continue to Report</button>)}
    </div>
  )
}
