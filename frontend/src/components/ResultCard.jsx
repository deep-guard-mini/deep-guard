
import { motion } from 'framer-motion'
import { AlertTriangle, CheckCircle2, ShieldAlert } from 'lucide-react'
import ConfidenceCircle from './ConfidenceCircle'

export default function ResultCard({ result }) {
  const assessment =
    result?.assessment ||
    (result?.prediction === 'FAKE'
      ? 'LIKELY MANIPULATED'
      : result?.prediction === 'REAL'
        ? 'LIKELY AUTHENTIC'
        : 'INCONCLUSIVE')

  const accentClass =
    result?.assessmentTone ||
    (result?.prediction === 'FAKE'
      ? 'danger'
      : result?.prediction === 'REAL'
        ? 'safe'
        : 'amber')

  // Show measurements only if they are actually available from the pipeline.
  const facesText =
    result?.facesDetected == null
      ? 'Not analyzed'
      : `${result.facesDetected} regions`

  const artifactsText =
    result?.artifactsDetected == null
      ? 'Not analyzed'
      : `${result.artifactsDetected} detected`

  const isVideo = result?.mediaType === 'video' || result?.type === 'video'
  const modelName = isVideo
    ? result?.demoMode
      ? 'DeepGuard Temporal (Demo)'
      : 'DeepGuard Video Classifier (ResNet50)'
    : 'DeepGuard Vision v1 (ResNet50)'

  return (
    <motion.section
      className="result-card"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="result-card-top">
        <span className="eyebrow">FORENSIC ASSESSMENT</span>
      </div>

      <div className="result-summary">
        <ConfidenceCircle value={result.confidence} />

        <div className="result-status-block">
          <div className={`status-badge ${accentClass}`}>
            {accentClass === 'danger' ? (
              <AlertTriangle size={18} />
            ) : accentClass === 'amber' ? (
              <ShieldAlert size={18} />
            ) : (
              <CheckCircle2 size={18} />
            )}
            {assessment}
          </div>

          <h3>{result.confidence.toFixed(1)}% MODEL CONFIDENCE</h3>
          <p>{result.note}</p>
        </div>
      </div>

      <div className="result-tile-grid">
        <div className="result-tile">
          <span>Media Type</span>
          <strong>{result.mediaType?.toUpperCase() || 'IMAGE'}</strong>
        </div>

        <div className="result-tile">
          <span>Model</span>
          <strong>{modelName}</strong>
        </div>

        <div className="result-tile">
          <span>Confidence</span>
          <strong>{result.confidence.toFixed(1)}%</strong>
        </div>

        <div className="result-tile">
          <span>Analysis Time</span>
          <strong>{result.processingTime.toFixed(2)}s</strong>
        </div>

        <div className="result-tile">
          <span>Regions Analyzed</span>
          <strong>{facesText}</strong>
        </div>

        <div className="result-tile">
          <span>Artifacts</span>
          <strong>{artifactsText}</strong>
        </div>
      </div>

      <div className="security-note">
        <ShieldAlert size={16} />
        AI-ASSISTED FORENSIC ASSESSMENT • NOT DEFINITIVE PROOF
      </div>
    </motion.section>
  )
}
