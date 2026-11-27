
import { motion } from 'framer-motion'

export default function ScanProgress({ isScanning, result, evidenceId, isVideo }) {
  const hasResult = Boolean(result)
  const activeIsVideo = Boolean(isVideo || result?.mediaType === 'video')
  const frames = result?.framesAnalyzed ?? (hasResult ? (activeIsVideo ? null : 1) : null)
  const faces = result?.facesDetected
  const artifacts = result?.artifactsDetected
  const confidence = result?.confidence

  // Workflow steps representing actual ML operations
  const steps = activeIsVideo
    ? [
        { label: 'MEDIA VALIDATION', status: hasResult ? 'complete' : isScanning ? 'running' : 'waiting' },
        { label: 'FRAME EXTRACTION', status: hasResult ? 'complete' : isScanning ? 'running' : 'waiting' },
        { label: 'TEMPORAL INFERENCE', status: hasResult ? 'complete' : isScanning ? 'running' : 'waiting' },
        { label: 'PREDICTION GENERATION', status: hasResult ? 'complete' : 'waiting' },
      ]
    : [
        { label: 'MEDIA VALIDATION', status: hasResult ? 'complete' : isScanning ? 'running' : 'waiting' },
        { label: 'IMAGE PREPROCESSING', status: hasResult ? 'complete' : isScanning ? 'running' : 'waiting' },
        { label: 'MODEL INFERENCE', status: hasResult ? 'complete' : isScanning ? 'running' : 'waiting' },
        { label: 'PREDICTION GENERATION', status: hasResult ? 'complete' : 'waiting' },
      ]

  // Display unavailable measurements honestly without fabricating numbers.
  const formatMetric = (value, suffix = '') =>
    value == null ? 'Not available' : `${value}${suffix}`

  const resolvedEvidenceId = evidenceId || result?.reportId || 'DG-2026-PENDING'

  const analysisModeLabel = activeIsVideo
    ? hasResult
      ? result?.demoMode
        ? 'VIDEO AI (DEMO)'
        : 'VIDEO AI (ML)'
      : 'VIDEO AI'
    : 'RESNET50 ML'

  const modelLabel = activeIsVideo
    ? hasResult
      ? result?.demoMode
        ? 'DeepGuard Temporal (Demo)'
        : 'DeepGuard Video Classifier (ResNet50)'
      : 'DeepGuard Video Classifier'
    : 'DeepGuard Vision v1 (ResNet50)'

  return (
    <aside className="scan-progress-panel">
      <div className="progress-header">
        <span className="eyebrow">Pipeline</span>
        <span className="live-tag">
          {isScanning ? 'ANALYZING' : hasResult ? 'COMPLETE' : 'READY'}
        </span>
      </div>

      <div className="scan-identity-row">
        <div>
          <span>EVIDENCE ID</span>
          <strong>{resolvedEvidenceId}</strong>
        </div>
        <div>
          <span>ANALYSIS MODE</span>
          <strong>{analysisModeLabel}</strong>
        </div>
      </div>

      <div className="progress-list">
        {steps.map((step, index) => (
          <div key={step.label} className={`progress-item ${step.status}`}>
            <span className="step-index">
              {String(index + 1).padStart(2, '0')}
            </span>

            <div className="step-copy">
              <strong>{step.label}</strong>
              <span>
                {step.status === 'complete'
                  ? '✓ Completed'
                  : step.status === 'running'
                    ? '◉ Processing'
                    : '○ Waiting'}
              </span>
            </div>

            <span className="step-bullet" aria-hidden="true" />
          </div>
        ))}
      </div>

      <div className="scan-metrics-grid">
        <div>
          <span className="metric-label">{activeIsVideo ? 'FRAMES ANALYZED' : 'IMAGES ANALYZED'}</span>
          <strong>{formatMetric(frames)}</strong>
        </div>

        <div>
          <span className="metric-label">FACES DETECTED</span>
          <strong>{formatMetric(faces)}</strong>
        </div>

        <div>
          <span className="metric-label">ARTIFACTS</span>
          <strong>{formatMetric(artifacts)}</strong>
        </div>

        <div>
          <span className="metric-label">MODEL CONFIDENCE</span>
          <strong>{confidence != null ? `${confidence.toFixed(1)}%` : 'Not analyzed'}</strong>
        </div>
      </div>

      <div className="processing-bar-wrap">
        <div className="processing-meta">
          <span>PROCESSING STATUS</span>
          <strong>{hasResult ? '100%' : isScanning ? 'In progress' : '0%'}</strong>
        </div>

        <div className="processing-bar">
          <motion.div
            className="processing-fill"
            initial={{ width: 0 }}
            animate={{ width: hasResult ? '100%' : isScanning ? '65%' : '0%' }}
            transition={{ duration: 0.7, ease: 'easeInOut' }}
          />
        </div>
      </div>

      <motion.div
        className="meta-block"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div>
          <span>MODEL</span>
          <strong>{modelLabel}</strong>
        </div>
        <div>
          <span>ENGINE</span>
          <strong>PyTorch</strong>
        </div>
        <div>
          <span>ANALYSIS MODE</span>
          <strong>{activeIsVideo ? 'VIDEO CLASSIFICATION' : 'IMAGE CLASSIFICATION'}</strong>
        </div>
      </motion.div>
    </aside>
  )
}
