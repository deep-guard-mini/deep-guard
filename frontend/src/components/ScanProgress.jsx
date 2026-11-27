export default function ScanProgress({ isScanning, result, evidenceId, isVideo }) {
  const hasResult = Boolean(result)
  const activeIsVideo = Boolean(isVideo || result?.mediaType === 'video')

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

  const resolvedEvidenceId = evidenceId || result?.reportId || 'DG-2026-PENDING'

  const analysisModeLabel = activeIsVideo
    ? hasResult
      ? result?.demoMode
        ? 'VIDEO AI (DEMO)'
        : 'VIDEO AI (ML)'
      : 'VIDEO AI'
    : 'RESNET50 ML'

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
    </aside>
  )
}
