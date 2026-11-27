
import { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { AlertTriangle, Download, Printer } from 'lucide-react'
import { Link } from 'react-router-dom'
import UploadZone from '../components/UploadZone'
import MediaPreview from '../components/MediaPreview'
import Scanner from '../components/Scanner'
import ResultCard from '../components/ResultCard'
import ForensicIndicators from '../components/ForensicIndicators'
import { detectImage, detectVideo, isVideoFile } from '../services/api'

export default function AnalyzePage({ onAnalysisComplete }) {
  const [selectedFile, setSelectedFile] = useState(null)
  const [evidenceId, setEvidenceId] = useState(null)
  const [isScanning, setIsScanning] = useState(false)
  const [scanResult, setScanResult] = useState(null)
  const [scanError, setScanError] = useState(null)
  const inputRef = useRef(null)

  const previewUrl = useMemo(() => {
    if (!selectedFile) return ''
    return URL.createObjectURL(selectedFile)
  }, [selectedFile])

  useEffect(() => {
    return () => {
      if (previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  const generateEvidenceId = () =>
    `DG-2026-${String(Math.floor(1000 + Math.random() * 9000))}`

  const handleFileSelection = (file) => {
    setSelectedFile(file)
    setScanResult(null)
    setScanError(null)
    setIsScanning(false)
    setEvidenceId(generateEvidenceId())
  }

  const handleRemove = () => {
    setSelectedFile(null)
    setScanResult(null)
    setScanError(null)
    setIsScanning(false)
    setEvidenceId(null)

    if (inputRef.current) {
      inputRef.current.value = ''
    }
  }

  const downloadReport = () => {
    if (!scanResult) return

    const formatVal = (val, suffix = '') =>
      val == null || val === undefined || Number.isNaN(val) ? 'Not available' : `${val}${suffix}`

    const isVideo = scanResult.mediaType === 'video' || scanResult.type === 'video'
    const reportModelName = isVideo
      ? scanResult.demoMode
        ? 'DeepGuard Temporal (Demo)'
        : 'DeepGuard Video Classifier (ResNet50)'
      : 'DeepGuard Vision v1 (ResNet50)'

    const reportText = `=======================================================
DEEPGUARD FORENSIC MEDIA ANALYSIS REPORT
=======================================================
Project Name: DeepGuard Media Forensics
Report ID: ${scanResult.reportId || 'DG-2026-UNSPECIFIED'}
Analysis Date: ${scanResult.analysisDate ? new Date(scanResult.analysisDate).toUTCString() : new Date().toUTCString()}

EVIDENCE INFORMATION
-------------------------------------------------------
Filename: ${scanResult.filename || 'Unknown'}
Media Type: ${(scanResult.mediaType || scanResult.type || 'image').toUpperCase()}
File Size: ${scanResult.fileSize || 'Not available'}
Resolution: ${scanResult.resolution && scanResult.resolution !== '—' ? scanResult.resolution : 'Not available'}
Frames Analyzed: ${formatVal(scanResult.framesAnalyzed)}
Faces Detected: ${formatVal(scanResult.facesDetected)}
Artifacts Detected: ${formatVal(scanResult.artifactsDetected)}

AUTHENTICITY ASSESSMENT
-------------------------------------------------------
Prediction: ${scanResult.prediction || 'INCONCLUSIVE'}
Assessment: ${scanResult.assessment || 'INCONCLUSIVE'}
Model Confidence: ${scanResult.confidence != null ? scanResult.confidence.toFixed(1) + '%' : 'Not available'}
Model: ${reportModelName}
Processing Time: ${scanResult.processingTime != null ? scanResult.processingTime.toFixed(2) + 's' : 'Not available'}
Analysis Mode: ${scanResult.demoMode ? 'DEMO MODE' : 'TRAINED ML INFERENCE'}

FORENSIC INDICATORS
-------------------------------------------------------
Face Consistency: ${formatVal(scanResult.indicators?.faceConsistency, '%')}
Texture Consistency: ${formatVal(scanResult.indicators?.textureConsistency, '%')}
Lighting Consistency: ${formatVal(scanResult.indicators?.lightingConsistency, '%')}
Edge Artifacts: ${formatVal(scanResult.indicators?.edgeArtifacts, '%')}
Compression Artifacts: ${formatVal(scanResult.indicators?.compressionArtifacts, '%')}

SUMMARY & OBSERVATIONS
-------------------------------------------------------
${scanResult.note || 'No observations recorded.'}

LIMITATIONS & DISCLAIMER
-------------------------------------------------------
This assessment is produced by an automated AI deep learning model.
Model predictions represent statistical probability estimates and do NOT constitute
definitive proof of authenticity or manipulation. Results should be verified by
qualified forensic specialists alongside supporting provenance data.
=======================================================`

    const blob = new Blob([reportText], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    const cleanName = (scanResult.filename || 'deepguard-report').replace(/\.[^/.]+$/, '')
    link.href = url
    link.download = `${cleanName}_report_${scanResult.reportId}.txt`
    link.click()
    URL.revokeObjectURL(url)
  }

  const printReport = () => {
    window.print()
  }

  const startAnalysis = async () => {
    if (!selectedFile || isScanning) return

    const activeEvidenceId = evidenceId || generateEvidenceId()
    if (!evidenceId) {
      setEvidenceId(activeEvidenceId)
    }

    setIsScanning(true)
    setScanError(null)
    setScanResult(null)

    let result

    try {
      if (isVideoFile(selectedFile)) {
        result = await detectVideo(selectedFile)
      } else {
        result = await detectImage(selectedFile)
      }
    } catch (err) {
      setIsScanning(false)
      setScanError(
        err.userMessage ?? 'An unexpected error occurred. Please try again.'
      )
      return
    }

    const report = {
      ...result,
      filename: selectedFile.name,
      type: result.mediaType || (isVideoFile(selectedFile) ? 'video' : 'image'),
      mediaType: result.mediaType || (isVideoFile(selectedFile) ? 'video' : 'image'),
      reportId: activeEvidenceId,
      analysisDate: new Date().toISOString(),
      fileSize: `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`,
      resolution: 'Not available',
      framesAnalyzed: result.framesAnalyzed ?? (isVideoFile(selectedFile) ? null : 1),
      facesDetected: result.facesDetected,
      artifactsDetected: result.artifactsDetected,
      confidence: result.confidence ?? 0,
      processingTime: result.processingTime ?? 0,
      indicators: result.indicators ?? {},
      prediction: result.prediction,
      assessment: result.assessment,
      assessmentTone: result.assessmentTone,
      note: result.note,
      demoMode: result.demoMode,
    }

    setScanResult(report)
    setIsScanning(false)
    onAnalysisComplete(report)
  }

  return (
    <div className="page-shell analyze-page container">
      <div className="section-head-row page-label-row">
        <div>
          <span className="eyebrow">Media forensics</span>
          <h1>MEDIA FORENSICS</h1>
        </div>
      </div>

      <p className="lead-copy">
        Submit digital media for authenticity analysis.
      </p>

      {!selectedFile && (
        <UploadZone
          onSelectFile={handleFileSelection}
          inputRef={inputRef}
        />
      )}

      {selectedFile && !isScanning && !scanResult && !scanError && (
        <MediaPreview
          file={selectedFile}
          previewUrl={previewUrl}
          onRemove={handleRemove}
          onAnalyze={startAnalysis}
          isScanning={isScanning}
        />
      )}

      {/* Error state */}
      {scanError && (
        <motion.div
          className="scan-error-banner"
          role="alert"
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <AlertTriangle size={18} aria-hidden="true" />
          <span>{scanError}</span>
          <button
            type="button"
            className="ghost-button"
            onClick={handleRemove}
          >
            TRY AGAIN
          </button>
        </motion.div>
      )}

      {(isScanning || scanResult) && (
        <div className="analysis-workflow">
          <Scanner
            file={selectedFile}
            previewUrl={previewUrl}
            isScanning={isScanning}
            result={scanResult}
            evidenceId={evidenceId}
          />

          {scanResult && (
            <>
              <div className="report-actions">
                <button
                  type="button"
                  className="primary-button"
                  onClick={downloadReport}
                >
                  <Download size={16} /> DOWNLOAD REPORT
                </button>

                <button
                  type="button"
                  className="secondary-button"
                  onClick={printReport}
                >
                  <Printer size={16} /> PRINT REPORT
                </button>

                <button
                  type="button"
                  className="ghost-button"
                  onClick={handleRemove}
                >
                  NEW ANALYSIS
                </button>

                <Link to="/history" className="ghost-button">
                  VIEW HISTORY
                </Link>
              </div>

              <motion.div
                className="report-grid"
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <ResultCard result={scanResult} />

                <ForensicIndicators indicators={scanResult.indicators} />

                {/* Region analysis placeholder: no fake anomaly regions */}
                <div className="suspicious-panel">
                  <div className="section-title-row">
                    <h3>REGION-LEVEL ANALYSIS</h3>
                  </div>

                  <div className="suspicious-viewport">
                    {selectedFile?.type?.startsWith('video/') ? (
                      <video
                        src={previewUrl}
                        className="suspicious-media"
                        playsInline
                        muted
                        loop
                        autoPlay
                      />
                    ) : (
                      <img
                        src={previewUrl}
                        alt="Uploaded media preview"
                        className="suspicious-media"
                      />
                    )}

                    <div className="demo-tag">
                      REGION ANALYSIS NOT AVAILABLE
                    </div>
                  </div>

                  <p className="analysis-unavailable-note">
                    DeepGuard currently performs image-level classification.
                    Region-specific anomaly detection is not yet implemented,
                    so suspicious areas cannot be identified or highlighted.
                  </p>
                </div>
              </motion.div>
            </>
          )}
        </div>
      )}

      {!selectedFile && !isScanning && !scanResult && (
        <div className="info-grid compact-grid" aria-hidden="true" />
      )}
    </div>
  )
}
