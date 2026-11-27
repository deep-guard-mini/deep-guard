
import { motion } from 'framer-motion'

export default function ForensicIndicators({ indicators }) {
  // Preserve null when an indicator has not actually been measured.
  // Do not convert unavailable data into false 0% readings.
  const entries = [
    { label: 'Face consistency', value: indicators?.faceConsistency ?? null },
    { label: 'Texture consistency', value: indicators?.textureConsistency ?? null },
    { label: 'Lighting consistency', value: indicators?.lightingConsistency ?? null },
    { label: 'Compression artifacts', value: indicators?.compressionArtifacts ?? null },
    { label: 'Edge artifacts', value: indicators?.edgeArtifacts ?? null },
  ]

  return (
    <div className="indicator-panel">
      <div className="section-title-row">
        <h3>Forensic Indicators</h3>
      </div>

      <div className="indicator-list">
        {entries.map((item) => {
          // Check whether this indicator has a real value.
          const isAvailable = item.value !== null

          return (
            <div key={item.label} className="indicator-row">
              <div className="indicator-labels">
                <span>{item.label}</span>

                <strong>
                  {isAvailable ? `${item.value}%` : 'Not analyzed'}
                </strong>
              </div>

              <div className="bar-track">
                {isAvailable && (
                  <motion.div
                    className="bar-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${item.value}%` }}
                    transition={{ duration: 0.8, ease: 'easeOut' }}
                  />
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
