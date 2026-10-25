/**
 * api.js — Phase 2 frontend–backend integration
 *
 * analyzeMedia(file) sends the file to the appropriate backend endpoint
 * using multipart/form-data (no manual Content-Type header — the browser
 * sets the boundary automatically).
 *
 * On success the backend returns the full detection payload that matches the
 * shape expected by ResultCard, ForensicIndicators and AnalyzePage.
 *
 * On failure a descriptive Error is thrown with a `userMessage` property that
 * AnalyzePage can surface in the UI.
 *
 * The legacy mockAnalysis export is preserved for any code that still imports
 * it directly.
 */

import { analyzeMedia as mockAnalyzeMedia } from './mockAnalysis.js'

/**
 * Determine which detection endpoint to use based on the file's MIME type.
 *
 * @param {File} file
 * @returns {'/api/detect/image' | '/api/detect/video'}
 */
function endpointForFile(file) {
  if (file && file.type && file.type.startsWith('video/')) {
    return '/api/detect/video'
  }
  return '/api/detect/image'
}

/**
 * Send `file` to the backend for deepfake analysis.
 *
 * @param {File} file  – The image or video File object chosen by the user.
 * @returns {Promise<object>}  – The parsed JSON detection payload.
 * @throws {Error}  – With a `.userMessage` string suitable for display.
 */
export async function analyzeMedia(file) {
  const endpoint = endpointForFile(file)

  // Build multipart body — do NOT set Content-Type manually; the browser
  // adds the boundary for us when we pass a FormData instance.
  const body = new FormData()
  body.append('file', file)

  let response
  try {
    response = await fetch(endpoint, {
      method: 'POST',
      body,
    })
  } catch (networkErr) {
    const err = new Error('Network error — could not reach the DeepGuard backend.')
    err.userMessage =
      'Unable to connect to the analysis server. Please check that the backend is running and try again.'
    throw err
  }

  if (!response.ok) {
    // Attempt to parse a structured error body from the backend.
    let detail = null
    try {
      const json = await response.json()
      detail = json?.detail ?? json
    } catch {
      // Body was not valid JSON — fall through to generic message.
    }

    const statusText = `HTTP ${response.status}`
    let userMessage

    if (response.status === 413) {
      userMessage = 'The file is too large. Please upload a file under 100 MB.'
    } else if (response.status === 415) {
      userMessage =
        detail?.message ??
        'Unsupported file format. Please upload a JPEG, PNG, WebP image or MP4, MOV, AVI video.'
    } else if (response.status === 422) {
      userMessage = 'The file field was missing from the request. Please try again.'
    } else if (response.status >= 500) {
      userMessage =
        detail?.message ??
        'The analysis server encountered an error. Please try again in a moment.'
    } else {
      userMessage = detail?.message ?? `Analysis failed (${statusText}).`
    }

    const err = new Error(`Backend returned ${statusText}: ${JSON.stringify(detail)}`)
    err.userMessage = userMessage
    err.status = response.status
    err.detail = detail
    throw err
  }

  const data = await response.json()
  return data
}

/**
 * Legacy export — preserved for backwards compatibility.
 * Returns synchronous mock data without hitting the backend.
 */
export function mockAnalysis(file) {
  return mockAnalyzeMedia(file)
}
