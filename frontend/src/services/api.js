/**
 * api.js — DeepGuard frontend-to-backend API integration
 *
 * Provides real API calls to the FastAPI backend:
 *   - checkHealth(): GET /api/health
 *   - detectImage(file): POST /api/detect/image
 *   - detectVideo(file): POST /api/detect/video
 *   - analyzeMedia(file): Dispatches to detectImage or detectVideo based on file type
 *
 * File uploads use multipart/form-data via FormData with field name "file".
 * Content-Type header is intentionally omitted so the browser automatically
 * computes the proper boundary.
 */

import { analyzeMedia as mockAnalyzeMedia } from './mockAnalysis.js'

const API_BASE_URL = (import.meta.env?.VITE_API_BASE_URL || '/api').replace(/\/$/, '')

/**
 * Handle network errors uniformly across all API requests.
 */
function handleNetworkError(err) {
  if (err?.userMessage) throw err
  const networkError = new Error('Network error — could not reach the DeepGuard backend.')
  networkError.userMessage =
    'Unable to connect to the analysis server. Please check that the backend is running at http://localhost:8000 and try again.'
  networkError.cause = err
  throw networkError
}

/**
 * Handle and parse HTTP response, converting error codes to user-friendly messages.
 */
async function handleResponse(response) {
  if (!response.ok) {
    let detail = null
    try {
      detail = await response.json()
    } catch {
      // Non-JSON response body
    }

    const errorPayload =
      detail && typeof detail === 'object' && 'detail' in detail
        ? detail.detail
        : detail

    let errorCode = ''
    let serverMessage = ''
    if (typeof errorPayload === 'object' && errorPayload !== null) {
      errorCode = errorPayload.error || ''
      serverMessage = errorPayload.message || ''
    } else if (typeof errorPayload === 'string') {
      serverMessage = errorPayload
    }

    let userMessage = ''
    if (response.status === 400 || errorCode === 'MISSING_FILE' || errorCode === 'INVALID_FILE') {
      userMessage = serverMessage || 'No file was provided or the file is invalid. Please select a valid file to analyze.'
    } else if (response.status === 413 || errorCode === 'FILE_TOO_LARGE') {
      userMessage = serverMessage || 'The file is too large. Please upload a file under 100 MB.'
    } else if (response.status === 415 || errorCode === 'UNSUPPORTED_FORMAT') {
      userMessage =
        serverMessage ||
        'Unsupported file format. Please upload a JPEG, PNG, WebP image or MP4, MOV, AVI video.'
    } else if (response.status === 422) {
      userMessage = 'The file field was missing or invalid. Please select a valid file to analyze.'
    } else if (response.status >= 500 || errorCode === 'PROCESSING_FAILED') {
      userMessage =
        serverMessage ||
        'The analysis server encountered an error processing your file. Please try again.'
    } else {
      userMessage = serverMessage || `Analysis failed (HTTP ${response.status}).`
    }

    const err = new Error(userMessage)
    err.status = response.status
    err.code = errorCode
    err.userMessage = userMessage
    err.detail = detail
    throw err
  }

  try {
    return await response.json()
  } catch {
    const err = new Error('Invalid JSON response received from backend.')
    err.userMessage = 'Received an invalid response from the analysis server.'
    throw err
  }
}

/**
 * Check backend service health.
 *
 * @returns {Promise<{ status: string, service: string, version: string }>}
 */
export async function checkHealth() {
  let response
  try {
    response = await fetch(`${API_BASE_URL}/health`)
  } catch (err) {
    handleNetworkError(err)
  }
  return handleResponse(response)
}

/**
 * Send an image file to the backend for deepfake analysis.
 *
 * @param {File} file - JPEG, PNG, or WebP file
 * @returns {Promise<object>} - Backend detection response
 */
export async function detectImage(file) {
  if (!file) {
    const err = new Error('No file provided')
    err.userMessage = 'Please select an image file to analyze.'
    throw err
  }

  const formData = new FormData()
  formData.append('file', file)

  let response
  try {
    response = await fetch(`${API_BASE_URL}/detect/image`, {
      method: 'POST',
      body: formData,
    })
  } catch (err) {
    handleNetworkError(err)
  }

  return handleResponse(response)
}

/**
 * Send a video file to the backend for deepfake analysis.
 *
 * @param {File} file - MP4, MOV, or AVI file
 * @returns {Promise<object>} - Backend detection response
 */
export async function detectVideo(file) {
  if (!file) {
    const err = new Error('No file provided')
    err.userMessage = 'Please select a video file to analyze.'
    throw err
  }

  const formData = new FormData()
  formData.append('file', file)

  let response
  try {
    response = await fetch(`${API_BASE_URL}/detect/video`, {
      method: 'POST',
      body: formData,
    })
  } catch (err) {
    handleNetworkError(err)
  }

  return handleResponse(response)
}

/**
 * Check whether a file is a video by MIME type or extension.
 *
 * @param {File} file
 * @returns {boolean}
 */
export function isVideoFile(file) {
  if (!file) return false
  if (file.type && file.type.startsWith('video/')) return true
  const ext = (file.name || '').split('.').pop()?.toLowerCase()
  return ['mp4', 'mov', 'avi'].includes(ext)
}

/**
 * Unified media analysis dispatcher: automatically calls detectVideo or detectImage.
 *
 * @param {File} file - Image or video file object
 * @returns {Promise<object>} - Detection result from FastAPI
 */
export async function analyzeMedia(file) {
  if (isVideoFile(file)) {
    return detectVideo(file)
  }
  return detectImage(file)
}

/**
 * Legacy export preserved for backwards compatibility with tests/demos.
 */
export function mockAnalysis(file) {
  return mockAnalyzeMedia(file)
}
