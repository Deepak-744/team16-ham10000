/**
 * API Service for communicating with FastAPI backend
 */

const isLocalFrontend = ['localhost', '127.0.0.1', '[::1]'].includes(window.location.hostname);
const API_BASE_URL = import.meta.env.VITE_API_URL?.trim().replace(/\/+$/, '') ||
  (import.meta.env.DEV || isLocalFrontend ? 'http://localhost:8000' : '');
const API_NOT_CONFIGURED = 'The prediction server has not been configured for this site.';
const API_UNAVAILABLE = 'The prediction server is temporarily unavailable. Please try again in a minute.';

async function requestApi(path, options) {
  if (!API_BASE_URL) {
    throw new Error(API_NOT_CONFIGURED);
  }

  try {
    return await fetch(`${API_BASE_URL}${path}`, options);
  } catch (error) {
    throw new Error(API_UNAVAILABLE, { cause: error });
  }
}

export async function checkBackendHealth(options) {
  try {
    const res = await requestApi('/api/health', options);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    return { status: 'offline', error: err.message, model_loaded: false };
  }
}

export async function waitForPredictionServer(signal) {
  if (!API_BASE_URL) throw new Error(API_NOT_CONFIGURED);

  const deadline = Date.now() + 75_000;
  for (let remaining = deadline - Date.now(); remaining > 0; remaining = deadline - Date.now()) {
    signal.throwIfAborted();
    const healthRequest = new AbortController();
    const abortHealthRequest = () => healthRequest.abort();
    signal.addEventListener('abort', abortHealthRequest, { once: true });
    const timeout = setTimeout(abortHealthRequest, Math.min(remaining, 10_000));
    const health = await checkBackendHealth({ cache: 'no-store', signal: healthRequest.signal });
    clearTimeout(timeout);
    signal.removeEventListener('abort', abortHealthRequest);
    signal.throwIfAborted();
    if (health.model_loaded) return;
    if (health.status === 'degraded') {
      throw new Error('The prediction server could not load its model. Please try again later.');
    }

    const delay = Math.min(deadline - Date.now(), 4_000);
    if (delay > 0) await new Promise((resolve) => setTimeout(resolve, delay));
  }

  throw new Error(API_UNAVAILABLE);
}

export async function getSupportedClasses() {
  const res = await requestApi('/api/classes');
  if (!res.ok) throw new Error('Failed to fetch class metadata');
  return await res.json();
}

export async function getSampleImages() {
  try {
    const res = await requestApi('/api/sample-images');
    if (!res.ok) return { samples: [] };
    return await res.json();
  } catch {
    return { samples: [] };
  }
}

export async function predictSkinLesion(fileOrBlob, generateGradCAM = true, signal) {
  const formData = new FormData();
  formData.append('file', fileOrBlob);

  const res = await requestApi(`/api/predict?gradcam=${generateGradCAM}`, {
    method: 'POST',
    body: formData,
    signal,
  });

  if ([502, 503, 504].includes(res.status)) throw new Error(API_UNAVAILABLE);

  if (!res.ok) {
    let errMsg = 'Prediction failed';
    try {
      const errData = await res.json();
      errMsg = errData.detail || errMsg;
    } catch {}
    throw new Error(errMsg);
  }

  return await res.json();
}

export async function downloadPdfReport(inferenceResult) {
  const res = await requestApi('/api/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(inferenceResult),
  });

  if (!res.ok) {
    throw new Error('Failed to generate PDF report');
  }

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `Skin_Lesion_Report_${inferenceResult?.predicted_class?.code?.toUpperCase() || 'Report'}.pdf`;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}
