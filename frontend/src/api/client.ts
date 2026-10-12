import {
  ArticleInput,
  ExplanationResponse,
  FeedbackCreate,
  FeedbackResponse,
  HealthResponse,
  PredictionResponse,
} from '../types';

const API_BASE = '/api/v1';

export async function predictArticle(payload: ArticleInput): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Prediction failed' }));
    throw new Error(err.detail || 'Prediction failed');
  }
  return res.json();
}

export async function explainArticle(
  payload: ArticleInput,
  method: 'lime' | 'shap' = 'lime'
): Promise<ExplanationResponse> {
  const res = await fetch(`${API_BASE}/explain?method=${method}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Explainability analysis failed' }));
    throw new Error(err.detail || 'Explainability analysis failed');
  }
  return res.json();
}

export async function submitFeedback(payload: FeedbackCreate): Promise<FeedbackResponse> {
  const res = await fetch(`${API_BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Feedback submission failed' }));
    throw new Error(err.detail || 'Feedback submission failed');
  }
  return res.json();
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error('Health check probe failed');
  }
  return res.json();
}

