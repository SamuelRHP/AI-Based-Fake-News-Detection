export type VerdictType = 'Fake' | 'Real' | 'Uncertain';
export type ImpactType = 'fake' | 'real' | 'neutral';

export interface ArticleInput {
  text?: string;
  url?: string;
  title?: string;
  model_type?: 'baseline' | 'transformer';
}

export interface PredictionResponse {
  prediction_id?: string;
  article_id?: string;
  title?: string;
  verdict: VerdictType;
  confidence_score: number;
  prob_fake: number;
  prob_real: number;
  requires_human_review: boolean;
  model_type: string;
  threshold_applied: {
    high: number;
    low: number;
  };
}

export interface TokenAttribution {
  text: string;
  weight: number;
  impact: ImpactType;
  intensity: number;
}

export interface DriverWord {
  word: string;
  weight: number;
}

export interface ExplanationResponse {
  prediction: PredictionResponse;
  explainer_method: string;
  tokens: TokenAttribution[];
  top_fake_drivers: DriverWord[];
  top_real_drivers: DriverWord[];
  summary: string;
}

export interface FeedbackCreate {
  prediction_id: string;
  user_verdict: VerdictType;
  comments?: string;
}

export interface FeedbackResponse {
  id: string;
  prediction_id: string;
  user_verdict: string;
  comments?: string;
  created_at: string;
}

export interface HealthResponse {
  status: string;
  environment: string;
  active_model: string;
  database_connected: boolean;
  version: string;
}

