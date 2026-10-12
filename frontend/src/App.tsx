import React, { useState, useEffect } from 'react';
import { ArticleInput } from './components/ArticleInput';
import { VerdictBadge } from './components/VerdictBadge';
import { ConfidenceBar } from './components/ConfidenceBar';
import { TextExplainer } from './components/TextExplainer';
import { FeedbackModal } from './components/FeedbackModal';
import { explainArticle, submitFeedback, checkHealth } from './api/client';
import { ArticleInput as ArticleInputType, ExplanationResponse, HealthResponse, VerdictType } from './types';
import { ShieldCheck, Activity, MessageSquarePlus, AlertCircle } from 'lucide-react';

export const App: React.FC = () => {
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isFeedbackOpen, setIsFeedbackOpen] = useState(false);

  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const handleAnalyze = async (input: ArticleInputType, explainMethod: 'lime' | 'shap') => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await explainArticle(input, explainMethod);
      setExplanation(data);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Analysis failed';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeedbackSubmit = async (
    predictionId: string,
    verdict: VerdictType,
    comments: string
  ) => {
    await submitFeedback({
      prediction_id: predictionId,
      user_verdict: verdict,
      comments: comments || undefined,
    });
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      {/* Top Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-brand-600 text-white flex items-center justify-center shadow-sm">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <h1 className="font-extrabold text-slate-900 text-lg leading-tight">
                AI-Based Fake News Detection
              </h1>
              <p className="text-xs text-slate-500">
                Production-Grade Explainable Misinformation Analysis
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {health && (
              <div className="flex items-center gap-2 px-3 py-1 bg-slate-100 rounded-full text-xs font-medium text-slate-600">
                <span
                  className={`w-2 h-2 rounded-full ${
                    health.status === 'ok' ? 'bg-emerald-500' : 'bg-amber-500'
                  }`}
                />
                <span className="capitalize">System: {health.status}</span>
                <span className="text-slate-400">|</span>
                <span className="font-mono text-xs">{health.active_model}</span>
              </div>
            )}
            <a
              href="/docs"
              target="_blank"
              rel="noreferrer"
              className="text-xs text-slate-600 hover:text-slate-900 font-semibold flex items-center gap-1"
            >
              <Activity className="w-4 h-4 text-brand-600" />
              API Docs
            </a>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3 text-red-800 text-sm">
            <AlertCircle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-bold">Analysis Request Failed</p>
              <p className="text-red-700">{error}</p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Input */}
          <div className="lg:col-span-5 space-y-6">
            <ArticleInput onSubmit={handleAnalyze} isLoading={isLoading} />

            <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm text-xs text-slate-500 space-y-2">
              <h4 className="font-bold text-slate-700 uppercase">Architecture Highlights</h4>
              <p>
                <strong>Calibrated Thresholds:</strong> Verdicts are separated into Fake (P &ge; 0.70),
                Real (P &le; 0.30), and Uncertain bands.
              </p>
              <p>
                <strong>Dual Classifier Support:</strong> Fast TF-IDF Logistic Regression baseline alongside
                HuggingFace DistilBERT/RoBERTa sequence transformer.
              </p>
              <p>
                <strong>Explainability:</strong> Token attributions are computed locally via LIME or SHAP,
                pinpointing exact words that drove the decision.
              </p>
            </div>
          </div>

          {/* Right Column: Output & Explainability */}
          <div className="lg:col-span-7 space-y-6">
            {explanation ? (
              <>
                <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block mb-1">
                      Classification Output
                    </span>
                    <VerdictBadge
                      verdict={explanation.prediction.verdict}
                      requiresReview={explanation.prediction.requires_human_review}
                    />
                  </div>

                  {explanation.prediction.prediction_id && (
                    <button
                      type="button"
                      onClick={() => setIsFeedbackOpen(true)}
                      className="px-3.5 py-2 text-xs font-semibold text-brand-700 bg-brand-50 border border-brand-200 hover:bg-brand-100 rounded-lg flex items-center gap-1.5 transition-colors"
                    >
                      <MessageSquarePlus className="w-4 h-4" />
                      Submit Review Feedback
                    </button>
                  )}
                </div>

                <ConfidenceBar
                  confidenceScore={explanation.prediction.confidence_score}
                  probFake={explanation.prediction.prob_fake}
                  probReal={explanation.prediction.prob_real}
                />

                <TextExplainer
                  tokens={explanation.tokens}
                  topFakeDrivers={explanation.top_fake_drivers}
                  topRealDrivers={explanation.top_real_drivers}
                  summary={explanation.summary}
                  explainerMethod={explanation.explainer_method}
                />

                {explanation.prediction.prediction_id && (
                  <FeedbackModal
                    predictionId={explanation.prediction.prediction_id}
                    isOpen={isFeedbackOpen}
                    onClose={() => setIsFeedbackOpen(false)}
                    onSubmit={handleFeedbackSubmit}
                  />
                )}
              </>
            ) : (
              <div className="bg-white border-2 border-dashed border-slate-200 rounded-xl p-12 text-center text-slate-400">
                <ShieldCheck className="w-12 h-12 mx-auto text-slate-300 mb-3" />
                <h3 className="font-bold text-slate-700 text-base mb-1">No Article Analyzed Yet</h3>
                <p className="text-xs max-w-sm mx-auto">
                  Submit an article URL or paste text on the left to inspect authenticity verdicts,
                  confidence distributions, and token attribution heatmaps.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-6 text-center text-xs text-slate-500">
        AI-Based Fake News Detection System &bull; Production Architecture &bull; MLflow &bull; DVC &bull; FastAPI &bull; React
      </footer>
    </div>
  );
};

export default App;

