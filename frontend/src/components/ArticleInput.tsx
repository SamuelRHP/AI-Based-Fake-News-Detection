import React, { useState } from 'react';
import { FileText, Globe, Sparkles, RefreshCw } from 'lucide-react';
import { ArticleInput as ArticleInputType } from '../types';

interface ArticleInputProps {
  onSubmit: (input: ArticleInputType, explainMethod: 'lime' | 'shap') => void;
  isLoading: boolean;
}

export const ArticleInput: React.FC<ArticleInputProps> = ({ onSubmit, isLoading }) => {
  const [activeTab, setActiveTab] = useState<'text' | 'url'>('text');
  const [text, setText] = useState('');
  const [url, setUrl] = useState('');
  const [title, setTitle] = useState('');
  const [modelType, setModelType] = useState<'baseline' | 'transformer'>('baseline');
  const [explainMethod, setExplainMethod] = useState<'lime' | 'shap'>('lime');

  const handleSampleClick = () => {
    setActiveTab('text');
    setTitle('Shocking Miracle Herb Cures All Sickness Overnight');
    setText(
      'Anonymous whistleblowers and secret scientists confirm you will not believe this breakthrough. ' +
      'Doctors hate this one weird trick that completely reverses all diseases in twenty-four hours. ' +
      'Government officials are allegedly trying to censor this shocking report before it goes viral!'
    );
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (activeTab === 'text' && !text.trim()) return;
    if (activeTab === 'url' && !url.trim()) return;

    onSubmit(
      {
        text: activeTab === 'text' ? text.trim() : undefined,
        url: activeTab === 'url' ? url.trim() : undefined,
        title: title.trim() || undefined,
        model_type: modelType,
      },
      explainMethod
    );
  };

  return (
    <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
      {/* Tab Switcher */}
      <div className="flex items-center justify-between border-b border-slate-200 pb-3 mb-5">
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setActiveTab('text')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-colors ${
              activeTab === 'text'
                ? 'bg-brand-50 text-brand-700 border border-brand-200'
                : 'text-slate-600 hover:bg-slate-100'
            }`}
          >
            <FileText className="w-4 h-4" />
            Raw Text
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('url')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-colors ${
              activeTab === 'url'
                ? 'bg-brand-50 text-brand-700 border border-brand-200'
                : 'text-slate-600 hover:bg-slate-100'
            }`}
          >
            <Globe className="w-4 h-4" />
            Article URL
          </button>
        </div>

        <button
          type="button"
          onClick={handleSampleClick}
          className="text-xs text-brand-600 hover:text-brand-800 font-medium underline flex items-center gap-1"
        >
          <Sparkles className="w-3.5 h-3.5" />
          Load Sample Article
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
            Article Title / Headline (Optional)
          </label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g. Breaking News Headline..."
            className="w-full px-3.5 py-2.5 rounded-lg border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-brand-500"
          />
        </div>

        {activeTab === 'text' ? (
          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
              Article Body Text *
            </label>
            <textarea
              rows={6}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste article content or statements here..."
              required
              className="w-full px-3.5 py-2.5 rounded-lg border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-brand-500"
            />
          </div>
        ) : (
          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
              Article Web URL *
            </label>
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com/news/article-name"
              required
              className="w-full px-3.5 py-2.5 rounded-lg border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-brand-500"
            />
            <p className="text-xs text-slate-500 mt-1">
              The system will extract body text and headlines via the server-side scraper.
            </p>
          </div>
        )}

        {/* Options Row */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Classification Model
            </label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value as 'baseline' | 'transformer')}
              className="w-full px-3 py-2 rounded-lg border border-slate-300 text-sm bg-white"
            >
              <option value="baseline">Baseline (TF-IDF + Logistic Regression - Fast)</option>
              <option value="transformer">Transformer (DistilBERT / RoBERTa - Deep)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Explainability Method
            </label>
            <select
              value={explainMethod}
              onChange={(e) => setExplainMethod(e.target.value as 'lime' | 'shap')}
              className="w-full px-3 py-2 rounded-lg border border-slate-300 text-sm bg-white"
            >
              <option value="lime">LIME (Local Interpretable Explanations)</option>
              <option value="shap">SHAP (Shapley Additive Explanations)</option>
            </select>
          </div>
        </div>

        {/* Submit */}
        <div className="pt-2">
          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 px-4 bg-brand-600 hover:bg-brand-700 disabled:bg-slate-400 text-white font-semibold rounded-lg shadow-sm transition-colors flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <RefreshCw className="w-5 h-5 animate-spin" />
                Analyzing Article & Generating Attributions...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5" />
                Analyze Authenticity & Explain
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

