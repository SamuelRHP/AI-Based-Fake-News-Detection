import React, { useState } from 'react';
import { VerdictType } from '../types';
import { MessageSquare, X, Send } from 'lucide-react';

interface FeedbackModalProps {
  predictionId: string;
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (predictionId: string, verdict: VerdictType, comments: string) => Promise<void>;
}

export const FeedbackModal: React.FC<FeedbackModalProps> = ({
  predictionId,
  isOpen,
  onClose,
  onSubmit,
}) => {
  const [userVerdict, setUserVerdict] = useState<VerdictType>('Real');
  const [comments, setComments] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      await onSubmit(predictionId, userVerdict, comments);
      setSubmitted(true);
      setTimeout(() => {
        setSubmitted(false);
        onClose();
      }, 1500);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-xl max-w-md w-full p-6 shadow-xl border border-slate-200">
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-5 h-5 text-brand-600" />
            <h3 className="font-bold text-slate-900 text-lg">Human Feedback Review</h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 p-1 rounded-lg"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {submitted ? (
          <div className="p-4 bg-emerald-50 text-emerald-800 rounded-lg text-sm text-center font-semibold">
            Thank you! Your feedback was logged for model evaluation.
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase mb-2">
                Your Assessment
              </label>
              <div className="grid grid-cols-3 gap-2">
                {(['Fake', 'Real', 'Uncertain'] as VerdictType[]).map((v) => (
                  <button
                    key={v}
                    type="button"
                    onClick={() => setUserVerdict(v)}
                    className={`py-2 text-xs font-bold rounded-lg border transition-all ${
                      userVerdict === v
                        ? 'border-brand-600 bg-brand-50 text-brand-700 ring-2 ring-brand-500'
                        : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    {v}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase mb-1">
                Notes & Context (Optional)
              </label>
              <textarea
                rows={3}
                value={comments}
                onChange={(e) => setComments(e.target.value)}
                placeholder="Why is this article real or fake? Add reference links or notes..."
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-brand-500"
              />
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-100 rounded-lg"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isSubmitting}
                className="px-4 py-2 text-xs font-semibold bg-brand-600 hover:bg-brand-700 text-white rounded-lg flex items-center gap-1.5"
              >
                <Send className="w-3.5 h-3.5" />
                Submit Feedback
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

