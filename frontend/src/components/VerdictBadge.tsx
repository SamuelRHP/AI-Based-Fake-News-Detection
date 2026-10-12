import React from 'react';
import { AlertTriangle, CheckCircle, HelpCircle } from 'lucide-react';
import { VerdictType } from '../types';

interface VerdictBadgeProps {
  verdict: VerdictType;
  requiresReview?: boolean;
}

export const VerdictBadge: React.FC<VerdictBadgeProps> = ({ verdict, requiresReview }) => {
  if (verdict === 'Fake') {
    return (
      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full font-bold text-sm bg-red-100 text-red-800 border border-red-300">
        <AlertTriangle className="w-5 h-5 text-red-600" />
        <span>VERDICT: FAKE NEWS</span>
      </div>
    );
  }

  if (verdict === 'Real') {
    return (
      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full font-bold text-sm bg-emerald-100 text-emerald-800 border border-emerald-300">
        <CheckCircle className="w-5 h-5 text-emerald-600" />
        <span>VERDICT: VERIFIED REAL</span>
      </div>
    );
  }

  return (
    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full font-bold text-sm bg-amber-100 text-amber-800 border border-amber-300">
      <HelpCircle className="w-5 h-5 text-amber-600" />
      <span>VERDICT: UNCERTAIN {requiresReview ? '(HUMAN REVIEW RECOMMENDED)' : ''}</span>
    </div>
  );
};

