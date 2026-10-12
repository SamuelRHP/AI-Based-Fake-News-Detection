import React from 'react';

interface ConfidenceBarProps {
  confidenceScore: number;
  probFake: number;
  probReal: number;
}

export const ConfidenceBar: React.FC<ConfidenceBarProps> = ({
  confidenceScore,
  probFake,
  probReal,
}) => {
  const confPercent = Math.round(confidenceScore * 100);
  const fakePercent = Math.round(probFake * 100);
  const realPercent = Math.round(probReal * 100);

  return (
    <div className="space-y-4 bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
      <div className="flex justify-between items-center">
        <span className="text-sm font-semibold text-slate-700">Model Confidence</span>
        <span className="text-lg font-bold text-slate-900">{confPercent}%</span>
      </div>

      <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
        <div
          className="bg-brand-600 h-3 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${confPercent}%` }}
        />
      </div>

      <div className="pt-2 border-t border-slate-100">
        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block mb-2">
          Estimated Probability Distribution
        </span>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="bg-red-50 p-2.5 rounded-lg border border-red-100">
            <span className="text-xs text-red-600 block font-medium">P(Fake)</span>
            <span className="text-lg font-bold text-red-700">{fakePercent}%</span>
          </div>
          <div className="bg-emerald-50 p-2.5 rounded-lg border border-emerald-100">
            <span className="text-xs text-emerald-600 block font-medium">P(Real)</span>
            <span className="text-lg font-bold text-emerald-700">{realPercent}%</span>
          </div>
        </div>
      </div>
    </div>
  );
};

