import React from 'react';
import { TokenAttribution, DriverWord } from '../types';
import { AlertCircle, CheckCircle2, Info } from 'lucide-react';

interface TextExplainerProps {
  tokens: TokenAttribution[];
  topFakeDrivers: DriverWord[];
  topRealDrivers: DriverWord[];
  summary: string;
  explainerMethod: string;
}

export const TextExplainer: React.FC<TextExplainerProps> = ({
  tokens,
  topFakeDrivers,
  topRealDrivers,
  summary,
  explainerMethod,
}) => {
  return (
    <div className="space-y-6 bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 pb-4">
        <div>
          <h3 className="text-lg font-bold text-slate-900">Explainable AI Attribution</h3>
          <p className="text-xs text-slate-500">
            Powered by {explainerMethod.toUpperCase()} token importance attribution
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-3.5 rounded bg-red-200 border border-red-300 inline-block" />
            <span className="text-slate-600">Drives Fake</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-3.5 rounded bg-emerald-200 border border-emerald-300 inline-block" />
            <span className="text-slate-600">Drives Real</span>
          </div>
        </div>
      </div>

      {/* Summary Banner */}
      <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 flex items-start gap-3">
        <Info className="w-5 h-5 text-brand-600 shrink-0 mt-0.5" />
        <p className="text-sm text-slate-700 leading-relaxed">{summary}</p>
      </div>

      {/* Interactive Token Heatmap */}
      <div>
        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
          Token-Level Heatmap
        </h4>
        <div className="p-4 bg-slate-50/70 rounded-lg border border-slate-200 leading-relaxed text-slate-800 text-sm max-h-72 overflow-y-auto">
          {tokens.map((token, idx) => {
            let bgColor = 'transparent';
            if (token.impact === 'fake') {
              bgColor = `rgba(239, 68, 68, ${Math.max(0.15, token.intensity * 0.5)})`;
            } else if (token.impact === 'real') {
              bgColor = `rgba(16, 185, 129, ${Math.max(0.15, token.intensity * 0.5)})`;
            }

            return (
              <span
                key={idx}
                style={{ backgroundColor: bgColor }}
                title={`Token: "${token.text}" | Weight: ${token.weight} | Impact: ${token.impact}`}
                className={`transition-colors rounded-sm px-0.5 py-0.2 ${
                  token.impact !== 'neutral' ? 'font-medium cursor-help' : ''
                }`}
              >
                {token.text}
              </span>
            );
          })}
        </div>
      </div>

      {/* Top Drivers Columns */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
        {/* Fake Drivers */}
        <div className="bg-red-50/60 p-4 rounded-lg border border-red-100">
          <div className="flex items-center gap-2 mb-3">
            <AlertCircle className="w-4 h-4 text-red-600" />
            <h4 className="text-xs font-bold text-red-800 uppercase tracking-wider">
              Top Words Driving "Fake"
            </h4>
          </div>
          {topFakeDrivers.length > 0 ? (
            <ul className="space-y-1.5 text-xs">
              {topFakeDrivers.map((driver, i) => (
                <li key={i} className="flex justify-between items-center text-red-900 bg-white/70 px-2 py-1 rounded">
                  <span className="font-semibold">"{driver.word}"</span>
                  <span className="font-mono text-red-600">+{driver.weight.toFixed(3)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-red-500 italic">No strong fake signals detected.</p>
          )}
        </div>

        {/* Real Drivers */}
        <div className="bg-emerald-50/60 p-4 rounded-lg border border-emerald-100">
          <div className="flex items-center gap-2 mb-3">
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            <h4 className="text-xs font-bold text-emerald-800 uppercase tracking-wider">
              Top Words Driving "Real"
            </h4>
          </div>
          {topRealDrivers.length > 0 ? (
            <ul className="space-y-1.5 text-xs">
              {topRealDrivers.map((driver, i) => (
                <li key={i} className="flex justify-between items-center text-emerald-900 bg-white/70 px-2 py-1 rounded">
                  <span className="font-semibold">"{driver.word}"</span>
                  <span className="font-mono text-emerald-600">-{driver.weight.toFixed(3)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-emerald-500 italic">No strong real signals detected.</p>
          )}
        </div>
      </div>
    </div>
  );
};

