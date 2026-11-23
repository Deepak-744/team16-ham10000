import React from 'react';
import { CheckCircle2, AlertCircle } from './Icons';

export default function ConfidenceChart({ distribution, predictedCode }) {
  if (!distribution || distribution.length === 0) return null;

  return (
    <div className="bg-white rounded-2xl p-6 sm:p-8 border-2 border-black shadow-xs">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h3 className="font-black text-black text-lg sm:text-xl">Seven-Class Confidence Distribution</h3>
          <p className="text-sm text-neutral-600">Softmax output scores across all HAM10000 diagnosis categories</p>
        </div>
        <span className="text-xs sm:text-sm font-bold bg-neutral-100 text-black px-3 py-1 rounded-md border border-black">
          Ranked by Confidence
        </span>
      </div>

      <div className="space-y-4">
        {distribution.map((item, index) => {
          const isTop = item.code === predictedCode;
          const conf = item.confidence_percent;
          
          // Badge styling
          let badgeClasses = 'bg-neutral-100 text-neutral-900 border border-black';
          if (item.risk_level === 'High Risk') badgeClasses = 'bg-rose-50 text-rose-950 border border-black font-semibold';
          else if (item.risk_level === 'Moderate Risk') badgeClasses = 'bg-amber-50 text-amber-950 border border-black font-semibold';

          return (
            <div 
              key={item.code}
              className={`p-3.5 rounded-xl border-2 transition-all ${
                isTop 
                  ? 'bg-neutral-100 border-black shadow-xs' 
                  : 'bg-white border-black/30 hover:border-black'
              }`}
            >
              <div className="flex items-center justify-between mb-2 text-sm sm:text-base">
                <div className="flex items-center space-x-2.5">
                  <span className="font-mono font-bold text-sm px-2.5 py-0.5 rounded bg-neutral-100 border border-black text-black uppercase">
                    {item.code}
                  </span>
                  <span className={`font-bold ${isTop ? 'text-black text-base sm:text-lg' : 'text-neutral-800'}`}>
                    {item.short_name}
                  </span>
                  {isTop && (
                    <span className="flex items-center space-x-1.5 text-xs font-bold text-black bg-neutral-200 border border-black px-2.5 py-0.5 rounded-full">
                      <CheckCircle2 className="w-4 h-4 text-black" />
                      <span>Primary Prediction</span>
                    </span>
                  )}
                </div>

                <div className="flex items-center space-x-3">
                  <span className={`text-xs px-2.5 py-0.5 rounded-full ${badgeClasses}`}>
                    {item.risk_level}
                  </span>
                  <span className={`font-mono font-black ${isTop ? 'text-black text-lg' : 'text-neutral-900 text-base'}`}>
                    {conf.toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Progress Bar Container */}
              <div className="w-full h-3 bg-neutral-200 border border-neutral-400 rounded-full overflow-hidden">
                <div 
                  className={`h-full rounded-full transition-all duration-700 ${
                    isTop 
                      ? 'bg-black' 
                      : 'bg-neutral-600'
                  }`}
                  style={{ width: `${Math.max(conf, 1.5)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-5 pt-3 border-t border-neutral-300 flex items-center justify-between text-xs text-neutral-600 font-medium">
        <span>*Scores reflect neural network softmax output distribution.</span>
        <span>Not calibrated clinical disease probabilities.</span>
      </div>
    </div>
  );
}
