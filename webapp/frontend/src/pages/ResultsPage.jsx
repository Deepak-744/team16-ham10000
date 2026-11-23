import React, { useState } from 'react';
import { 
  ShieldAlert, ShieldCheck, Download, RefreshCw, Eye, EyeOff, 
  AlertTriangle, CheckCircle2, Info, ArrowRight, FileText, Sparkles
} from '../components/Icons';
import ConfidenceChart from '../components/ConfidenceChart';
import { downloadPdfReport } from '../services/api';

export default function ResultsPage({ predictionResult, onReset, setActiveTab }) {
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);
  const [viewMode, setViewMode] = useState('both'); // 'both' | 'original' | 'gradcam'

  if (!predictionResult) {
    return (
      <div className="max-w-2xl mx-auto py-16 text-center space-y-4 bg-white text-black" style={{ backgroundColor: '#ffffff' }}>
        <p className="text-lg text-neutral-700">No prediction data available.</p>
        <button
          onClick={onReset}
          className="px-8 py-3 bg-black text-white border-2 border-black rounded-xl font-bold text-lg shadow-xs cursor-pointer"
        >
          Upload New Image
        </button>
      </div>
    );
  }

  const pred = predictionResult.predicted_class || {};
  const top3 = predictionResult.top_3_predictions || [];
  const dist = predictionResult.confidence_distribution || [];
  const gradcam = predictionResult.gradcam || {};
  const origB64 = predictionResult.original_image_base64;

  const handleDownloadPdf = async () => {
    try {
      setIsDownloadingPdf(true);
      await downloadPdfReport(predictionResult);
    } catch (err) {
      alert("Error generating PDF: " + err.message);
    } finally {
      setIsDownloadingPdf(false);
    }
  };

  // Risk level badge styling
  let riskBg = 'bg-neutral-100 text-neutral-900 border-2 border-black';
  let RiskIcon = ShieldCheck;
  if (pred.risk_level === 'High Risk') {
    riskBg = 'bg-rose-50 text-rose-950 border-2 border-black font-bold';
    RiskIcon = ShieldAlert;
  } else if (pred.risk_level === 'Moderate Risk') {
    riskBg = 'bg-amber-50 text-amber-950 border-2 border-black font-bold';
    RiskIcon = AlertTriangle;
  }

  return (
    <div className="max-w-5xl mx-auto py-8 px-4 sm:px-6 space-y-10 bg-white text-black" style={{ backgroundColor: '#ffffff', color: '#000000' }}>
      {/* Top Action Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b-2 border-black">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-xs sm:text-sm font-bold uppercase tracking-wider text-black bg-neutral-100 px-3 py-1 rounded-md border border-black">
              Analysis Complete
            </span>
            <span className="text-sm text-neutral-600 font-mono">HAM10000 Benchmark</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-black text-black mt-1">Classification Results</h2>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={onReset}
            className="px-5 py-3 text-base font-bold text-black bg-white hover:bg-neutral-100 border-2 border-black rounded-xl shadow-xs transition-colors flex items-center space-x-2 cursor-pointer"
          >
            <RefreshCw className="w-5 h-5 text-black" />
            <span>Analyze Another</span>
          </button>

          <button
            onClick={handleDownloadPdf}
            disabled={isDownloadingPdf}
            className="px-6 py-3 text-base font-bold text-white bg-black hover:bg-neutral-800 border-2 border-black rounded-xl shadow-xs transition-all flex items-center space-x-2 cursor-pointer"
          >
            <Download className="w-5 h-5 text-white" />
            <span>{isDownloadingPdf ? 'Generating PDF...' : 'Download PDF Report'}</span>
          </button>
        </div>
      </div>

      {/* Primary Diagnosis & Risk Banner */}
      <div className="bg-white rounded-3xl p-6 sm:p-8 border-2 border-black shadow-xs" style={{ backgroundColor: '#ffffff' }}>
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-6">
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <span className="font-mono text-sm font-bold uppercase bg-black text-white px-3 py-1 rounded-md">
                {pred.code}
              </span>
              <span className={`flex items-center space-x-2 text-sm font-bold px-3.5 py-1 rounded-full ${riskBg}`}>
                <RiskIcon className="w-5 h-5 text-black" />
                <span>{pred.risk_level}</span>
              </span>
              <span className="text-sm font-semibold text-neutral-600">{pred.type}</span>
            </div>

            <h3 className="text-3xl sm:text-5xl font-black text-black tracking-tight">
              {pred.name}
            </h3>

            <p className="text-base sm:text-lg text-neutral-700 max-w-2xl leading-relaxed">
              {pred.description}
            </p>

            {pred.clinical_guidance && (
              <div className="mt-3 p-4 bg-neutral-50 border-2 border-black rounded-xl text-sm sm:text-base text-neutral-900 leading-relaxed">
                <span className="font-bold text-black block mb-1">Clinical Evaluation Guidance:</span>
                {pred.clinical_guidance}
              </div>
            )}
          </div>

          {/* Confidence Metric Card */}
          <div className="bg-neutral-50 rounded-2xl p-6 border-2 border-black text-center shrink-0 min-w-[220px]">
            <span className="text-sm font-bold text-neutral-600 uppercase tracking-wider block mb-1">
              Model Confidence
            </span>
            <div className="text-5xl sm:text-6xl font-black text-black font-mono tracking-tight">
              {pred.confidence_percent}%
            </div>
            <span className="text-xs text-neutral-600 block mt-2 font-medium">
              Softmax Output Score
            </span>
            <span className="text-[11px] text-neutral-500 block italic mt-0.5">
              (Not a clinical probability)
            </span>
          </div>
        </div>
      </div>

      {/* Visual Comparison: Original vs Grad-CAM */}
      <div className="bg-white rounded-3xl p-6 sm:p-8 border-2 border-black shadow-xs space-y-6" style={{ backgroundColor: '#ffffff' }}>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-xl sm:text-2xl font-black text-black">Visual Morphological Inspection</h3>
            <p className="text-sm text-neutral-600">Comparison of the original dermoscopic photograph and the Grad-CAM visual attention overlay.</p>
          </div>

          {gradcam.available && (
            <div className="flex items-center space-x-1.5 bg-neutral-100 border-2 border-black p-1 rounded-xl text-sm font-bold">
              <button
                onClick={() => setViewMode('both')}
                className={`px-3.5 py-1.5 rounded-lg transition-colors cursor-pointer ${viewMode === 'both' ? 'bg-black text-white' : 'text-neutral-700'}`}
              >
                Side by Side
              </button>
              <button
                onClick={() => setViewMode('original')}
                className={`px-3.5 py-1.5 rounded-lg transition-colors cursor-pointer ${viewMode === 'original' ? 'bg-black text-white' : 'text-neutral-700'}`}
              >
                Original Only
              </button>
              <button
                onClick={() => setViewMode('gradcam')}
                className={`px-3.5 py-1.5 rounded-lg transition-colors cursor-pointer ${viewMode === 'gradcam' ? 'bg-black text-white' : 'text-neutral-700'}`}
              >
                Grad-CAM Only
              </button>
            </div>
          )}
        </div>

        <div className={`grid gap-6 ${viewMode === 'both' && gradcam.available ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1 max-w-xl mx-auto'}`}>
          {/* Original Image */}
          {(viewMode === 'both' || viewMode === 'original') && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm font-bold text-black px-1">
                <span>Original Dermoscopic Image</span>
                <span className="text-xs font-normal text-neutral-600">224x224 scaled</span>
              </div>
              <div className="rounded-2xl overflow-hidden border-2 border-black bg-neutral-50 aspect-4/3 flex items-center justify-center">
                <img
                  src={origB64}
                  alt="Original lesion"
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
          )}

          {/* Grad-CAM Saliency Map */}
          {gradcam.available && (viewMode === 'both' || viewMode === 'gradcam') && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm font-bold text-black px-1">
                <span className="flex items-center space-x-1.5 text-black">
                  <Sparkles className="w-4 h-4 text-neutral-700" />
                  <span>Grad-CAM Saliency Heatmap</span>
                </span>
                <span className="text-xs font-normal text-neutral-600">Peak focus zones</span>
              </div>
              <div className="rounded-2xl overflow-hidden border-2 border-black bg-neutral-50 aspect-4/3 flex items-center justify-center relative">
                <img
                  src={gradcam.image_base64}
                  alt="Grad-CAM Saliency Overlay"
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
          )}
        </div>

        {gradcam.note && (
          <div className="p-4 bg-neutral-50 border-2 border-black rounded-xl text-sm text-neutral-800 flex items-start space-x-3">
            <Info className="w-5 h-5 text-black shrink-0 mt-0.5" />
            <div>{gradcam.note}</div>
          </div>
        )}
      </div>

      {/* Differential Diagnosis: Top 3 Candidates */}
      <div className="bg-white rounded-3xl p-6 sm:p-8 border-2 border-black shadow-xs space-y-4" style={{ backgroundColor: '#ffffff' }}>
        <div>
          <h3 className="text-xl sm:text-2xl font-black text-black">Top 3 Differential Candidates</h3>
          <p className="text-sm text-neutral-600">The three highest-scoring categories identified by the classification head.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {top3.map((cand, idx) => {
            const isRank1 = idx === 0;
            return (
              <div 
                key={cand.code}
                className={`p-5 rounded-2xl border-2 transition-all ${
                  isRank1 
                    ? 'bg-neutral-100 border-black shadow-xs' 
                    : 'bg-white border-black/40'
                }`}
                style={{ backgroundColor: isRank1 ? '#f5f5f5' : '#ffffff' }}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-sm font-bold px-2.5 py-0.5 rounded bg-neutral-200 border border-black text-black">
                    Rank #{idx + 1}
                  </span>
                  <span className={`text-xs font-bold px-2.5 py-0.5 rounded-full border border-black ${
                    cand.risk_level === 'High Risk' ? 'bg-rose-50 text-rose-950' :
                    cand.risk_level === 'Moderate Risk' ? 'bg-amber-50 text-amber-950' :
                    'bg-emerald-50 text-emerald-950'
                  }`}>
                    {cand.risk_level}
                  </span>
                </div>

                <div className="text-lg font-bold text-black mt-2 truncate">
                  {cand.short_name}
                </div>
                <div className="text-sm text-neutral-600 mb-3 font-medium">{cand.type}</div>

                <div className="flex items-baseline justify-between pt-3 border-t border-neutral-300">
                  <span className="text-sm text-neutral-600">Confidence</span>
                  <span className="text-2xl font-black text-black font-mono">
                    {cand.confidence_percent}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Full 7-Class Distribution Chart */}
      <ConfidenceChart distribution={dist} predictedCode={pred.code} />
    </div>
  );
}
