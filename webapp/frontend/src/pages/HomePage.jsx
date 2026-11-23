import React from 'react';
import { 
  ShieldCheck, UploadCloud, Cpu, FileText, ArrowRight, Layers
} from '../components/Icons';

export default function HomePage({ setActiveTab, classes = [] }) {
  const steps = [
    {
      number: '01',
      title: 'Upload Dermoscopic Image',
      description: 'Upload a clear photograph or dermatoscopic scan in JPG, JPEG, or PNG format, or choose from standard sample lesions.',
      icon: UploadCloud
    },
    {
      number: '02',
      title: 'Deep Learning Inference',
      description: 'Our trained convolutional neural network analyzes multi-scale morphological features, texture patterns, and pigmentation symmetry.',
      icon: Cpu
    },
    {
      number: '03',
      title: 'Diagnostic Report & Grad-CAM',
      description: 'Inspect class probabilities, top candidates, Grad-CAM saliency heatmaps, and download a clinical summary report.',
      icon: FileText
    }
  ];

  return (
    <div className="space-y-14 py-8 max-w-6xl mx-auto px-4 sm:px-6 bg-white text-black" style={{ backgroundColor: '#ffffff' }}>
      {/* Hero Section - Clean White Background with Black Outer Layer */}
      <section className="bg-white border-2 border-black rounded-3xl p-8 sm:p-14 text-black shadow-xs" style={{ backgroundColor: '#ffffff' }}>
        <div className="max-w-3xl mx-auto text-center space-y-6">
          <h1 className="text-4xl sm:text-6xl font-black tracking-tight text-black leading-tight">
            Skin Lesion Classification
          </h1>

          <p className="text-lg sm:text-xl text-neutral-700 max-w-2xl mx-auto leading-relaxed">
            Automated multiclass classification of skin lesions using deep learning on the HAM10000 dataset. Upload a skin lesion image to view prediction confidence scores, differential candidate rankings, and Grad-CAM visual attention overlays.
          </p>

          <div className="pt-3 flex flex-col sm:flex-row items-center justify-center gap-4">
            <button
              onClick={() => setActiveTab('upload')}
              className="w-full sm:w-auto px-9 py-4 rounded-xl font-bold text-lg text-white bg-black hover:bg-neutral-800 border-2 border-black transition-all flex items-center justify-center space-x-2.5 shadow-xs cursor-pointer"
            >
              <UploadCloud className="w-6 h-6 text-neutral-300" />
              <span>Start Lesion Analysis</span>
            </button>

            <button
              onClick={() => setActiveTab('about')}
              className="w-full sm:w-auto px-7 py-4 rounded-xl font-semibold text-lg text-black bg-white hover:bg-neutral-100 border-2 border-black transition-all flex items-center justify-center space-x-2.5 cursor-pointer"
            >
              <Layers className="w-5 h-5 text-neutral-700" />
              <span>Model & HAM10000 Details</span>
            </button>
          </div>

          {/* Metrics Row - White Background with Black Outer Border */}
          <div className="pt-8 grid grid-cols-2 sm:grid-cols-4 gap-4 border-t-2 border-black">
            <div className="p-4 bg-neutral-50 border border-black rounded-xl">
              <div className="text-3xl sm:text-4xl font-extrabold text-black font-mono">83.46%</div>
              <div className="text-sm font-semibold text-neutral-600 mt-1">Test Accuracy</div>
            </div>
            <div className="p-4 bg-neutral-50 border border-black rounded-xl">
              <div className="text-3xl sm:text-4xl font-extrabold text-black font-mono">81.23%</div>
              <div className="text-sm font-semibold text-neutral-600 mt-1">Balanced Accuracy</div>
            </div>
            <div className="p-4 bg-neutral-50 border border-black rounded-xl">
              <div className="text-3xl sm:text-4xl font-extrabold text-black font-mono">94.76%</div>
              <div className="text-sm font-semibold text-neutral-600 mt-1">Macro AUC</div>
            </div>
            <div className="p-4 bg-neutral-50 border border-black rounded-xl">
              <div className="text-3xl sm:text-4xl font-extrabold text-black font-mono">7 Classes</div>
              <div className="text-sm font-semibold text-neutral-600 mt-1">HAM10000 Taxonomy</div>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="space-y-6">
        <div className="text-center max-w-2xl mx-auto">
          <h2 className="text-3xl sm:text-4xl font-black text-black tracking-tight">How the Classification System Works</h2>
          <p className="text-neutral-700 text-base sm:text-lg mt-2">
            A three-stage workflow ensuring rigorous image normalization, deep-feature classification, and explainable visualization.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {steps.map((step) => {
            const Icon = step.icon;
            return (
              <div key={step.number} className="bg-white rounded-2xl p-7 border-2 border-black shadow-xs flex flex-col justify-between" style={{ backgroundColor: '#ffffff' }}>
                <div>
                  <div className="flex items-center justify-between mb-5">
                    <div className="w-13 h-13 rounded-xl bg-neutral-100 border-2 border-black text-neutral-800 flex items-center justify-center font-bold">
                      <Icon className="w-7 h-7 text-neutral-800" />
                    </div>
                    <span className="font-mono text-base font-bold text-neutral-500">{step.number}</span>
                  </div>
                  <h3 className="text-xl font-bold text-black mb-2">{step.title}</h3>
                  <p className="text-base text-neutral-700 leading-relaxed">{step.description}</p>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Supported Lesion Categories Grid */}
      <section className="space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between">
          <div>
            <h2 className="text-3xl sm:text-4xl font-black text-black tracking-tight">Seven Diagnosed Lesion Categories</h2>
            <p className="text-base text-neutral-700 mt-1">Diagnostic classes from the HAM10000 benchmark dataset.</p>
          </div>
          <button 
            onClick={() => setActiveTab('about')}
            className="mt-3 sm:mt-0 text-base font-bold text-black underline flex items-center space-x-1 cursor-pointer"
          >
            <span>Read full medical taxonomy</span>
            <ArrowRight className="w-5 h-5 text-black" />
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {classes.map((cls) => {
            let riskBadge = 'bg-neutral-100 text-neutral-900 border border-black';
            if (cls.risk_level === 'High Risk') riskBadge = 'bg-rose-50 text-rose-950 border border-black font-semibold';
            else if (cls.risk_level === 'Moderate Risk') riskBadge = 'bg-amber-50 text-amber-950 border border-black font-semibold';

            return (
              <div 
                key={cls.code} 
                className="bg-white rounded-xl p-5 border-2 border-black shadow-xs flex flex-col justify-between"
                style={{ backgroundColor: '#ffffff' }}
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono text-sm font-bold uppercase px-2.5 py-0.5 rounded bg-neutral-100 border border-black text-black">
                      {cls.code}
                    </span>
                    <span className={`text-xs px-2.5 py-0.5 rounded-full ${riskBadge}`}>
                      {cls.risk_level}
                    </span>
                  </div>
                  <h4 className="font-bold text-black text-base mb-1">{cls.short_name}</h4>
                  <p className="text-sm text-neutral-700 line-clamp-3 leading-relaxed">{cls.description}</p>
                </div>
                <div className="mt-4 pt-3 border-t border-neutral-300 text-xs font-semibold text-neutral-600">
                  {cls.type}
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
