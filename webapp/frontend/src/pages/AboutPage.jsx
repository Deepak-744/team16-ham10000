import React from 'react';
import { 
  Database, Brain, CheckCircle2, AlertTriangle, Layers, 
  ShieldAlert, BookOpen, Cpu 
} from '../components/Icons';

export default function AboutPage({ classes = [] }) {
  const metrics = [
    { label: "Test Accuracy", value: "83.46%", description: "Overall classification accuracy across test set (1,016 images)" },
    { label: "Balanced Accuracy", value: "81.23%", description: "Macro average of per-class recall to prevent majority class bias" },
    { label: "Macro F1-Score", value: "77.83%", description: "Harmonic mean of precision and recall unweighted across all 7 classes" },
    { label: "Macro AUC", value: "94.76%", description: "Area under the receiver operating characteristic curve" },
  ];

  const perClassRecalls = [
    { code: "MEL", name: "Melanoma", recall: "71.43%", support: 112, type: "Malignant" },
    { code: "BCC", name: "Basal Cell Carcinoma", recall: "88.89%", support: 54, type: "Malignant" },
    { code: "AKIEC", name: "Actinic Keratoses", recall: "60.98%", support: 41, type: "Pre-malignant" },
    { code: "BKL", name: "Benign Keratosis", recall: "76.36%", support: 110, type: "Benign" },
    { code: "NV", name: "Melanocytic Nevi", recall: "87.22%", support: 673, type: "Benign" },
    { code: "DF", name: "Dermatofibroma", recall: "90.00%", support: 10, type: "Benign" },
    { code: "VASC", name: "Vascular Lesions", recall: "93.75%", support: 16, type: "Benign" },
  ];

  return (
    <div className="max-w-5xl mx-auto py-8 px-4 sm:px-6 space-y-12 bg-white text-black" style={{ backgroundColor: '#ffffff', color: '#000000' }}>
      {/* Header */}
      <div className="text-center max-w-2xl mx-auto">
        <span className="text-sm font-bold uppercase tracking-wider text-black bg-neutral-100 px-3.5 py-1 rounded-full border border-black">
          Dataset & Architecture
        </span>
        <h2 className="text-3xl sm:text-5xl font-black text-black mt-3 tracking-tight">
          About the Project & Model
        </h2>
        <p className="text-neutral-700 mt-2 text-base sm:text-lg leading-relaxed">
          Documentation of the HAM10000 benchmark dataset, deep neural architecture, experimental methodology, and clinical limitations.
        </p>
      </div>

      {/* 1. Dataset Information */}
      <section className="bg-white rounded-3xl p-6 sm:p-8 border-2 border-black shadow-xs space-y-4" style={{ backgroundColor: '#ffffff' }}>
        <div className="flex items-center space-x-3 text-black">
          <Database className="w-7 h-7 text-black" />
          <h3 className="text-2xl font-black text-black">HAM10000 Dataset</h3>
        </div>
        <p className="text-base sm:text-lg text-neutral-700 leading-relaxed">
          The <strong>HAM10000</strong> ("Human Against Machine with 10,000 training images") benchmark dataset consists of <strong>10,015 dermatoscopic images</strong> collected from the Department of Dermatology at the Medical University of Vienna and the Cliff Rosendahl clinic in Queensland, Australia. More than 50% of lesions are confirmed by pathology/biopsy, while remaining benign cases are validated through follow-up or expert consensus.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
          <div className="p-4 bg-neutral-50 rounded-xl border border-black">
            <div className="text-xs font-bold text-neutral-600 uppercase">Training Split</div>
            <div className="text-3xl font-extrabold text-black font-mono mt-1">7,991</div>
            <div className="text-sm text-neutral-700 mt-0.5 font-medium">Images (79.8%)</div>
          </div>
          <div className="p-4 bg-neutral-50 rounded-xl border border-black">
            <div className="text-xs font-bold text-neutral-600 uppercase">Validation Split</div>
            <div className="text-3xl font-extrabold text-black font-mono mt-1">1,008</div>
            <div className="text-sm text-neutral-700 mt-0.5 font-medium">Images (10.1%)</div>
          </div>
          <div className="p-4 bg-neutral-50 rounded-xl border border-black">
            <div className="text-xs font-bold text-neutral-600 uppercase">Held-out Test Split</div>
            <div className="text-3xl font-extrabold text-black font-mono mt-1">1,016</div>
            <div className="text-sm text-neutral-700 mt-0.5 font-medium">Images (10.1%)</div>
          </div>
        </div>
      </section>

      {/* 2. Model Architecture */}
      <section className="bg-white rounded-3xl p-6 sm:p-8 border-2 border-black shadow-xs space-y-4" style={{ backgroundColor: '#ffffff' }}>
        <div className="flex items-center space-x-3 text-black">
          <Cpu className="w-7 h-7 text-black" />
          <h3 className="text-2xl font-black text-black">Deep Neural Network Architecture</h3>
        </div>
        <p className="text-base sm:text-lg text-neutral-700 leading-relaxed">
          The deep learning model utilizes 7x7 depthwise convolutions, inverted bottlenecks, and smooth GELU activations to optimize multi-scale feature extraction across subtle pigmentary networks and boundary contours.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 text-sm sm:text-base text-neutral-800">
          <div className="p-5 bg-neutral-50 rounded-xl border-2 border-black space-y-2">
            <h4 className="font-bold text-black text-base sm:text-lg">Architectural Specifications:</h4>
            <ul className="list-disc pl-5 space-y-1.5 text-neutral-700">
              <li><strong>7x7 Depthwise Convolutions:</strong> Wide receptive field capturing global lesion contour and symmetry.</li>
              <li><strong>Inverted Bottleneck:</strong> Preserves spatial richness during feature compression.</li>
              <li><strong>LayerNorm & GELU:</strong> Smooth gradient propagation and stable normalization.</li>
            </ul>
          </div>

          <div className="p-5 bg-neutral-50 rounded-xl border-2 border-black space-y-2">
            <h4 className="font-bold text-black text-base sm:text-lg">Training Configuration:</h4>
            <ul className="list-disc pl-5 space-y-1.5 text-neutral-700">
              <li><strong>Input Resolution:</strong> 224 x 224 RGB, standard mean & std normalization.</li>
              <li><strong>Optimization Schedule:</strong> 25 epochs with Cosine Annealing learning rate schedule.</li>
              <li><strong>Model Selection:</strong> Epoch 20 selected with Exponential Moving Average (EMA) weights.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 3. Reported Evaluation Results */}
      <section className="bg-white rounded-3xl p-6 sm:p-8 border-2 border-black shadow-xs space-y-6" style={{ backgroundColor: '#ffffff' }}>
        <div>
          <span className="text-sm font-bold text-black bg-neutral-100 px-3 py-0.5 rounded-full border border-black">
            Validated Metrics
          </span>
          <h3 className="text-2xl font-black text-black mt-2">Model Performance on Held-Out Test Set</h3>
          <p className="text-sm text-neutral-600">Evaluated on the 1,016 test images strictly kept untouched during model training.</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {metrics.map((m) => (
            <div key={m.label} className="p-5 bg-neutral-50 rounded-2xl border-2 border-black text-center">
              <div className="text-3xl sm:text-4xl font-black text-black font-mono">{m.value}</div>
              <div className="text-sm font-bold text-black mt-1.5">{m.label}</div>
              <div className="text-xs text-neutral-600 mt-1 leading-snug">{m.description}</div>
            </div>
          ))}
        </div>

        {/* Per-class Recall Table */}
        <div className="mt-6">
          <h4 className="text-base font-bold text-black mb-3">Class-wise Test Set Recall & Sensitivity</h4>
          <div className="overflow-x-auto rounded-xl border-2 border-black">
            <table className="w-full text-sm text-left">
              <thead className="bg-neutral-100 text-black font-bold border-b-2 border-black">
                <tr>
                  <th className="py-3 px-4">Class Code</th>
                  <th className="py-3 px-4">Diagnostic Category</th>
                  <th className="py-3 px-4">Pathology Nature</th>
                  <th className="py-3 px-4 text-right">Test Support</th>
                  <th className="py-3 px-4 text-right">Recall</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-200">
                {perClassRecalls.map((row) => (
                  <tr key={row.code} className="hover:bg-neutral-50">
                    <td className="py-3 px-4 font-mono font-bold text-black">{row.code}</td>
                    <td className="py-3 px-4 font-bold text-black">{row.name}</td>
                    <td className="py-3 px-4 text-neutral-700">{row.type}</td>
                    <td className="py-3 px-4 font-mono text-neutral-800 text-right font-medium">{row.support}</td>
                    <td className="py-3 px-4 font-mono font-black text-black text-right">{row.recall}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* 4. Clinical Limitations */}
      <section className="bg-neutral-50 border-2 border-black rounded-3xl p-6 sm:p-8 space-y-4">
        <div className="flex items-center space-x-3 text-black">
          <ShieldAlert className="w-7 h-7 text-black" />
          <h3 className="text-2xl font-black text-black">Known Limitations of the AI Model</h3>
        </div>
        <ul className="list-disc pl-5 space-y-2 text-sm sm:text-base text-neutral-800 leading-relaxed">
          <li><strong>Domain Shift:</strong> The model was trained predominantly on dermatoscopic lenses. Performance on non-dermoscopic smartphone camera photos or darker skin tones may vary.</li>
          <li><strong>Softmax Calibration:</strong> Raw softmax confidence values represent network relative layer excitation, not calibrated probabilities of disease.</li>
          <li><strong>Absence of Clinical Metadata:</strong> The model operates purely on pixel inputs without factoring in patient age, anatomical location, lesion history, or familial risk factors.</li>
          <li><strong>Histological Ground Truth:</strong> Visual inspection alone is never definitive. Full surgical excision and dermatopathological examination remain the definitive gold standard.</li>
        </ul>
      </section>
    </div>
  );
}
