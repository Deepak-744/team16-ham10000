import React from 'react';
import { Info, AlertCircle, Eye, Compass, Palette, Maximize2, Clock } from './Icons';

export default function ABCDChecklist() {
  const criteria = [
    {
      letter: 'A',
      title: 'Asymmetry',
      icon: Compass,
      description: 'One half of the spot does not visually reflect the other half in shape, structure, or contour.',
      benignExample: 'Symmetrical circular or oval profile.',
      suspiciousExample: 'Irregular contour where axes differ substantially.'
    },
    {
      letter: 'B',
      title: 'Border Irregularity',
      icon: Eye,
      description: 'The perimeter of the lesion is notched, scalloped, ragged, blurred, or poorly demarcated.',
      benignExample: 'Smooth, clearly defined margin.',
      suspiciousExample: 'Frayed edges fading indistinctly into surrounding skin.'
    },
    {
      letter: 'C',
      title: 'Color Variation',
      icon: Palette,
      description: 'The pigmentation is non-uniform, exhibiting multiple variegated hues: shades of tan, brown, black, red, white, or slate blue.',
      benignExample: 'Single uniform shade of light or medium brown.',
      suspiciousExample: 'Variegated pigmentation with black, red, or depigmented zones.'
    },
    {
      letter: 'D',
      title: 'Diameter (> 6mm)',
      icon: Maximize2,
      description: 'Lesion dimension exceeds 6 millimeters (roughly the size of a standard pencil eraser), though early melanomas may be smaller.',
      benignExample: 'Small discrete papule or macule < 5mm.',
      suspiciousExample: 'Expanding patch or plaque measuring > 6mm.'
    },
    {
      letter: 'E',
      title: 'Evolving / Changing',
      icon: Clock,
      description: 'Any perceptible change in size, shape, pigmentation, elevation, or novel clinical symptoms (bleeding, persistent itching, crusting).',
      benignExample: 'Stable appearance documented over years.',
      suspiciousExample: 'Rapid evolution over weeks or recent onset of bleeding.'
    }
  ];

  return (
    <div className="max-w-5xl mx-auto py-8 px-4 sm:px-6 bg-white text-black space-y-8">
      {/* Header */}
      <div className="text-center max-w-2xl mx-auto">
        <span className="px-3 py-1 rounded-full text-xs font-semibold bg-neutral-100 text-black border border-black">
          Dermatological Education
        </span>
        <h2 className="text-3xl font-extrabold text-black tracking-tight mt-3">
          The ABCDE Rule of Skin Lesions
        </h2>
        <p className="text-neutral-600 mt-2 text-sm sm:text-base leading-relaxed">
          Standard clinical mnemonic endorsed by the American Academy of Dermatology to educate patients on warning signs of suspicious pigmented lesions.
        </p>
      </div>

      {/* Mandatory Clinical Notice */}
      <div className="p-4 bg-neutral-50 border-2 border-black rounded-2xl flex items-start space-x-3 text-xs sm:text-sm text-neutral-800 shadow-xs">
        <AlertCircle className="w-5 h-5 text-black shrink-0 mt-0.5" />
        <div>
          <span className="font-bold text-black">Educational Notice:</span> This checklist is solely intended for general public education and personal awareness. It is not an algorithmic diagnostic tool and cannot substitute for a full-body skin examination performed by a board-certified dermatologist.
        </div>
      </div>

      {/* Grid of ABCDE Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {criteria.map((item) => {
          const Icon = item.icon;
          return (
            <div key={item.letter} className="bg-white rounded-2xl p-6 border-2 border-black shadow-xs flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-4">
                  <span className="w-12 h-12 rounded-xl bg-neutral-100 border-2 border-black flex items-center justify-center text-black font-extrabold text-xl shadow-xs">
                    {item.letter}
                  </span>
                  <div className="p-2 bg-neutral-50 rounded-lg border border-black text-neutral-700">
                    <Icon className="w-5 h-5 text-neutral-700" />
                  </div>
                </div>

                <h3 className="text-lg font-bold text-black mb-2">{item.title}</h3>
                <p className="text-sm text-neutral-600 leading-relaxed mb-4">{item.description}</p>
              </div>

              <div className="pt-4 border-t border-neutral-200 space-y-2 text-xs">
                <div className="flex items-start space-x-2">
                  <span className="font-semibold text-emerald-900 bg-emerald-50 px-1.5 py-0.5 rounded border border-black shrink-0">Typical:</span>
                  <span className="text-neutral-700">{item.benignExample}</span>
                </div>
                <div className="flex items-start space-x-2">
                  <span className="font-semibold text-rose-900 bg-rose-50 px-1.5 py-0.5 rounded border border-black shrink-0">Warning:</span>
                  <span className="text-neutral-700">{item.suspiciousExample}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
