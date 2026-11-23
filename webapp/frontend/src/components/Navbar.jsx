import React from 'react';
import { ShieldCheck, Brain, UploadCloud } from './Icons';

export default function Navbar({ activeTab, setActiveTab }) {
  return (
    <header className="sticky top-0 z-50 bg-white border-b-2 border-black shadow-xs py-1" style={{ backgroundColor: '#ffffff' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-18">
          {/* Logo & Brand */}
          <div 
            className="flex items-center space-x-3 cursor-pointer group"
            onClick={() => setActiveTab('home')}
          >
            <div className="w-11 h-11 rounded-xl bg-neutral-100 border-2 border-black flex items-center justify-center text-neutral-800 group-hover:scale-105 transition-transform shadow-xs">
              <ShieldCheck className="w-7 h-7 text-neutral-800" />
            </div>
            <div>
              <span className="font-black text-2xl text-black tracking-tight">
                Skin Lesion Classification
              </span>
            </div>
          </div>

          {/* Navigation Links - Larger Font Size */}
          <nav className="flex items-center space-x-3">
            <button
              onClick={() => setActiveTab('home')}
              className={`px-4 py-2.5 rounded-xl text-base font-semibold transition-colors cursor-pointer ${
                activeTab === 'home'
                  ? 'text-white bg-black'
                  : 'text-neutral-700 hover:text-black hover:bg-neutral-100'
              }`}
            >
              Home
            </button>
            <button
              onClick={() => setActiveTab('upload')}
              className={`px-4 py-2.5 rounded-xl text-base font-semibold transition-colors cursor-pointer ${
                activeTab === 'upload' || activeTab === 'results'
                  ? 'text-white bg-black'
                  : 'text-neutral-700 hover:text-black hover:bg-neutral-100'
              }`}
            >
              <span className="flex items-center space-x-2">
                <UploadCloud className="w-5 h-5 text-neutral-500" />
                <span>Upload & Predict</span>
              </span>
            </button>
            <button
              onClick={() => setActiveTab('about')}
              className={`px-4 py-2.5 rounded-xl text-base font-semibold transition-colors cursor-pointer ${
                activeTab === 'about'
                  ? 'text-white bg-black'
                  : 'text-neutral-700 hover:text-black hover:bg-neutral-100'
              }`}
            >
              <span className="flex items-center space-x-2">
                <Brain className="w-5 h-5 text-neutral-500" />
                <span>Model & HAM10000</span>
              </span>
            </button>
          </nav>
        </div>
      </div>
    </header>
  );
}
