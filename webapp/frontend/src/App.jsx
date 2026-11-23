import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import AboutPage from './pages/AboutPage';
import { checkBackendHealth, getSupportedClasses, getSampleImages } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [backendStatus, setBackendStatus] = useState(null);
  const [classes, setClasses] = useState([]);
  const [sampleImages, setSampleImages] = useState([]);
  const [predictionResult, setPredictionResult] = useState(null);

  useEffect(() => {
    // Force pure light mode and clean any residual dark mode class/storage
    document.documentElement.className = '';
    document.documentElement.classList.remove('dark');
    document.documentElement.style.backgroundColor = '#ffffff';
    document.body.style.backgroundColor = '#ffffff';
    try {
      localStorage.removeItem('skin_ai_theme');
    } catch (e) {}

    // Initial health check
    const pollHealth = async () => {
      const status = await checkBackendHealth();
      setBackendStatus(status);
    };
    pollHealth();

    // Fetch classes and sample images
    const loadMetadata = async () => {
      try {
        const clsData = await getSupportedClasses();
        if (clsData?.classes) setClasses(clsData.classes);
      } catch (err) {
        console.warn('Classes fetch error:', err);
      }

      try {
        const samplesData = await getSampleImages();
        if (samplesData?.samples) setSampleImages(samplesData.samples);
      } catch (err) {
        console.warn('Samples fetch error:', err);
      }
    };
    loadMetadata();

    // Periodic heartbeat every 30s
    const interval = setInterval(pollHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handlePredictionSuccess = (result) => {
    setPredictionResult(result);
    setActiveTab('results');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleReset = () => {
    setPredictionResult(null);
    setActiveTab('upload');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen flex flex-col bg-white text-black font-sans text-base" style={{ backgroundColor: '#ffffff', color: '#000000' }}>
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
      />

      <main className="flex-1 pb-16 bg-white" style={{ backgroundColor: '#ffffff' }}>
        {activeTab === 'home' && (
          <HomePage setActiveTab={setActiveTab} classes={classes} />
        )}

        {activeTab === 'upload' && (
          <UploadPage 
            onPredictionSuccess={handlePredictionSuccess} 
            sampleImages={sampleImages}
          />
        )}

        {activeTab === 'results' && (
          <ResultsPage 
            predictionResult={predictionResult} 
            onReset={handleReset} 
            setActiveTab={setActiveTab}
          />
        )}

        {activeTab === 'about' && (
          <AboutPage classes={classes} />
        )}
      </main>
    </div>
  );
}
