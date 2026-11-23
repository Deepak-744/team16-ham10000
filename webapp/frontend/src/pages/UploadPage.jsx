import React, { useEffect, useState, useRef } from 'react';
import { 
  UploadCloud, X, Sparkles, AlertCircle, 
  Loader2, FileCheck
} from '../components/Icons';
import { predictSkinLesion, waitForPredictionServer } from '../services/api';

export default function UploadPage({ onPredictionSuccess, sampleImages = [] }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState('Initializing...');
  const [generateGradCAM, setGenerateGradCAM] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);
  const predictionRequestRef = useRef(null);

  useEffect(() => () => predictionRequestRef.current?.abort(), []);

  const validateAndSetFile = (file) => {
    setErrorMsg(null);
    if (!file) return;

    // Check size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
      setErrorMsg("File size exceeds 10MB limit. Please upload a smaller image.");
      return;
    }

    // Check format
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type.toLowerCase())) {
      setErrorMsg("Unsupported image format. Please select a JPG, JPEG, or PNG file.");
      return;
    }

    setSelectedFile(file);
    const reader = new FileReader();
    reader.onload = () => setPreviewUrl(reader.result);
    reader.readAsDataURL(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleSampleSelect = async (sample) => {
    try {
      setErrorMsg(null);
      // Convert base64 to Blob
      const res = await fetch(sample.image_base64);
      const blob = await res.blob();
      const file = new File([blob], sample.filename, { type: "image/jpeg" });
      setSelectedFile(file);
      setPreviewUrl(sample.image_base64);
    } catch (e) {
      setErrorMsg("Failed to load sample image: " + e.message);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile || predictionRequestRef.current) return;

    const request = new AbortController();
    predictionRequestRef.current = request;

    setIsLoading(true);
    setErrorMsg(null);
    setLoadingStage('Waiting for the prediction server to be ready…');

    try {
      await waitForPredictionServer(request.signal);
      setLoadingStage(generateGradCAM ? 'Running classification and generating Grad-CAM…' : 'Running classification…');

      const result = await predictSkinLesion(selectedFile, generateGradCAM, request.signal);
      onPredictionSuccess(result, previewUrl);
    } catch (err) {
      if (!request.signal.aborted) setErrorMsg(err.message || 'Prediction failed. Please try again.');
    } finally {
      predictionRequestRef.current = null;
      if (!request.signal.aborted) setIsLoading(false);
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setErrorMsg(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="max-w-4xl mx-auto py-8 px-4 sm:px-6 space-y-8 bg-white text-black" style={{ backgroundColor: '#ffffff', color: '#000000' }}>
      {/* Title */}
      <div className="text-center max-w-xl mx-auto">
        <h2 className="text-3xl sm:text-4xl font-black text-black tracking-tight">Upload Skin Lesion Image</h2>
        <p className="text-base sm:text-lg text-neutral-700 mt-2">
          Provide a close-up dermoscopic or macro skin lesion photograph for automated multi-class classification.
        </p>
      </div>

      {/* Error Banner */}
      {errorMsg && (
        <div className="p-4 bg-rose-50 border-2 border-black rounded-xl flex items-start space-x-3 text-base text-rose-900">
          <AlertCircle className="w-6 h-6 text-rose-700 shrink-0 mt-0.5" />
          <div className="flex-1 font-semibold">{errorMsg}</div>
          <button onClick={() => setErrorMsg(null)} className="text-black hover:text-neutral-600 cursor-pointer">
            <X className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* Upload Box / Preview - Pure White Background with Black Outer Border */}
      <div className="bg-white rounded-3xl p-6 sm:p-8 border-2 border-black shadow-xs" style={{ backgroundColor: '#ffffff' }}>
        {!previewUrl ? (
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{ backgroundColor: '#ffffff' }}
            className={`border-2 border-dashed rounded-2xl p-10 sm:p-14 text-center cursor-pointer transition-all flex flex-col items-center justify-center space-y-4 ${
              isDragOver 
                ? 'border-black bg-neutral-100 scale-[1.01]' 
                : 'border-neutral-400 hover:border-black hover:bg-neutral-50'
            }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => e.target.files && validateAndSetFile(e.target.files[0])}
              accept="image/jpeg,image/jpg,image/png"
              className="hidden"
            />
            <div className="w-18 h-18 rounded-2xl bg-neutral-100 border-2 border-black text-neutral-800 flex items-center justify-center shadow-xs">
              <UploadCloud className="w-9 h-9 text-neutral-800" />
            </div>
            <div>
              <p className="text-lg sm:text-xl font-bold text-black">
                Drag and drop your skin lesion image here, or <span className="underline font-black">browse files</span>
              </p>
              <p className="text-sm text-neutral-600 mt-1">Supports JPG, JPEG, and PNG (Max 10MB)</p>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="relative rounded-2xl overflow-hidden border-2 border-black bg-neutral-50 max-h-[400px] flex items-center justify-center">
              <img
                src={previewUrl}
                alt="Uploaded skin lesion"
                className="max-h-[400px] w-auto object-contain rounded-xl"
              />
              <button
                onClick={clearSelection}
                disabled={isLoading}
                className="absolute top-3 right-3 p-2.5 bg-black hover:bg-neutral-800 text-white rounded-full border border-black transition-colors shadow-xs cursor-pointer"
                title="Remove image"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex flex-col sm:flex-row sm:items-center justify-between p-4 bg-neutral-50 rounded-xl text-sm sm:text-base text-neutral-800 space-y-2 sm:space-y-0 border-2 border-black">
              <div className="flex items-center space-x-2.5">
                <FileCheck className="w-5 h-5 text-black" />
                <span className="font-bold text-black truncate max-w-xs">{selectedFile?.name || "Selected Image"}</span>
                <span className="text-neutral-600">({(selectedFile?.size ? (selectedFile.size / 1024).toFixed(0) : 0)} KB)</span>
              </div>
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
                className="text-black hover:text-neutral-700 font-bold underline text-sm cursor-pointer"
              >
                Change Image
              </button>
            </div>
          </div>
        )}

        {/* Options & Action Button */}
        <div className="mt-6 pt-6 border-t-2 border-black flex flex-col sm:flex-row items-center justify-between gap-4">
          <label className="flex items-center space-x-2.5 text-sm sm:text-base text-neutral-800 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={generateGradCAM}
              onChange={(e) => setGenerateGradCAM(e.target.checked)}
              disabled={isLoading}
              className="w-5 h-5 rounded text-black focus:ring-black border-2 border-black accent-black"
            />
            <span className="font-semibold text-black">Generate Grad-CAM saliency map visualization</span>
          </label>

          <button
            onClick={handlePredict}
            disabled={!selectedFile || isLoading}
            className={`w-full sm:w-auto px-9 py-3.5 rounded-xl font-bold text-base sm:text-lg text-white flex items-center justify-center space-x-2.5 transition-all cursor-pointer ${
              !selectedFile || isLoading
                ? 'bg-neutral-300 border-2 border-neutral-300 text-neutral-500 cursor-not-allowed shadow-none'
                : 'bg-black hover:bg-neutral-800 border-2 border-black shadow-xs'
            }`}
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin text-white" />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5 text-neutral-300" />
                <span>Run AI Classification</span>
              </>
            )}
          </button>
        </div>

        {/* Loading Progress Feedback */}
        {isLoading && (
          <div className="mt-4 p-4 bg-neutral-50 border-2 border-black rounded-xl text-center space-y-2">
            <div className="flex items-center justify-center space-x-2 text-sm font-bold text-black">
              <Loader2 className="w-5 h-5 animate-spin text-black" />
              <span>{loadingStage}</span>
            </div>
            <div className="w-full bg-neutral-200 border border-black h-2.5 rounded-full overflow-hidden">
              <div className="bg-black h-full rounded-full animate-pulse w-3/4" />
            </div>
          </div>
        )}
      </div>

      {/* Instant Testing with Sample Images */}
      {sampleImages && sampleImages.length > 0 && (
        <div className="bg-white rounded-3xl p-6 border-2 border-black shadow-xs" style={{ backgroundColor: '#ffffff' }}>
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="font-bold text-black text-base sm:text-lg">Quick Demo: Test with Sample Images</h3>
              <p className="text-sm text-neutral-600">Select any test image from the HAM10000 evaluation set to test instantly.</p>
            </div>
            <span className="text-xs sm:text-sm text-black font-bold bg-neutral-100 px-3 py-1 rounded-full border border-black">
              1-Click Demo
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
            {sampleImages.map((sample, idx) => (
              <button
                key={sample.filename || idx}
                onClick={() => handleSampleSelect(sample)}
                disabled={isLoading}
                className="group relative rounded-xl overflow-hidden border-2 border-black p-2 transition-all bg-neutral-50 hover:bg-neutral-100 text-left cursor-pointer"
              >
                <div className="h-28 w-full rounded-lg overflow-hidden bg-neutral-200 border border-neutral-300 mb-2">
                  <img
                    src={sample.image_base64}
                    alt={sample.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                  />
                </div>
                <div className="text-xs font-mono font-bold text-black truncate px-1">
                  {sample.filename}
                </div>
                <div className="text-[11px] text-neutral-600 px-1 font-medium">Click to select</div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
