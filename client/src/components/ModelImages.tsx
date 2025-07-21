import React, { useState } from 'react';

interface ModelImagesProps {
  onClose: () => void;
}

const ModelImages: React.FC<ModelImagesProps> = ({ onClose }) => {
  const [zoomedImage, setZoomedImage] = useState<string | null>(null);

  const handleImageClick = (src: string) => {
    setZoomedImage(src);
  };

  const handleCloseZoom = () => {
    setZoomedImage(null);
  };

  return (
    <>
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75 transition-opacity duration-300">
        <div className="relative w-full max-w-4xl p-8 bg-gray-900 rounded-2xl shadow-2xl transform transition-all duration-300 scale-95 hover:scale-100">
          <button
            onClick={onClose}
            className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors duration-300"
          >
            <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
          <h2 className="text-3xl font-extrabold text-white mb-6 text-center">Model Performance</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
              <h3 className="text-xl font-bold text-white mb-4 text-center">Confusion Matrix</h3>
              <img
                src="/confusion_matrix.png"
                alt="Confusion Matrix"
                className="w-full h-auto rounded-lg border-2 border-gray-700 cursor-pointer"
                onClick={() => handleImageClick('/confusion_matrix.png')}
              />
            </div>
            <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
              <h3 className="text-xl font-bold text-white mb-4 text-center">ROC Curve</h3>
              <img
                src="/roc_curve.png"
                alt="ROC Curve"
                className="w-full h-auto rounded-lg border-2 border-gray-700 cursor-pointer"
                onClick={() => handleImageClick('/roc_curve.png')}
              />
            </div>
          </div>
        </div>
      </div>

      {zoomedImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-90"
          onClick={handleCloseZoom}
        >
          <img
            src={zoomedImage}
            alt="Zoomed"
            className="max-w-screen-lg max-h-screen-lg"
          />
        </div>
      )}
    </>
  );
};

export default ModelImages; 