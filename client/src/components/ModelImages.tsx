import React from 'react';

interface ModelImagesProps {
  onClose: () => void;
}

const ModelImages: React.FC<ModelImagesProps> = ({ onClose }) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75 transition-opacity duration-300">
      <div className="relative w-full max-w-7xl p-8 bg-gray-900 rounded-2xl shadow-2xl transform transition-all duration-300 scale-95 hover:scale-100 overflow-y-auto max-h-[90vh] no-scrollbar">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors duration-300"
        >
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
        <h2 className="text-3xl font-extrabold text-white mb-6 text-center">Model Performance Analysis</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
            <h3 className="text-xl font-bold text-white mb-4 text-center">Learning Curves</h3>
            <img 
              src="/learning_curves.png" 
              alt="Learning Curves" 
              className="w-full h-auto rounded-lg border-2 border-gray-700 cursor-pointer"
              onClick={() => window.open('/learning_curves.png', '_blank')}
            />
            <p className="text-gray-300 mt-4 text-sm">
              Shows how the model's performance improves during training. The blue line represents training accuracy, 
              and the green line represents validation accuracy. Close curves indicate good fit.
            </p>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
            <h3 className="text-xl font-bold text-white mb-4 text-center">Feature Importance</h3>
            <img 
              src="/feature_importance.png" 
              alt="Feature Importance" 
              className="w-full h-auto rounded-lg border-2 border-gray-700 cursor-pointer"
              onClick={() => window.open('/feature_importance.png', '_blank')}
            />
            <p className="text-gray-300 mt-4 text-sm">
              Shows which features have the strongest influence on the model's predictions. 
              Larger values indicate more important features in determining diabetes risk.
            </p>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
            <h3 className="text-xl font-bold text-white mb-4 text-center">Calibration Curve</h3>
            <img 
              src="/calibration_curve.png" 
              alt="Calibration Curve" 
              className="w-full h-auto rounded-lg border-2 border-gray-700 cursor-pointer"
              onClick={() => window.open('/calibration_curve.png', '_blank')}
            />
            <p className="text-gray-300 mt-4 text-sm">
              Shows how well the predicted probabilities match actual probabilities. 
              Closer to the diagonal line means better calibrated predictions.
            </p>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
            <h3 className="text-xl font-bold text-white mb-4 text-center">Decision Boundary</h3>
            <img 
              src="/decision_boundary.png" 
              alt="Decision Boundary" 
              className="w-full h-auto rounded-lg border-2 border-gray-700 cursor-pointer"
              onClick={() => window.open('/decision_boundary.png', '_blank')}
            />
            <p className="text-gray-300 mt-4 text-sm">
              Visualizes how the model makes decisions using the top 2 most important features. 
              Different colors represent different predicted outcomes.
            </p>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
            <h3 className="text-xl font-bold text-white mb-4 text-center">Confusion Matrix</h3>
            <img 
              src="/confusion_matrix.png" 
              alt="Confusion Matrix" 
              className="w-full h-auto rounded-lg border-2 border-gray-700 cursor-pointer"
              onClick={() => window.open('/confusion_matrix.png', '_blank')}
            />
            <p className="text-gray-300 mt-4 text-sm">
              Shows the model's prediction accuracy across different classes. 
              True positives and true negatives are shown on the diagonal.
            </p>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
            <h3 className="text-xl font-bold text-white mb-4 text-center">ROC Curve</h3>
            <img 
              src="/roc_curve.png" 
              alt="ROC Curve" 
              className="w-full h-auto rounded-lg border-2 border-gray-700 cursor-pointer"
              onClick={() => window.open('/roc_curve.png', '_blank')}
            />
            <p className="text-gray-300 mt-4 text-sm">
              Shows the trade-off between true positive rate and false positive rate. 
              A larger area under the curve (AUC) indicates better model performance.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelImages; 