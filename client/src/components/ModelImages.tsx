import React from 'react';

interface ModelImagesProps {
  onClose: () => void;
  selectedModel: 'logreg' | 'xgboost' | 'knn' | 'randomforest';
  logregAccuracy?: number | null;
  logregAuc?: number | null;
  logregSensitivity?: number | null;
  logregSpecificity?: number | null;
  xgboostAccuracy?: number | null;
  xgboostAuc?: number | null;
  xgboostSensitivity?: number | null;
  xgboostSpecificity?: number | null;
  randomforestAccuracy?: number | null;
  randomforestAuc?: number | null;
  randomforestSensitivity?: number | null;
  randomforestSpecificity?: number | null;
  knnAccuracy?: number | null;
  knnAuc?: number | null;
  knnSensitivity?: number | null;
  knnSpecificity?: number | null;
}

const ModelImages: React.FC<ModelImagesProps> = ({ onClose, selectedModel, logregAccuracy, logregAuc, logregSensitivity, logregSpecificity, xgboostAccuracy, xgboostAuc, xgboostSensitivity, xgboostSpecificity, randomforestAccuracy, randomforestAuc, randomforestSensitivity, randomforestSpecificity, knnAccuracy, knnAuc, knnSensitivity, knnSpecificity }) => {
  const getImagePaths = (model: string) => {
    switch (model) {
      case 'xgboost':
        return {
          featureImportance: '/xgboost_feature_importance.png',
          calibrationCurve: '/xgboost_calibration_curve.png',
          confusionMatrix: '/xgboost_confusion_matrix.png',
          rocCurve: '/xgboost_roc_curve.png'
        };
      case 'knn':
        return {
          featureImportance: '/knn_feature_importance.png',
          calibrationCurve: '/knn_calibration_curve.png',
          confusionMatrix: '/knn_confusion_matrix.png',
          rocCurve: '/knn_roc_curve.png'
        };
      case 'randomforest':
        return {
          featureImportance: '/randomforest_feature_importance.png',
          calibrationCurve: '/randomforest_calibration_curve.png',
          confusionMatrix: '/randomforest_confusion_matrix.png',
          rocCurve: '/randomforest_roc_curve.png'
        };
      default: // logreg
        return {
          featureImportance: '/logreg_feature_importance.png',
          calibrationCurve: '/logreg_calibration_curve.png',
          confusionMatrix: '/logreg_confusion_matrix.png',
          rocCurve: '/logreg_roc_curve.png'
        };
    }
  };

  const imagePaths = getImagePaths(selectedModel);
  
  const getModelName = (model: string) => {
    switch (model) {
      case 'logreg': return 'Logistic Regression';
      case 'xgboost': return 'XGBoost';
      case 'knn': return 'K-Nearest Neighbors';
      case 'randomforest': return 'Random Forest';
      default: return 'Logistic Regression';
    }
  };

  const getAccuracyDisplay = () => {
    if (selectedModel === 'xgboost' && xgboostAccuracy !== null && xgboostAccuracy !== undefined) {
      return (
        <div className="mb-6 p-4 bg-gradient-to-r from-purple-600 to-indigo-600 rounded-lg">
          <h3 className="text-xl font-bold text-white mb-2">{getModelName(selectedModel)} Performance Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-white">{(xgboostAccuracy * 100).toFixed(1)}%</div>
              <div className="text-purple-200">Accuracy</div>
            </div>
            {xgboostAuc !== null && xgboostAuc !== undefined && (
              <div className="text-center">
                <div className="text-3xl font-bold text-white">{(xgboostAuc * 100).toFixed(1)}%</div>
                <div className="text-purple-200">AUC Score</div>
              </div>
            )}
            {xgboostSensitivity !== null && xgboostSensitivity !== undefined && (
              <div className="text-center">
                <div className="text-3xl font-bold text-white">{(xgboostSensitivity * 100).toFixed(1)}%</div>
                <div className="text-purple-200">Sensitivity</div>
              </div>
            )}
            {xgboostSpecificity !== null && xgboostSpecificity !== undefined && (
              <div className="text-center">
                <div className="text-3xl font-bold text-white">{(xgboostSpecificity * 100).toFixed(1)}%</div>
                <div className="text-purple-200">Specificity</div>
              </div>
            )}
          </div>
        </div>
      );
    } else if (selectedModel === 'logreg' && logregAccuracy !== null && logregAccuracy !== undefined) {
      return (
        <div className="mb-6 p-4 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-lg">
          <h3 className="text-xl font-bold text-white mb-2">{getModelName(selectedModel)} Performance Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-white">{(logregAccuracy * 100).toFixed(1)}%</div>
              <div className="text-blue-200">Accuracy</div>
            </div>
            {logregAuc !== null && logregAuc !== undefined && (
              <div className="text-center">
                <div className="text-3xl font-bold text-white">{(logregAuc * 100).toFixed(1)}%</div>
                <div className="text-blue-200">AUC Score</div>
              </div>
            )}
            {logregSensitivity !== null && logregSensitivity !== undefined && (
              <div className="text-center">
                <div className="text-3xl font-bold text-white">{(logregSensitivity * 100).toFixed(1)}%</div>
                <div className="text-blue-200">Sensitivity</div>
              </div>
            )}
            {logregSpecificity !== null && logregSpecificity !== undefined && (
              <div className="text-center">
                <div className="text-3xl font-bold text-white">{(logregSpecificity * 100).toFixed(1)}%</div>
                <div className="text-blue-200">Specificity</div>
              </div>
            )}
          </div>
        </div>
      );
    } else if (selectedModel === 'randomforest' && randomforestAccuracy !== null && randomforestAccuracy !== undefined) {
      return (
        <div className="mb-6 p-4 bg-gradient-to-r from-yellow-600 to-orange-600 rounded-lg">
          <h3 className="text-xl font-bold text-white mb-2">{getModelName(selectedModel)} Performance Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-white">{(randomforestAccuracy * 100).toFixed(1)}%</div>
              <div className="text-yellow-200">Accuracy</div>
            </div>
            {randomforestAuc !== null && randomforestAuc !== undefined && (
              <div className="text-center">
                <div className="text-3xl font-bold text-white">{(randomforestAuc * 100).toFixed(1)}%</div>
                <div className="text-yellow-200">AUC Score</div>
              </div>
            )}
            {randomforestSensitivity !== null && randomforestSensitivity !== undefined && (
              <div className="text-center">
                <div className="text-3xl font-bold text-white">{(randomforestSensitivity * 100).toFixed(1)}%</div>
                <div className="text-yellow-200">Sensitivity</div>
              </div>
            )}
            {randomforestSpecificity !== null && randomforestSpecificity !== undefined && (
              <div className="text-center">
                <div className="text-3xl font-bold text-white">{(randomforestSpecificity * 100).toFixed(1)}%</div>
                <div className="text-yellow-200">Specificity</div>
              </div>
            )}
          </div>
        </div>
      );
    } else if (selectedModel === 'knn' && knnAccuracy !== null && knnAccuracy !== undefined) {
      return (
        <div className="mb-6 p-4 bg-gradient-to-r from-green-600 to-emerald-600 rounded-lg">
          <h3 className="text-xl font-bold text-white mb-2">{getModelName(selectedModel)} Performance Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-white">{(knnAccuracy * 100).toFixed(1)}%</div>
              <div className="text-green-200">Accuracy</div>
            </div>
            {knnAuc !== null && knnAuc !== undefined && (
              <div className="text-center">
                <div className="text-3xl font-bold text-white">{(knnAuc * 100).toFixed(1)}%</div>
                <div className="text-green-200">AUC Score</div>
              </div>
            )}
            {knnSensitivity !== null && knnSensitivity !== undefined && (
              <div className="text-center">
                <div className="text-3xl font-bold text-white">{(knnSensitivity * 100).toFixed(1)}%</div>
                <div className="text-green-200">Sensitivity</div>
              </div>
            )}
            {knnSpecificity !== null && knnSpecificity !== undefined && (
              <div className="text-center">
                <div className="text-3xl font-bold text-white">{(knnSpecificity * 100).toFixed(1)}%</div>
                <div className="text-green-200">Specificity</div>
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };
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
        <h2 className="text-3xl font-extrabold text-white mb-6 text-center">
          {getModelName(selectedModel)} Performance Analysis
        </h2>
        {getAccuracyDisplay()}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
            <h3 className="text-xl font-bold text-white mb-4 text-center">Feature Importance</h3>
            <img 
              src={imagePaths.featureImportance} 
              alt="Feature Importance" 
              className="w-full h-auto rounded-lg border-2 border-gray-700 cursor-pointer"
              onClick={() => window.open(imagePaths.featureImportance, '_blank')}
            />
            <p className="text-gray-300 mt-4 text-sm">
              Shows which features have the strongest influence on the model's predictions. 
              Larger values indicate more important features in determining diabetes risk.
            </p>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
            <h3 className="text-xl font-bold text-white mb-4 text-center">Calibration Curve</h3>
            <img 
              src={imagePaths.calibrationCurve} 
              alt="Calibration Curve" 
              className="w-full h-auto rounded-lg border-2 border-gray-700 cursor-pointer"
              onClick={() => window.open(imagePaths.calibrationCurve, '_blank')}
            />
            <p className="text-gray-300 mt-4 text-sm">
              Shows how well the predicted probabilities match actual probabilities. 
              Closer to the diagonal line means better calibrated predictions.
            </p>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
            <h3 className="text-xl font-bold text-white mb-4 text-center">Confusion Matrix</h3>
            <img 
              src={imagePaths.confusionMatrix} 
              alt="Confusion Matrix" 
              className="w-full h-auto rounded-lg border-2 border-gray-700 cursor-pointer"
              onClick={() => window.open(imagePaths.confusionMatrix, '_blank')}
            />
            <p className="text-gray-300 mt-4 text-sm">
              Shows the model's prediction accuracy across different classes. 
              True positives and true negatives are shown on the diagonal.
            </p>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
            <h3 className="text-xl font-bold text-white mb-4 text-center">ROC Curve</h3>
            <img 
              src={imagePaths.rocCurve} 
              alt="ROC Curve" 
              className="w-full h-auto rounded-lg border-2 border-gray-700 cursor-pointer"
              onClick={() => window.open(imagePaths.rocCurve, '_blank')}
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