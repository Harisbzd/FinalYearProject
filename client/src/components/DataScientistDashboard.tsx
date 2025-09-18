import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import DiabetesPredictionInputForm from './DiabetesPredictionInputForm';
import PatientInputForm from './PatientInputForm';
import ModelImages from './ModelImages';
import StyledButton from './StyledButton';

const DataScientistDashboard: React.FC = () => {
  const { token } = useAuth();
  const [, setPrediction] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [, setError] = useState<string | null>(null);
  const [predictionExists, setPredictionExists] = useState(false);
  const [predictionInput, setPredictionInput] = useState<any>(null);
  const [showPatientInput, setShowPatientInput] = useState(false);
  const [showModelImages, setShowModelImages] = useState(false);
  const [selectedModel, setSelectedModel] = useState<'logreg' | 'xgboost' | 'knn' | 'randomforest'>('logreg');
  
  // Model results
  const [logregResult, setLogregResult] = useState<string | null>(null);
  const [logregAccuracy, setLogregAccuracy] = useState<number | null>(null);
  const [logregAuc, setLogregAuc] = useState<number | null>(null);
  const [logregSensitivity, setLogregSensitivity] = useState<number | null>(null);
  const [logregSpecificity, setLogregSpecificity] = useState<number | null>(null);
  const [logregFetchedInput, setLogregFetchedInput] = useState<any>(null);
  
  const [xgboostResult, setXgboostResult] = useState<string | null>(null);
  const [xgboostAccuracy, setXgboostAccuracy] = useState<number | null>(null);
  const [xgboostAuc, setXgboostAuc] = useState<number | null>(null);
  const [xgboostSensitivity, setXgboostSensitivity] = useState<number | null>(null);
  const [xgboostSpecificity, setXgboostSpecificity] = useState<number | null>(null);
  const [xgboostFetchedInput, setXgboostFetchedInput] = useState<any>(null);
  
  const [randomforestResult, setRandomforestResult] = useState<string | null>(null);
  const [randomforestAccuracy, setRandomforestAccuracy] = useState<number | null>(null);
  const [randomforestAuc, setRandomforestAuc] = useState<number | null>(null);
  const [randomforestSensitivity, setRandomforestSensitivity] = useState<number | null>(null);
  const [randomforestSpecificity, setRandomforestSpecificity] = useState<number | null>(null);
  const [randomforestFetchedInput, setRandomforestFetchedInput] = useState<any>(null);
  
  const [knnResult, setKnnResult] = useState<string | null>(null);
  const [knnAccuracy, setKnnAccuracy] = useState<number | null>(null);
  const [knnAuc, setKnnAuc] = useState<number | null>(null);
  const [knnSensitivity, setKnnSensitivity] = useState<number | null>(null);
  const [knnSpecificity, setKnnSpecificity] = useState<number | null>(null);
  const [knnFetchedInput, setKnnFetchedInput] = useState<any>(null);

  const getCurrentResult = () => {
    if (selectedModel === 'logreg' && logregResult !== null) {
      return {
        result: logregResult,
        accuracy: logregAccuracy,
        auc: logregAuc,
        sensitivity: logregSensitivity,
        specificity: logregSpecificity,
        modelName: 'Logistic Regression',
        input: logregFetchedInput
      };
    } else if (selectedModel === 'xgboost' && xgboostResult !== null) {
      return {
        result: xgboostResult,
        accuracy: xgboostAccuracy,
        auc: xgboostAuc,
        sensitivity: xgboostSensitivity,
        specificity: xgboostSpecificity,
        modelName: 'XGBoost',
        input: xgboostFetchedInput
      };
    } else if (selectedModel === 'knn' && knnResult !== null) {
      return {
        result: knnResult,
        accuracy: knnAccuracy,
        auc: knnAuc,
        sensitivity: knnSensitivity,
        specificity: knnSpecificity,
        modelName: 'K-Nearest Neighbors',
        input: knnFetchedInput
      };
    } else if (selectedModel === 'randomforest' && randomforestResult !== null) {
      return {
        result: randomforestResult,
        accuracy: randomforestAccuracy,
        auc: randomforestAuc,
        sensitivity: randomforestSensitivity,
        specificity: randomforestSpecificity,
        modelName: 'Random Forest',
        input: randomforestFetchedInput
      };
    }
    return null;
  };

  const currentResult = getCurrentResult();

  useEffect(() => {
    const fetchPrediction = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch('/api/predict-diabetes', {
          method: 'GET',
          headers: {
            ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
          },
        });
        if (response.ok) {
          const result = await response.json();
          setPrediction(result.prediction);
          setPredictionInput(result.input);
          setLogregResult(result.prediction);
          setLogregAccuracy(result.accuracy);
          setLogregAuc(result.auc_score);
          setLogregSensitivity(result.sensitivity);
          setLogregSpecificity(result.specificity);
          setLogregFetchedInput(result.input);
          setPredictionExists(true);
        } else {
          setPrediction(null);
          setPredictionInput(null);
        }
      } catch (err) {
        setPrediction(null);
        setPredictionInput(null);
      } finally {
        setLoading(false);
      }
    };
    if (token) fetchPrediction();
  }, [token]);

  const handlePredictionSubmit = async (formData: any) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/predict-diabetes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        const result = await response.json();
        setPrediction(result.prediction);
        setPredictionInput(formData);
        setLogregResult(result.prediction);
        setLogregAccuracy(result.accuracy);
        setLogregAuc(result.auc_score);
        setLogregSensitivity(result.sensitivity);
        setLogregSpecificity(result.specificity);
        setLogregFetchedInput(formData);
        setPredictionExists(true);
      } else {
        const errorResult = await response.json();
        setError(errorResult.message || 'Failed to create prediction.');
      }
    } catch (err) {
      setError('Failed to create prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDeletePrediction = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/predict-diabetes', {
        method: 'DELETE',
        headers: {
          ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
        },
      });

      if (response.ok) {
        setPrediction(null);
        setPredictionInput(null);
        setLogregResult(null);
        setLogregAccuracy(null);
        setLogregAuc(null);
        setLogregSensitivity(null);
        setLogregSpecificity(null);
        setLogregFetchedInput(null);
        setXgboostResult(null);
        setXgboostAccuracy(null);
        setXgboostAuc(null);
        setXgboostSensitivity(null);
        setXgboostSpecificity(null);
        setXgboostFetchedInput(null);
        setRandomforestResult(null);
        setRandomforestAccuracy(null);
        setRandomforestAuc(null);
        setRandomforestSensitivity(null);
        setRandomforestSpecificity(null);
        setRandomforestFetchedInput(null);
        setKnnResult(null);
        setKnnAccuracy(null);
        setKnnAuc(null);
        setKnnSensitivity(null);
        setKnnSpecificity(null);
        setKnnFetchedInput(null);
        setPredictionExists(false);
      }
    } catch (err) {
      setError('Failed to delete prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleModelClick = async (modelType: 'logreg' | 'xgboost' | 'knn' | 'randomforest') => {
    setSelectedModel(modelType);
    setLoading(true);
    setError(null);
    
    try {
      let endpoint = '';
      switch (modelType) {
        case 'logreg':
          endpoint = '/api/predict-diabetes-logreg';
          break;
        case 'xgboost':
          endpoint = '/api/predict-diabetes-xgboost';
          break;
        case 'knn':
          endpoint = '/api/predict-diabetes-knn';
          break;
        case 'randomforest':
          endpoint = '/api/predict-diabetes-randomforest';
          break;
      }

      if (predictionInput) {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
          },
          body: JSON.stringify(predictionInput),
        });

        if (response.ok) {
          const result = await response.json();
          switch (modelType) {
            case 'logreg':
              setLogregResult(result.prediction);
              setLogregAccuracy(result.accuracy);
              setLogregAuc(result.auc_score);
              setLogregSensitivity(result.sensitivity);
              setLogregSpecificity(result.specificity);
              setLogregFetchedInput(predictionInput);
              break;
            case 'xgboost':
              setXgboostResult(result.prediction);
              setXgboostAccuracy(result.accuracy);
              setXgboostAuc(result.auc_score);
              setXgboostSensitivity(result.sensitivity);
              setXgboostSpecificity(result.specificity);
              setXgboostFetchedInput(predictionInput);
              break;
            case 'knn':
              setKnnResult(result.prediction);
              setKnnAccuracy(result.accuracy);
              setKnnAuc(result.auc_score);
              setKnnSensitivity(result.sensitivity);
              setKnnSpecificity(result.specificity);
              setKnnFetchedInput(predictionInput);
              break;
            case 'randomforest':
              setRandomforestResult(result.prediction);
              setRandomforestAccuracy(result.accuracy);
              setRandomforestAuc(result.auc_score);
              setRandomforestSensitivity(result.sensitivity);
              setRandomforestSpecificity(result.specificity);
              setRandomforestFetchedInput(predictionInput);
              break;
          }
        } else {
          setError(`Failed to get ${modelType} prediction.`);
        }
      } else {
        setError('No input data available. Please submit a prediction first.');
      }
    } catch (err) {
      setError(`Failed to get ${modelType} prediction. Please try again.`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div 
      className="min-h-screen flex items-center justify-center p-4 bg-cover bg-center bg-no-repeat relative"
      style={{
        backgroundImage: "url('/1.jpg')",
      }}
    >
      <div className="absolute inset-0 bg-black bg-opacity-40"></div>
      <div className="relative z-10 w-full max-w-6xl">
        {!predictionExists ? (
          /* Centered Form Layout */
          <div className="flex flex-col items-center">
            <div className="text-center mb-8">
              <h1 className="text-4xl font-extrabold mb-4 text-white">Data Scientist Dashboard</h1>
              <p className="text-lg text-gray-300 max-w-2xl mx-auto">
                Complete the health assessment form below to get diabetes risk predictions using different machine learning models.
              </p>
            </div>
            <DiabetesPredictionInputForm onSubmit={handlePredictionSubmit} />
          </div>
        ) : (
          /* Two Column Layout for Results */
          <div className="flex flex-col md:flex-row gap-8">
            {/* Left Side */}
            <div className="flex-1 flex flex-col justify-center items-center text-gray-100 md:pr-4 mb-8 md:mb-0">
              <h1 className="text-4xl font-extrabold mb-4 text-center md:text-left">Data Scientist Dashboard</h1>
              <p className="text-lg text-gray-300 text-center md:text-left max-w-md">
                Select a machine learning model to view predictions and performance metrics.
              </p>
              
              {currentResult && (
                <div className="mt-8 w-80 mx-auto">
                  <div className={`p-6 rounded-xl border-2 shadow-2xl ${
                    Number(currentResult.result) === 0 
                      ? 'bg-gradient-to-r from-green-600 to-emerald-600 border-green-500' 
                      : 'bg-gradient-to-r from-red-600 to-orange-600 border-red-500'
                  }`}>
                    <div className="text-center">
                      <div className="mb-4">
                        <div className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center mb-4 ${
                          Number(currentResult.result) === 0 
                            ? 'bg-green-500 shadow-lg' 
                            : 'bg-red-500 shadow-lg animate-pulse'
                        }`}>
                          <span className="text-white text-3xl font-bold">
                            {Number(currentResult.result) === 0 ? "✓" : "⚠"}
                          </span>
                        </div>
                        <h2 className={`text-4xl font-bold mb-2 ${
                          Number(currentResult.result) === 0 ? 'text-white' : 'text-white'
                        }`}>
                          {Number(currentResult.result) === 0 ? "Negative" : "Positive"}
                        </h2>
                        <p className={`text-lg ${
                          Number(currentResult.result) === 0 ? 'text-green-100' : 'text-red-100'
                        }`}>
                          {Number(currentResult.result) === 0 
                            ? "Low Risk of Diabetes" 
                            : "High Risk of Diabetes - Medical Attention Recommended"
                          }
                        </p>
                        <p className="text-sm text-white mt-2 opacity-80">
                          {currentResult.modelName} Model Prediction
                        </p>
                      </div>
                      
                      {currentResult.accuracy !== null && (
                        <div className="mt-6 p-4 bg-white bg-opacity-20 rounded-lg border border-white border-opacity-30">
                          <div className="text-sm text-white mb-2">Model Accuracy</div>
                          <div className="text-2xl font-bold text-white">
                            {(currentResult.accuracy * 100).toFixed(1)}%
                          </div>
                          <div className="w-full bg-white bg-opacity-30 rounded-full h-3 mt-2">
                            <div 
                              className={`h-3 rounded-full transition-all duration-500 ${
                                Number(currentResult.result) === 0 
                                  ? 'bg-green-400' 
                                  : 'bg-red-400'
                              }`}
                              style={{ width: `${(currentResult.accuracy * 100)}%` }}
                            ></div>
                          </div>
                          {currentResult.auc !== null && (
                            <div className="text-sm text-white mt-2">
                              AUC Score: {(currentResult.auc * 100).toFixed(1)}%
                            </div>
                          )}
                          {currentResult.sensitivity !== null && (
                            <div className="text-sm text-white mt-1">
                              Sensitivity: {(currentResult.sensitivity * 100).toFixed(1)}%
                            </div>
                          )}
                          {currentResult.specificity !== null && (
                            <div className="text-sm text-white mt-1">
                              Specificity: {(currentResult.specificity * 100).toFixed(1)}%
                            </div>
                          )}
                        </div>
                      )}
                      
                      {Number(currentResult.result) === 1 && (
                        <div className="mt-4 p-4 bg-red-500 bg-opacity-20 rounded-lg border border-red-400">
                          <div className="flex items-center justify-center text-red-200">
                            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                            </svg>
                            <span className="font-semibold">Please consult with a healthcare professional</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Right Side */}
            <div className="flex-1 flex flex-col justify-center items-center">
              <div className="flex flex-col gap-4 justify-center items-center">
                <div className="text-center mb-6">
                  <h3 className="text-2xl font-bold text-white mb-2">Model Selection</h3>
                  <p className="text-gray-300">Click on any model to view its prediction and performance metrics</p>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-4xl">
                  <div className="flex justify-center">
                    <StyledButton
                      onClick={() => handleModelClick('logreg')}
                      label="Logistic Regression"
                      color="blue"
                      icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2z" /></svg>}
                      disabled={loading}
                    />
                  </div>
                  <div className="flex justify-center">
                    <StyledButton
                      onClick={() => handleModelClick('xgboost')}
                      label="XGBoost"
                      color="purple"
                      icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>}
                      disabled={loading}
                    />
                  </div>
                  <div className="flex justify-center">
                    <StyledButton
                      onClick={() => handleModelClick('knn')}
                      label="K-Nearest Neighbors"
                      color="green"
                      icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" /></svg>}
                      disabled={loading}
                    />
                  </div>
                  <div className="flex justify-center">
                    <StyledButton
                      onClick={() => handleModelClick('randomforest')}
                      label="Random Forest"
                      color="yellow"
                      icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" /></svg>}
                      disabled={loading}
                    />
                  </div>
                </div>

                <div className="mt-6 flex flex-col items-center gap-4">
                  {currentResult && currentResult.input && (
                    <StyledButton
                      onClick={() => setShowPatientInput(true)}
                      label="View Patient Input Data"
                      color="blue"
                      icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>}
                      disabled={loading}
                    />
                  )}
                  <StyledButton
                    onClick={() => setShowModelImages(true)}
                    label="View Model Images"
                    color="purple"
                    icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>}
                    disabled={loading}
                  />
                  <StyledButton
                    onClick={handleDeletePrediction}
                    label="Delete Prediction"
                    color="red"
                    icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>}
                    disabled={loading}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {showPatientInput && currentResult && currentResult.input && (
        <PatientInputForm 
          data={currentResult.input} 
          onClose={() => setShowPatientInput(false)} 
        />
      )}
      {showModelImages && (
        <ModelImages
          onClose={() => setShowModelImages(false)}
          selectedModel={selectedModel}
          logregAccuracy={logregAccuracy}
          logregAuc={logregAuc}
          logregSensitivity={logregSensitivity}
          logregSpecificity={logregSpecificity}
          xgboostAccuracy={xgboostAccuracy}
          xgboostAuc={xgboostAuc}
          xgboostSensitivity={xgboostSensitivity}
          xgboostSpecificity={xgboostSpecificity}
          randomforestAccuracy={randomforestAccuracy}
          randomforestAuc={randomforestAuc}
          randomforestSensitivity={randomforestSensitivity}
          randomforestSpecificity={randomforestSpecificity}
          knnAccuracy={knnAccuracy}
          knnAuc={knnAuc}
          knnSensitivity={knnSensitivity}
          knnSpecificity={knnSpecificity}
        />
      )}
    </div>
  );
};

export default DataScientistDashboard;
