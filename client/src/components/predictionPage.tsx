import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import DiabetesPredictionInputForm from './DiabetesPredictionInputForm';
import PatientInputForm from './PatientInputForm';
import ModelImages from './ModelImages';
import StyledButton from './StyledButton';

const PredictionPage: React.FC = () => {
  const { token } = useAuth();
  const [, setPrediction] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [, setError] = useState<string | null>(null);
  const [predictionExists, setPredictionExists] = useState(false);
  const [predictionInput, setPredictionInput] = useState<any>(null);
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
  const [showPatientInput, setShowPatientInput] = useState(false);
  const [showModelImages, setShowModelImages] = useState(false);
  const [selectedModel, setSelectedModel] = useState<'logreg' | 'xgboost' | 'knn' | 'randomforest'>('logreg');

  // Function to get model-specific content
  const getCurrentResult = () => {
    if (selectedModel === 'logreg' && logregResult !== null) {
      return {
        result: logregResult,
        accuracy: logregAccuracy,
        auc: logregAuc,
        sensitivity: logregSensitivity,
        specificity: logregSpecificity,
        modelName: 'XGBoost'
      };
    } else if (selectedModel === 'xgboost' && xgboostResult !== null) {
      return {
        result: xgboostResult,
        accuracy: xgboostAccuracy,
        auc: xgboostAuc,
        sensitivity: xgboostSensitivity,
        specificity: xgboostSpecificity,
        modelName: 'XGBoost'
      };
    } else if (selectedModel === 'knn' && knnResult !== null) {
      return {
        result: knnResult,
        accuracy: knnAccuracy,
        auc: knnAuc,
        sensitivity: knnSensitivity,
        specificity: knnSpecificity,
        modelName: 'K-Nearest Neighbors'
      };
    } else if (selectedModel === 'randomforest' && randomforestResult !== null) {
      return {
        result: randomforestResult,
        accuracy: randomforestAccuracy,
        auc: randomforestAuc,
        sensitivity: randomforestSensitivity,
        specificity: randomforestSpecificity,
        modelName: 'Random Forest'
      };
    }
    return null;
  };

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _ = getCurrentResult();

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

  useEffect(() => {
    const hasAnyPrediction = logregResult !== null || 
                           xgboostResult !== null || 
                           randomforestResult !== null || 
                           knnResult !== null;
    setPredictionExists(hasAnyPrediction);
  }, [logregResult, xgboostResult, randomforestResult, knnResult]);

  const handlePredictionSubmit = async (formData: any) => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    // Don't reset predictionExists here - let the useEffect handle it

    const numericData = Object.entries(formData).reduce((acc, [key, value]) => {
      acc[key] = Number(value);
      return acc;
    }, {} as { [key: string]: number });

    try {
      const response = await fetch('/api/predict-diabetes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(numericData),
      });
      const result = await response.json();
      if (response.ok) {
        setPrediction(result.prediction);
        setPredictionInput(numericData);
        setLogregResult(result.prediction);
        setLogregAccuracy(result.accuracy);
        setLogregAuc(result.auc_score);
        setLogregSensitivity(result.sensitivity);
        setLogregSpecificity(result.specificity);
        setLogregFetchedInput(numericData);
        setPredictionExists(true);
      } else if (response.status === 409) {
        setError(result.error || 'You already have a prediction. Delete it to re-enter.');
        setPredictionExists(true);
      } else {
        setError(result.message || 'Prediction failed.');
      }
    } catch (err) {
      setError('Prediction failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDeletePrediction = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/predict-diabetes', {
        method: 'DELETE',
        headers: {
          ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
        },
      });
      const result = await response.json();
      if (response.ok) {
        setError(null);
        setPrediction(null);
        setPredictionExists(false);
        setPredictionInput(null);
        setLogregResult(null);
        setLogregAccuracy(null);
        setLogregFetchedInput(null);
        setXgboostResult(null);
        setXgboostAccuracy(null);
        setXgboostAuc(null);
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
      } else {
        setError(result.error || 'Failed to delete prediction.');
      }
    } catch (err) {
      setError('Failed to delete prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleLogisticRegression = async () => {
    setLoading(true);
    setError(null);
    setSelectedModel('logreg');
    setXgboostResult(null);
    setXgboostAccuracy(null);
    setXgboostAuc(null);
    setXgboostFetchedInput(null);
    setRandomforestResult(null);
    setRandomforestAccuracy(null);
    setRandomforestAuc(null);
    setRandomforestFetchedInput(null);
    setKnnResult(null);
    setKnnAccuracy(null);
    setKnnAuc(null);
    setKnnFetchedInput(null);
    
    try {
      // Always make a fresh XGBoost prediction using the saved input data
      if (predictionInput) {
        const logregResponse = await fetch('/api/predict-diabetes', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
          },
          body: JSON.stringify(predictionInput),
        });
        
        if (logregResponse.ok) {
          const logregResult = await logregResponse.json();
          setLogregResult(logregResult.prediction);
          setLogregAccuracy(logregResult.accuracy);
          setLogregFetchedInput(predictionInput);
        } else {
          const errorResult = await logregResponse.json();
          setError(errorResult.message || 'Failed to create logistic regression prediction.');
        }
      } else {
        setError('No input data available for logistic regression prediction. Please submit a prediction first.');
      }
    } catch (err) {
      setError('Failed to create logistic regression prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Handler for XGBoost button
  const handleXGBoost = async () => {
    setLoading(true);
    setError(null);
    setSelectedModel('xgboost');
    
    // Clear other model results when switching to XGBoost
    setLogregResult(null);
    setLogregAccuracy(null);
    setLogregFetchedInput(null);
    setRandomforestResult(null);
    setRandomforestAccuracy(null);
    setRandomforestAuc(null);
    setRandomforestFetchedInput(null);
    setKnnResult(null);
    setKnnAccuracy(null);
    setKnnAuc(null);
    setKnnFetchedInput(null);
    
    try {
      // Always make a fresh XGBoost prediction using the saved input data
      if (predictionInput) {
        const xgboostResponse = await fetch('/api/predict-diabetes-xgboost', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
          },
          body: JSON.stringify(predictionInput),
        });
        
        if (xgboostResponse.ok) {
          const xgboostResult = await xgboostResponse.json();
          setXgboostResult(xgboostResult.prediction);
          setXgboostAccuracy(xgboostResult.accuracy);
          setXgboostAuc(xgboostResult.auc_score);
          setXgboostSensitivity(xgboostResult.sensitivity);
          setXgboostSpecificity(xgboostResult.specificity);
          setXgboostFetchedInput(predictionInput);
        } else {
          const errorResult = await xgboostResponse.json();
          setError(errorResult.message || 'Failed to create XGBoost prediction.');
        }
      } else {
        setError('No input data available for XGBoost prediction. Please submit a prediction first.');
      }
    } catch (err) {
      setError('Failed to create XGBoost prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Handler for Random Forest button
  const handleRandomForest = async () => {
    setLoading(true);
    setError(null);
    setSelectedModel('randomforest');
    
    // Clear other model results when switching to Random Forest
    setLogregResult(null);
    setLogregAccuracy(null);
    setLogregFetchedInput(null);
    setXgboostResult(null);
    setXgboostAccuracy(null);
    setXgboostAuc(null);
    setXgboostFetchedInput(null);
    setKnnResult(null);
    setKnnAccuracy(null);
    setKnnAuc(null);
    setKnnFetchedInput(null);
    
    try {
      // Always make a fresh Random Forest prediction using the saved input data
      if (predictionInput) {
        const randomforestResponse = await fetch('/api/predict-diabetes-randomforest', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
          },
          body: JSON.stringify(predictionInput),
        });
        
        if (randomforestResponse.ok) {
          const randomforestResult = await randomforestResponse.json();
          setRandomforestResult(randomforestResult.prediction);
          setRandomforestAccuracy(randomforestResult.accuracy);
          setRandomforestAuc(randomforestResult.auc_score);
          setRandomforestSensitivity(randomforestResult.sensitivity);
          setRandomforestSpecificity(randomforestResult.specificity);
          setRandomforestFetchedInput(predictionInput);
        } else {
          const errorResult = await randomforestResponse.json();
          setError(errorResult.message || 'Failed to create Random Forest prediction.');
        }
      } else {
        setError('No input data available for Random Forest prediction. Please submit a prediction first.');
      }
    } catch (err) {
      setError('Failed to create Random Forest prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Handler for KNN button
  const handleKNN = async () => {
    setLoading(true);
    setError(null);
    setSelectedModel('knn');
    
    // Clear other model results when switching to KNN
    setLogregResult(null);
    setLogregAccuracy(null);
    setLogregFetchedInput(null);
    setXgboostResult(null);
    setXgboostAccuracy(null);
    setXgboostAuc(null);
    setXgboostFetchedInput(null);
    setRandomforestResult(null);
    setRandomforestAccuracy(null);
    setRandomforestAuc(null);
    setRandomforestFetchedInput(null);
    
    try {
      // Always make a fresh KNN prediction using the saved input data
      if (predictionInput) {
        const knnResponse = await fetch('/api/predict-diabetes-knn', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
          },
          body: JSON.stringify(predictionInput),
        });
        
        if (knnResponse.ok) {
          const knnResult = await knnResponse.json();
          setKnnResult(knnResult.prediction);
          setKnnAccuracy(knnResult.accuracy);
          setKnnAuc(knnResult.auc_score);
          setKnnSensitivity(knnResult.sensitivity);
          setKnnSpecificity(knnResult.specificity);
          setKnnFetchedInput(predictionInput);
        } else {
          const errorResult = await knnResponse.json();
          setError(errorResult.message || 'Failed to create KNN prediction.');
        }
      } else {
        setError('No input data available for KNN prediction. Please submit a prediction first.');
      }
    } catch (err) {
      setError('Failed to create KNN prediction. Please try again.');
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
      <div className="relative z-10 w-full max-w-6xl flex flex-col md:flex-row gap-8">
        {/* Left Side */}
        <div className="flex-1 flex flex-col justify-center items-center text-gray-100 md:pr-4 mb-8 md:mb-0">
          <h1 className="text-4xl font-extrabold mb-4 text-center md:text-left">Data Scientist Dashboard</h1>
          <p className="text-lg text-gray-300 text-center md:text-left max-w-md">
            Select a machine learning model to view predictions and performance metrics.
          </p>
          
          {logregResult !== null && (
            <div className="mt-8 max-w-2xl mx-auto">
              <div className={`p-6 rounded-xl border-2 shadow-2xl ${
                Number(logregResult) === 0 
                  ? 'bg-gradient-to-r from-green-600 to-emerald-600 border-green-500' 
                  : 'bg-gradient-to-r from-red-600 to-orange-600 border-red-500'
              }`}>
                <div className="text-center">
                  <div className="mb-4">
                    <div className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center mb-4 ${
                      Number(logregResult) === 0 
                        ? 'bg-green-500 shadow-lg' 
                        : 'bg-red-500 shadow-lg animate-pulse'
                    }`}>
                      <span className="text-white text-3xl font-bold">
                        {Number(logregResult) === 0 ? "✓" : "⚠"}
                      </span>
                    </div>
                    <h2 className={`text-4xl font-bold mb-2 ${
                      Number(logregResult) === 0 ? 'text-white' : 'text-white'
                    }`}>
                      {Number(logregResult) === 0 ? "Negative" : "Positive"}
                    </h2>
                    <p className={`text-lg ${
                      Number(logregResult) === 0 ? 'text-green-100' : 'text-red-100'
                    }`}>
                      {Number(logregResult) === 0 
                        ? "Low Risk of Diabetes" 
                        : "High Risk of Diabetes - Medical Attention Recommended"
                      }
                    </p>
                  </div>
                  
                  {logregAccuracy !== null && (
                    <div className="mt-6 p-4 bg-white bg-opacity-20 rounded-lg border border-white border-opacity-30">
                      <div className="text-sm text-white mb-2">Model Confidence</div>
                      <div className="text-2xl font-bold text-white">
                        {(logregAccuracy * 100).toFixed(1)}%
                      </div>
                      <div className="w-full bg-white bg-opacity-30 rounded-full h-3 mt-2">
                        <div 
                          className={`h-3 rounded-full transition-all duration-500 ${
                            Number(logregResult) === 0 
                              ? 'bg-green-400' 
                              : 'bg-red-400'
                          }`}
                          style={{ width: `${(logregAccuracy * 100)}%` }}
                        ></div>
                      </div>
                      {logregAuc !== null && (
                        <div className="text-sm text-white mt-2">
                          AUC Score: {(logregAuc * 100).toFixed(1)}%
                        </div>
                      )}
                      {logregSensitivity !== null && (
                        <div className="text-sm text-white mt-1">
                          Sensitivity: {(logregSensitivity * 100).toFixed(1)}%
                        </div>
                      )}
                      {logregSpecificity !== null && (
                        <div className="text-sm text-white mt-1">
                          Specificity: {(logregSpecificity * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  )}
                  
                  {Number(logregResult) === 1 && (
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

          {xgboostResult !== null && (
            <div className="mt-8 max-w-2xl mx-auto">
              <div className={`p-6 rounded-xl border-2 shadow-2xl ${
                Number(xgboostResult) === 0 
                  ? 'bg-gradient-to-r from-green-600 to-emerald-600 border-green-500' 
                  : 'bg-gradient-to-r from-red-600 to-orange-600 border-red-500'
              }`}>
                <div className="text-center">
                  <div className="mb-4">
                    <div className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center mb-4 ${
                      Number(xgboostResult) === 0 
                        ? 'bg-green-500 shadow-lg' 
                        : 'bg-red-500 shadow-lg animate-pulse'
                    }`}>
                      <span className="text-white text-3xl font-bold">
                        {Number(xgboostResult) === 0 ? "✓" : "⚠"}
                      </span>
                    </div>
                    <h2 className={`text-4xl font-bold mb-2 ${
                      Number(xgboostResult) === 0 ? 'text-white' : 'text-white'
                    }`}>
                      {Number(xgboostResult) === 0 ? "Negative" : "Positive"}
                    </h2>
                    <p className={`text-lg ${
                      Number(xgboostResult) === 0 ? 'text-green-100' : 'text-red-100'
                    }`}>
                      {Number(xgboostResult) === 0 
                        ? "Low Risk of Diabetes" 
                        : "High Risk of Diabetes - Medical Attention Recommended"
                      }
                    </p>
                  </div>
                  
                  {xgboostAccuracy !== null && (
                    <div className="mt-6 p-4 bg-white bg-opacity-20 rounded-lg border border-white border-opacity-30">
                      <div className="text-sm text-white mb-2">Model Accuracy</div>
                      <div className="text-2xl font-bold text-white">
                        {(xgboostAccuracy * 100).toFixed(1)}%
                      </div>
                      <div className="w-full bg-white bg-opacity-30 rounded-full h-3 mt-2">
                        <div 
                          className={`h-3 rounded-full transition-all duration-500 ${
                            Number(xgboostResult) === 0 
                              ? 'bg-green-400' 
                              : 'bg-red-400'
                          }`}
                          style={{ width: `${(xgboostAccuracy * 100)}%` }}
                        ></div>
                      </div>
                      {xgboostAuc !== null && (
                        <div className="text-sm text-white mt-2">
                          AUC Score: {(xgboostAuc * 100).toFixed(1)}%
                        </div>
                      )}
                      {xgboostSensitivity !== null && (
                        <div className="text-sm text-white mt-1">
                          Sensitivity: {(xgboostSensitivity * 100).toFixed(1)}%
                        </div>
                      )}
                      {xgboostSpecificity !== null && (
                        <div className="text-sm text-white mt-1">
                          Specificity: {(xgboostSpecificity * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  )}
                  
                  {Number(xgboostResult) === 1 && (
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

          {knnResult !== null && (
            <div className="mt-8 max-w-2xl mx-auto">
              <div className={`p-6 rounded-xl border-2 shadow-2xl ${
                Number(knnResult) === 0 
                  ? 'bg-gradient-to-r from-green-600 to-emerald-600 border-green-500' 
                  : 'bg-gradient-to-r from-red-600 to-orange-600 border-red-500'
              }`}>
                <div className="text-center">
                  <div className="mb-4">
                    <div className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center mb-4 ${
                      Number(knnResult) === 0 
                        ? 'bg-green-500 shadow-lg' 
                        : 'bg-red-500 shadow-lg animate-pulse'
                    }`}>
                      <span className="text-white text-3xl font-bold">
                        {Number(knnResult) === 0 ? "✓" : "⚠"}
                      </span>
                    </div>
                    <h2 className={`text-4xl font-bold mb-2 ${
                      Number(knnResult) === 0 ? 'text-white' : 'text-white'
                    }`}>
                      {Number(knnResult) === 0 ? "Negative" : "Positive"}
                    </h2>
                    <p className={`text-lg ${
                      Number(knnResult) === 0 ? 'text-green-100' : 'text-red-100'
                    }`}>
                      {Number(knnResult) === 0 
                        ? "Low Risk of Diabetes" 
                        : "High Risk of Diabetes - Medical Attention Recommended"
                      }
                    </p>
                  </div>
                  
                  {knnAccuracy !== null && (
                    <div className="mt-6 p-4 bg-white bg-opacity-20 rounded-lg border border-white border-opacity-30">
                      <div className="text-sm text-white mb-2">Model Accuracy</div>
                      <div className="text-2xl font-bold text-white">
                        {(knnAccuracy * 100).toFixed(1)}%
                      </div>
                      <div className="w-full bg-white bg-opacity-30 rounded-full h-3 mt-2">
                        <div 
                          className={`h-3 rounded-full transition-all duration-500 ${
                            Number(knnResult) === 0 
                              ? 'bg-green-400' 
                              : 'bg-red-400'
                          }`}
                          style={{ width: `${(knnAccuracy * 100)}%` }}
                        ></div>
                      </div>
                      {knnAuc !== null && (
                        <div className="text-sm text-white mt-2">
                          AUC Score: {(knnAuc * 100).toFixed(1)}%
                        </div>
                      )}
                      {knnSensitivity !== null && (
                        <div className="text-sm text-white mt-1">
                          Sensitivity: {(knnSensitivity * 100).toFixed(1)}%
                        </div>
                      )}
                      {knnSpecificity !== null && (
                        <div className="text-sm text-white mt-1">
                          Specificity: {(knnSpecificity * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  )}
                  
                  {Number(knnResult) === 1 && (
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

          {logregFetchedInput && (
            <div className="mt-6 flex justify-center">
              <button
                onClick={() => setShowPatientInput(true)}
                className="group relative w-80 px-6 py-3 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 text-white rounded-2xl hover:from-indigo-700 hover:via-purple-700 hover:to-pink-700 transition-all duration-500 shadow-2xl hover:shadow-3xl transform hover:scale-105 hover:-translate-y-1 flex items-center font-semibold text-base"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-white to-transparent opacity-0 group-hover:opacity-20 rounded-2xl transition-opacity duration-500"></div>
                <div className="relative flex items-center">
                  <div className="mr-3 p-1.5 bg-white bg-opacity-20 rounded-full">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <span className="mr-2">View Patient Input Data</span>
                  <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-400 rounded-full animate-ping"></div>
              </button>
            </div>
          )}

          {xgboostFetchedInput && (
            <div className="mt-6 flex justify-center">
              <button
                onClick={() => setShowPatientInput(true)}
                className="group relative w-80 px-6 py-3 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 text-white rounded-2xl hover:from-indigo-700 hover:via-purple-700 hover:to-pink-700 transition-all duration-500 shadow-2xl hover:shadow-3xl transform hover:scale-105 hover:-translate-y-1 flex items-center font-semibold text-base"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-white to-transparent opacity-0 group-hover:opacity-20 rounded-2xl transition-opacity duration-500"></div>
                <div className="relative flex items-center">
                  <div className="mr-3 p-1.5 bg-white bg-opacity-20 rounded-full">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <span className="mr-2">View Patient Input Data</span>
                  <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-400 rounded-full animate-ping"></div>
              </button>
            </div>
          )}

          {knnFetchedInput && (
            <div className="mt-6 flex justify-center">
              <button
                onClick={() => setShowPatientInput(true)}
                className="group relative w-80 px-6 py-3 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 text-white rounded-2xl hover:from-indigo-700 hover:via-purple-700 hover:to-pink-700 transition-all duration-500 shadow-2xl hover:shadow-3xl transform hover:scale-105 hover:-translate-y-1 flex items-center font-semibold text-base"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-white to-transparent opacity-0 group-hover:opacity-20 rounded-2xl transition-opacity duration-500"></div>
                <div className="relative flex items-center">
                  <div className="mr-3 p-1.5 bg-white bg-opacity-20 rounded-full">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <span className="mr-2">View Patient Input Data</span>
                  <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-400 rounded-full animate-ping"></div>
              </button>
            </div>
          )}

          {logregResult !== null && (
            <div className="mt-4 flex justify-center">
              <button
                onClick={() => {
                  setSelectedModel('logreg');
                  setShowModelImages(true);
                }}
                className="group relative w-80 px-6 py-3 bg-gradient-to-r from-cyan-600 via-teal-600 to-emerald-600 text-white rounded-2xl hover:from-cyan-700 hover:via-teal-700 hover:to-emerald-700 transition-all duration-500 shadow-2xl hover:shadow-3xl transform hover:scale-105 hover:-translate-y-1 flex items-center font-semibold text-base"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-white to-transparent opacity-0 group-hover:opacity-20 rounded-2xl transition-opacity duration-500"></div>
                <div className="relative flex items-center">
                  <div className="mr-3 p-1.5 bg-white bg-opacity-20 rounded-full">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                  </div>
                  <span className="mr-2">View Model Performance</span>
                  <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </div>
              </button>
            </div>
          )}

          {xgboostResult !== null && (
            <div className="mt-4 flex justify-center">
              <button
                onClick={() => {
                  setSelectedModel('xgboost');
                  setShowModelImages(true);
                }}
                className="group relative w-80 px-6 py-3 bg-gradient-to-r from-cyan-600 via-teal-600 to-emerald-600 text-white rounded-2xl hover:from-cyan-700 hover:via-teal-700 hover:to-emerald-700 transition-all duration-500 shadow-2xl hover:shadow-3xl transform hover:scale-105 hover:-translate-y-1 flex items-center font-semibold text-base"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-white to-transparent opacity-0 group-hover:opacity-20 rounded-2xl transition-opacity duration-500"></div>
                <div className="relative flex items-center">
                  <div className="mr-3 p-1.5 bg-white bg-opacity-20 rounded-full">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                  </div>
                  <span className="mr-2">View XGBoost Performance</span>
                  <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </div>
              </button>
            </div>
          )}

          {randomforestResult !== null && (
            <div className="mt-8 max-w-2xl mx-auto">
              <div className={`p-6 rounded-xl border-2 shadow-2xl ${
                Number(randomforestResult) === 0 
                  ? 'bg-gradient-to-r from-green-600 to-emerald-600 border-green-500' 
                  : 'bg-gradient-to-r from-red-600 to-orange-600 border-red-500'
              }`}>
                <div className="text-center">
                  <div className="mb-4">
                    <div className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center mb-4 ${
                      Number(randomforestResult) === 0 
                        ? 'bg-green-500 shadow-lg' 
                        : 'bg-red-500 shadow-lg animate-pulse'
                    }`}>
                      <span className="text-white text-3xl font-bold">
                        {Number(randomforestResult) === 0 ? "✓" : "⚠"}
                      </span>
                    </div>
                    <h2 className={`text-4xl font-bold mb-2 ${
                      Number(randomforestResult) === 0 ? 'text-white' : 'text-white'
                    }`}>
                      {Number(randomforestResult) === 0 ? "Negative" : "Positive"}
                    </h2>
                    <p className={`text-lg ${
                      Number(randomforestResult) === 0 ? 'text-green-100' : 'text-red-100'
                    }`}>
                      {Number(randomforestResult) === 0 
                        ? "Low Risk of Diabetes" 
                        : "High Risk of Diabetes - Medical Attention Recommended"
                      }
                    </p>
                  </div>
                  
                  {randomforestAccuracy !== null && (
                    <div className="mt-6 p-4 bg-white bg-opacity-20 rounded-lg border border-white border-opacity-30">
                      <div className="text-sm text-white mb-2">Model Accuracy</div>
                      <div className="text-2xl font-bold text-white">
                        {(randomforestAccuracy * 100).toFixed(1)}%
                      </div>
                      <div className="w-full bg-white bg-opacity-30 rounded-full h-3 mt-2">
                        <div 
                          className={`h-3 rounded-full transition-all duration-500 ${
                            Number(randomforestResult) === 0 
                              ? 'bg-green-400' 
                              : 'bg-red-400'
                          }`}
                          style={{ width: `${(randomforestAccuracy * 100)}%` }}
                        ></div>
                      </div>
                      {randomforestAuc !== null && (
                        <div className="text-sm text-white mt-2">
                          AUC Score: {(randomforestAuc * 100).toFixed(1)}%
                        </div>
                      )}
                      {randomforestSensitivity !== null && (
                        <div className="text-sm text-white mt-1">
                          Sensitivity: {(randomforestSensitivity * 100).toFixed(1)}%
                        </div>
                      )}
                      {randomforestSpecificity !== null && (
                        <div className="text-sm text-white mt-1">
                          Specificity: {(randomforestSpecificity * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  )}
                  
                  {Number(randomforestResult) === 1 && (
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

          {randomforestFetchedInput && (
            <div className="mt-6 flex justify-center">
              <button
                onClick={() => setShowPatientInput(true)}
                className="group relative w-80 px-6 py-3 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 text-white rounded-2xl hover:from-indigo-700 hover:via-purple-700 hover:to-pink-700 transition-all duration-500 shadow-2xl hover:shadow-3xl transform hover:scale-105 hover:-translate-y-1 flex items-center font-semibold text-base"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-white to-transparent opacity-0 group-hover:opacity-20 rounded-2xl transition-opacity duration-500"></div>
                <div className="relative flex items-center">
                  <div className="mr-3 p-1.5 bg-white bg-opacity-20 rounded-full">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <span className="mr-2">View Patient Input Data</span>
                  <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-400 rounded-full animate-ping"></div>
              </button>
            </div>
          )}

          {randomforestResult !== null && (
            <div className="mt-4 flex justify-center">
              <button
                onClick={() => {
                  setSelectedModel('randomforest');
                  setShowModelImages(true);
                }}
                className="group relative w-80 px-6 py-3 bg-gradient-to-r from-cyan-600 via-teal-600 to-emerald-600 text-white rounded-2xl hover:from-cyan-700 hover:via-teal-700 hover:to-emerald-700 transition-all duration-500 shadow-2xl hover:shadow-3xl transform hover:scale-105 hover:-translate-y-1 flex items-center font-semibold text-base"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-white to-transparent opacity-0 group-hover:opacity-20 rounded-2xl transition-opacity duration-500"></div>
                <div className="relative flex items-center">
                  <div className="mr-3 p-1.5 bg-white bg-opacity-20 rounded-full">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                  </div>
                  <span className="mr-2">View Random Forest Performance</span>
                  <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-400 rounded-full animate-ping"></div>
              </button>
            </div>
          )}

          {knnResult !== null && (
            <div className="mt-4 flex justify-center">
              <button
                onClick={() => {
                  setSelectedModel('knn');
                  setShowModelImages(true);
                }}
                className="group relative w-80 px-6 py-3 bg-gradient-to-r from-cyan-600 via-teal-600 to-emerald-600 text-white rounded-2xl hover:from-cyan-700 hover:via-teal-700 hover:to-emerald-700 transition-all duration-500 shadow-2xl hover:shadow-3xl transform hover:scale-105 hover:-translate-y-1 flex items-center font-semibold text-base"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-white to-transparent opacity-0 group-hover:opacity-20 rounded-2xl transition-opacity duration-500"></div>
                <div className="relative flex items-center">
                  <div className="mr-3 p-1.5 bg-white bg-opacity-20 rounded-full">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                  </div>
                  <span className="mr-2">View KNN Performance</span>
                  <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-400 rounded-full animate-ping"></div>
              </button>
            </div>
          )}
        </div>
        {/* Right Side (Form or Buttons) */}
        <div className="flex-1 flex flex-col justify-center items-center">
          {!predictionExists && (
            <DiabetesPredictionInputForm onSubmit={handlePredictionSubmit} />
          )}
          {predictionExists && (
            <div className="flex flex-col gap-4 justify-center items-center">
              <StyledButton
                onClick={handleLogisticRegression}
                label="XGBoost"
                color="blue"
                icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2z" /></svg>}
                disabled={loading}
              />
              <StyledButton
                onClick={handleXGBoost}
                label="XGBoost"
                color="purple"
                icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>}
                disabled={loading}
              />
              <StyledButton
                onClick={handleKNN}
                label="KNN"
                color="green"
                icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 17.657L12 21l-5.657-3.343A8 8 0 1117.657 6.343L12 3l-5.657 3.343a8 8 0 1011.314 11.314z" /></svg>}
                disabled={loading}
              />
              <StyledButton
                onClick={handleRandomForest}
                label="Random Forest"
                color="yellow"
                icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l-1.586-1.586a2 2 0 010-2.828L16 8" /></svg>}
                disabled={loading}
              />
              <div className="mt-8">
                <StyledButton
                  onClick={handleDeletePrediction}
                  label="Delete My Prediction"
                  color="red"
                  icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>}
                  disabled={loading}
                />
              </div>
            </div>
          )}
        </div>
      </div>
      {showPatientInput && (logregFetchedInput || xgboostFetchedInput || randomforestFetchedInput || knnFetchedInput) && (
        <PatientInputForm 
          data={logregFetchedInput || xgboostFetchedInput || randomforestFetchedInput || knnFetchedInput} 
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

export default PredictionPage;