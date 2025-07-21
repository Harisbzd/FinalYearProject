import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import DiabetesPredictionInputForm from './DiabetesPredictionInputForm';
import PatientInputForm from './PatientInputForm';
import ModelImages from './ModelImages';
import StyledButton from './StyledButton';

const PredictionPage: React.FC = () => {
  const { user, token } = useAuth();
  const [prediction, setPrediction] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictionExists, setPredictionExists] = useState(false);
  const [predictionInput, setPredictionInput] = useState<any>(null);
  // Add state for logistic regression result
  const [logregResult, setLogregResult] = useState<string | null>(null);
  const [logregAccuracy, setLogregAccuracy] = useState<number | null>(null);
  const [logregFetchedInput, setLogregFetchedInput] = useState<any>(null);
  const [showPatientInput, setShowPatientInput] = useState(false);
  const [showModelImages, setShowModelImages] = useState(false);

  // Fetch user's prediction on mount
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
          setLogregFetchedInput(result.input);
          setPredictionExists(true);
        } else {
          setPrediction(null);
          setPredictionInput(null);
          setPredictionExists(false);
        }
      } catch (err) {
        setPrediction(null);
        setPredictionInput(null);
        setPredictionExists(false);
      } finally {
        setLoading(false);
      }
    };
    if (token) fetchPrediction();
  }, [token]);

  const handlePredictionSubmit = async (formData: any) => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    setPredictionExists(false);
    try {
      const response = await fetch('/api/predict-diabetes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(formData),
      });
      const result = await response.json();
      if (response.ok) {
        setPrediction(result.prediction);
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
      } else {
        setError(result.error || 'Failed to delete prediction.');
      }
    } catch (err) {
      setError('Failed to delete prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Handler for Logistic Regression button
  const handleLogisticRegression = async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch user's saved input and prediction from backend
      const response = await fetch('/api/predict-diabetes', {
        method: 'GET',
        headers: {
          ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
        },
      });
      const result = await response.json();
      if (response.ok) {
        setLogregResult(result.prediction);
        setLogregAccuracy(result.accuracy);
        setLogregFetchedInput(result.input);
      } else {
        setError(result.error || 'Failed to fetch prediction.');
      }
    } catch (err) {
      setError('Failed to fetch prediction. Please try again.');
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
          <h1 className="text-4xl font-extrabold mb-4 text-center md:text-left">Diabetes Prediction</h1>
          <p className="text-lg text-gray-300 text-center md:text-left max-w-md">
            Welcome! Fill out the form to the right to get a diabetes risk prediction. Your data is private and predictions are saved to your account.
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

          {logregFetchedInput && (
            <div className="mt-6 flex justify-center">
              <button
                onClick={() => setShowPatientInput(true)}
                className="group relative px-8 py-4 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 text-white rounded-2xl hover:from-indigo-700 hover:via-purple-700 hover:to-pink-700 transition-all duration-500 shadow-2xl hover:shadow-3xl transform hover:scale-105 hover:-translate-y-1 flex items-center font-semibold text-lg"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-white to-transparent opacity-0 group-hover:opacity-20 rounded-2xl transition-opacity duration-500"></div>
                <div className="relative flex items-center">
                  <div className="mr-3 p-2 bg-white bg-opacity-20 rounded-full">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <span className="mr-2">View Patient Input Data</span>
                  <svg className="w-5 h-5 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
                onClick={() => setShowModelImages(true)}
                className="group relative px-8 py-4 bg-gradient-to-r from-cyan-600 via-teal-600 to-emerald-600 text-white rounded-2xl hover:from-cyan-700 hover:via-teal-700 hover:to-emerald-700 transition-all duration-500 shadow-2xl hover:shadow-3xl transform hover:scale-105 hover:-translate-y-1 flex items-center font-semibold text-lg"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-white to-transparent opacity-0 group-hover:opacity-20 rounded-2xl transition-opacity duration-500"></div>
                <div className="relative flex items-center">
                  <div className="mr-3 p-2 bg-white bg-opacity-20 rounded-full">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                  </div>
                  <span className="mr-2">View Model Performance</span>
                  <svg className="w-5 h-5 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </div>
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
                label="Logistic Regression"
                color="blue"
                icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2z" /></svg>}
                disabled={loading}
              />
              <StyledButton
                onClick={() => {}}
                label="XGBoost"
                color="purple"
                icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>}
                disabled={loading}
              />
              <StyledButton
                onClick={() => {}}
                label="KNN"
                color="green"
                icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 17.657L12 21l-5.657-3.343A8 8 0 1117.657 6.343L12 3l-5.657 3.343a8 8 0 1011.314 11.314z" /></svg>}
                disabled={loading}
              />
              <StyledButton
                onClick={() => {}}
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
      {showPatientInput && logregFetchedInput && (
        <PatientInputForm 
          data={logregFetchedInput} 
          onClose={() => setShowPatientInput(false)} 
        />
      )}
      {showModelImages && (
        <ModelImages onClose={() => setShowModelImages(false)} />
      )}
    </div>
  );
};

export default PredictionPage;
