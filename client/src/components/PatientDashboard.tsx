import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import DiabetesPredictionInputForm from './DiabetesPredictionInputForm';
import PatientInputForm from './PatientInputForm';
import StyledButton from './StyledButton';

const PatientDashboard: React.FC = () => {
  const { token } = useAuth();
  const [prediction, setPrediction] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictionExists, setPredictionExists] = useState(false);
  const [predictionInput, setPredictionInput] = useState<any>(null);
  const [showPatientInput, setShowPatientInput] = useState(false);

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
      } else {
        setError(result.error || 'Failed to delete prediction.');
      }
    } catch (err) {
      setError('Failed to delete prediction. Please try again.');
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
        {/* Left Side - Patient Info */}
        <div className="flex-1 flex flex-col justify-center items-center text-gray-100 md:pr-4 mb-8 md:mb-0">
          <h1 className="text-4xl font-extrabold mb-4 text-center md:text-left">Patient Dashboard</h1>
          <p className="text-lg text-gray-300 text-center md:text-left max-w-md">
            Welcome to your personal health dashboard. Here you can get diabetes risk predictions and view your health data.
          </p>
          
          {prediction !== null && (
            <div className="mt-8 max-w-2xl mx-auto">
              <div className={`p-6 rounded-xl border-2 shadow-2xl ${
                Number(prediction) === 0 
                  ? 'bg-gradient-to-r from-green-600 to-emerald-600 border-green-500' 
                  : 'bg-gradient-to-r from-red-600 to-orange-600 border-red-500'
              }`}>
                <div className="text-center">
                  <div className="mb-4">
                    <div className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center mb-4 ${
                      Number(prediction) === 0 
                        ? 'bg-green-500 shadow-lg' 
                        : 'bg-red-500 shadow-lg animate-pulse'
                    }`}>
                      <span className="text-white text-3xl font-bold">
                        {Number(prediction) === 0 ? "✓" : "⚠"}
                      </span>
                    </div>
                    <h2 className={`text-4xl font-bold mb-2 ${
                      Number(prediction) === 0 ? 'text-white' : 'text-white'
                    }`}>
                      {Number(prediction) === 0 ? "Negative" : "Positive"}
                    </h2>
                    <p className={`text-lg ${
                      Number(prediction) === 0 ? 'text-green-100' : 'text-red-100'
                    }`}>
                      {Number(prediction) === 0 
                        ? "Low Risk of Diabetes" 
                        : "High Risk of Diabetes - Medical Attention Recommended"
                      }
                    </p>
                  </div>
                  
                  {Number(prediction) === 1 && (
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

          {predictionInput && (
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
                  <span className="mr-2">View My Health Data</span>
                  <svg className="w-5 h-5 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-400 rounded-full animate-ping"></div>
              </button>
            </div>
          )}
        </div>

        {/* Right Side - Form or Actions */}
        <div className="flex-1 flex flex-col justify-center items-center">
          {!predictionExists && (
            <DiabetesPredictionInputForm onSubmit={handlePredictionSubmit} />
          )}
          {predictionExists && (
            <div className="flex flex-col gap-4 justify-center items-center">
              <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-white mb-2">Your Prediction is Ready!</h3>
                <p className="text-gray-300">You can delete your current prediction to create a new one</p>
              </div>
              <StyledButton
                onClick={handleDeletePrediction}
                label="Delete My Prediction"
                color="red"
                icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>}
                disabled={loading}
              />
            </div>
          )}
        </div>
      </div>

      {showPatientInput && predictionInput && (
        <PatientInputForm 
          data={predictionInput} 
          onClose={() => setShowPatientInput(false)} 
        />
      )}
    </div>
  );
};

export default PatientDashboard;
