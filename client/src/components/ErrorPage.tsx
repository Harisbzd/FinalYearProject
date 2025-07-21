import React from 'react';
import { useRouteError } from 'react-router-dom';

const ErrorPage: React.FC = () => {
  const error = useRouteError() as Error;

  return (
    <div className="min-h-screen bg-gradient-to-br from-red-50 to-pink-100 flex items-center justify-center">
      <div className="bg-white rounded-lg shadow-xl p-8 max-w-2xl w-full mx-4 text-center">
        <h1 className="text-4xl font-bold text-red-600 mb-4">
          Oops!
        </h1>
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">
          Something went wrong
        </h2>
        <p className="text-gray-600 mb-6">
          {error?.message || 'An unexpected error occurred.'}
        </p>
        <button
          onClick={() => window.history.back()}
          className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
        >
          Go Back
        </button>
      </div>
    </div>
  );
};

export default ErrorPage; 