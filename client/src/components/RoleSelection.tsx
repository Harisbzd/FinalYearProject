import React from 'react';
import { useAuth } from '../context/AuthContext';

interface RoleSelectionProps {
  onRoleSelect: (role: 'patient' | 'data-scientist') => void;
}

const RoleSelection: React.FC<RoleSelectionProps> = ({ onRoleSelect }) => {
  const { user } = useAuth();

  return (
    <div 
      className="min-h-screen flex items-center justify-center p-4 bg-cover bg-center bg-no-repeat relative"
      style={{
        backgroundImage: "url('/1.jpg')",
      }}
    >
      <div className="absolute inset-0 bg-black bg-opacity-40"></div>
      <div className="relative z-10 w-full max-w-4xl flex flex-col items-center">
        {/* Welcome Message */}
        <div className="text-center mb-12">
          <h1 className="text-5xl md:text-6xl font-extrabold text-white mb-6 tracking-wide leading-tight">
            Welcome, {user?.username}!
          </h1>
          <p className="text-xl text-gray-200 max-w-2xl mx-auto">
            Please select your role to continue to the appropriate dashboard
          </p>
        </div>

        {/* Role Selection Cards */}
        <div className="grid md:grid-cols-2 gap-8 w-full max-w-4xl">
          {/* Patient Card */}
          <div 
            onClick={() => onRoleSelect('patient')}
            className="group cursor-pointer transform transition-all duration-300 hover:scale-105 hover:-translate-y-2"
          >
            <div className="bg-gradient-to-br from-blue-600 via-blue-700 to-indigo-800 rounded-3xl p-8 shadow-2xl border-2 border-blue-500 hover:border-blue-300 transition-all duration-300 h-80 flex flex-col justify-between">
              <div className="text-center flex-1 flex flex-col justify-center">
                <div className="w-20 h-20 mx-auto mb-4 bg-white bg-opacity-20 rounded-full flex items-center justify-center group-hover:bg-opacity-30 transition-all duration-300">
                  <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-white mb-3">Patient</h2>
                <p className="text-blue-100 text-base leading-relaxed">
                  Access your personal health dashboard, view predictions, and manage your health data
                </p>
              </div>
              <div className="mt-4 flex items-center justify-center text-white group-hover:text-blue-200 transition-colors duration-300">
                <span className="mr-2 font-semibold text-sm">Enter as Patient</span>
                <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </div>
            </div>
          </div>

          {/* Data Scientist Card */}
          <div 
            onClick={() => onRoleSelect('data-scientist')}
            className="group cursor-pointer transform transition-all duration-300 hover:scale-105 hover:-translate-y-2"
          >
            <div className="bg-gradient-to-br from-purple-600 via-purple-700 to-pink-800 rounded-3xl p-8 shadow-2xl border-2 border-purple-500 hover:border-purple-300 transition-all duration-300 h-80 flex flex-col justify-between">
              <div className="text-center flex-1 flex flex-col justify-center">
                <div className="w-20 h-20 mx-auto mb-4 bg-white bg-opacity-20 rounded-full flex items-center justify-center group-hover:bg-opacity-30 transition-all duration-300">
                  <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-white mb-3">Data Scientist</h2>
                <p className="text-purple-100 text-base leading-relaxed">
                  Access advanced ML models, view performance metrics, and analyze prediction results
                </p>
              </div>
              <div className="mt-4 flex items-center justify-center text-white group-hover:text-purple-200 transition-colors duration-300">
                <span className="mr-2 font-semibold text-sm">Enter as Data Scientist</span>
                <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Additional Info */}
        <div className="mt-12 text-center">
          <p className="text-gray-300 text-sm">
            You can switch between roles at any time from the navigation menu
          </p>
        </div>
      </div>
    </div>
  );
};

export default RoleSelection;
