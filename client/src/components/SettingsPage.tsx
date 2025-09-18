import React from 'react';
import { useAuth } from '../context/AuthContext';
import { FaUser, FaKey, FaCalendar } from 'react-icons/fa';

const SettingsPage: React.FC = () => {
  const { user } = useAuth();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl p-8 max-w-2xl w-full">
        <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
          Account Settings
        </h1>
        
        <div className="space-y-6">
          <div className="bg-blue-50 rounded-lg p-6">
            <h2 className="text-xl font-semibold text-blue-800 mb-4 flex items-center">
              <FaUser className="mr-2" />
              Profile Information
            </h2>
            <div className="space-y-3">
              <div className="flex items-center">
                <span className="font-medium text-gray-700 w-24">Username:</span>
                <span className="text-gray-900">{user?.username}</span>
              </div>
              <div className="flex items-center">
                <span className="font-medium text-gray-700 w-24">Email:</span>
                <span className="text-gray-900">{user?.email}</span>
              </div>
              <div className="flex items-center">
                <span className="font-medium text-gray-700 w-24">User ID:</span>
                <span className="text-gray-900 font-mono text-sm">{user?.id}</span>
              </div>
            </div>
          </div>

          <div className="bg-green-50 rounded-lg p-6">
            <h2 className="text-xl font-semibold text-green-800 mb-4 flex items-center">
              <FaKey className="mr-2" />
              Security Settings
            </h2>
            <div className="space-y-3">
              <p className="text-green-700">
                Your account is secured with JWT token authentication.
              </p>
              <div className="bg-green-100 rounded p-3">
                <p className="text-green-800 text-sm">
                  <strong>Security Status:</strong> Active
                </p>
                <p className="text-green-700 text-sm">
                  Token expires in 24 hours for enhanced security.
                </p>
              </div>
            </div>
          </div>

          {/* Account Actions */}
          <div className="bg-yellow-50 rounded-lg p-6">
            <h2 className="text-xl font-semibold text-yellow-800 mb-4 flex items-center">
              <FaCalendar className="mr-2" />
              Account Actions
            </h2>
            <div className="space-y-3">
              <p className="text-yellow-700">
                Manage your account settings and preferences.
              </p>
              <div className="space-y-2">
                <button className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors">
                  Change Password
                </button>
                <button className="w-full bg-gray-600 hover:bg-gray-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors">
                  Update Profile
                </button>
                <button className="w-full bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors">
                  Delete Account
                </button>
              </div>
            </div>
          </div>

          <div className="text-center">
            <p className="text-gray-500 text-sm">
              Use the navigation bar above to access different sections of the app.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsPage; 