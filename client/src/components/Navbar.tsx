import React, { useState, useRef, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { FaUser, FaSignOutAlt, FaCog, FaChevronDown, FaExchangeAlt } from 'react-icons/fa';

const Navbar: React.FC = () => {
  const { user, logout, isAuthenticated, userRole, setUserRole } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/');
    setIsDropdownOpen(false);
  };

  const handleLogoClick = () => {
    navigate('/dashboard');
  };

  const handleRoleSwitch = () => {
    if (userRole === 'patient') {
      setUserRole('data-scientist');
    } else if (userRole === 'data-scientist') {
      setUserRole('patient');
    }
    setIsDropdownOpen(false);
  };

  return (
    <nav className="shadow-lg border-b border-gray-300 fixed w-full z-50 bg-gray-600">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo and Brand */}
          <div className="flex items-center">
            <div 
              className="flex-shrink-0 cursor-pointer"
              onClick={handleLogoClick}
            >
              <h1 className="text-2xl font-bold text-white hover:text-gray-300 transition-colors drop-shadow-lg">
                Health Portal
              </h1>
            </div>
          </div>

          {/* Spacer to push user menu to the right */}
          <div className="flex-1"></div>

          {/* User Menu */}
          {isAuthenticated && user && (
            <div className="relative" ref={dropdownRef}>
              <button
                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                className="flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium text-gray-100 hover:text-white hover:bg-gray-700 transition-colors drop-shadow-lg"
              >
                <div className="w-8 h-8 bg-gray-700 rounded-full flex items-center justify-center">
                  <FaUser className="text-white text-sm" />
                </div>
                <span className="hidden md:block">{user.username}</span>
                <FaChevronDown className={`text-xs transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
              </button>

              {/* Dropdown Menu */}
              {isDropdownOpen && (
                <div className="absolute right-0 mt-2 w-56 bg-gray-900 bg-opacity-90 rounded-md shadow-lg py-1 z-50 border border-gray-700">
                  {/* User Info Section */}
                  <div className="px-4 py-3 border-b border-gray-700">
                    <p className="text-sm font-medium text-white">{user.username}</p>
                    <p className="text-sm text-gray-400">{user.email}</p>
                    {userRole && (
                      <p className="text-xs text-blue-400 mt-1">
                        Role: {userRole === 'patient' ? 'Patient' : 'Data Scientist'}
                      </p>
                    )}
                  </div>

                  {/* Menu Items */}
                  <div className="py-1">
                    {userRole && (
                      <button
                        onClick={handleRoleSwitch}
                        className="flex items-center w-full px-4 py-2 text-sm text-gray-200 hover:bg-gray-700 hover:text-white transition-colors"
                      >
                        <FaExchangeAlt className="mr-3 text-gray-400" />
                        Switch to {userRole === 'patient' ? 'Data Scientist' : 'Patient'}
                      </button>
                    )}
                    <button
                      onClick={() => {
                        navigate('/settings');
                        setIsDropdownOpen(false);
                      }}
                      className="flex items-center w-full px-4 py-2 text-sm text-gray-200 hover:bg-gray-700 hover:text-white transition-colors"
                    >
                      <FaCog className="mr-3 text-gray-400" />
                      Settings
                    </button>
                    <button
                      onClick={handleLogout}
                      className="flex items-center w-full px-4 py-2 text-sm text-red-400 hover:bg-red-900 hover:text-red-300 transition-colors"
                    >
                      <FaSignOutAlt className="mr-3" />
                      Logout
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-gray-200 hover:text-white hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-gray-500"
            >
              <span className="sr-only">Open main menu</span>
              <svg
                className={`${isDropdownOpen ? 'hidden' : 'block'} h-6 w-6`}
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
              <svg
                className={`${isDropdownOpen ? 'block' : 'hidden'} h-6 w-6`}
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        <div className={`${isDropdownOpen ? 'block' : 'hidden'} md:hidden`}>
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-gray-900 bg-opacity-90">
            {isAuthenticated && user && (
              <div className="border-t border-gray-700 pt-4">
                <div className="px-3 py-2">
                  <p className="text-sm font-medium text-white">{user.username}</p>
                  <p className="text-sm text-gray-400">{user.email}</p>
                </div>
                <button
                  onClick={() => {
                    navigate('/settings');
                    setIsDropdownOpen(false);
                  }}
                  className="block w-full text-left px-3 py-2 rounded-md text-base font-medium text-gray-200 hover:text-white hover:bg-gray-700"
                >
                  Settings
                </button>
                <button
                  onClick={handleLogout}
                  className="block w-full text-left px-3 py-2 rounded-md text-base font-medium text-red-400 hover:text-red-300 hover:bg-red-900"
                >
                  Logout
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar; 