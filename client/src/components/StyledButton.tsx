import React from 'react';

interface StyledButtonProps {
  onClick?: () => void;
  label: string;
  icon: JSX.Element;
  color: 'blue' | 'purple' | 'green' | 'yellow' | 'red';
  disabled?: boolean;
  type?: 'button' | 'submit' | 'reset';
}

const colorSchemes = {
  blue: 'from-blue-600 via-sky-600 to-cyan-600',
  purple: 'from-purple-600 via-fuchsia-600 to-pink-600',
  green: 'from-green-600 via-emerald-600 to-teal-600',
  yellow: 'from-yellow-600 via-amber-600 to-orange-600',
  red: 'from-red-600 via-rose-600 to-pink-600',
};

const StyledButton: React.FC<StyledButtonProps> = ({ onClick, label, icon, color, disabled, type = 'button' }) => {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`group relative w-full px-8 py-4 bg-gradient-to-r ${colorSchemes[color]} text-white rounded-2xl hover:shadow-2xl transition-all duration-500 shadow-lg transform hover:scale-105 hover:-translate-y-1 flex items-center justify-center font-semibold text-lg`}
    >
      <div className="absolute inset-0 bg-gradient-to-r from-white to-transparent opacity-0 group-hover:opacity-20 rounded-2xl transition-opacity duration-500"></div>
      <div className="relative flex items-center">
        <div className="mr-3 p-2 bg-white bg-opacity-20 rounded-full">
          {icon}
        </div>
        <span className="mr-2">{label}</span>
        <svg className="w-5 h-5 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
        </svg>
      </div>
    </button>
  );
};

export default StyledButton; 