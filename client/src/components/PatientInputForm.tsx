import React, { useState } from 'react';
import { columnDescriptions } from '../data/columnDetail';
import { valueMappings } from '../data/valueMappings';

interface PatientInputFormProps {
  data: any;
  onClose: () => void;
}

const PatientInputForm: React.FC<PatientInputFormProps> = ({ data, onClose }) => {
  const [description, setDescription] = useState<string | null>(null);

  const handleDescriptionClick = (key: string) => {
    setDescription(columnDescriptions[key] || 'No description available.');
  };

  const handleCloseDescription = () => {
    setDescription(null);
  };

  const formatFieldName = (key: string) => {
    return key
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, str => str.toUpperCase())
      .replace(/([A-Z])/g, ' $1')
      .trim();
  };

  const formatValue = (key: string, value: any) => {
    const binaryYesNoFields = [
      'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 
      'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
      'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk'
    ];

    if (binaryYesNoFields.includes(key)) {
      return Number(value) === 1 ? 'Yes' : 'No';
    }

    if (key === 'Sex') {
      return Number(value) === 1 ? 'Male' : 'Female';
    }

    if (key === 'BMI') {
      return `${value} (kg/m²)`;
    }

    if (key === 'MentHlth' || key === 'PhysHlth') {
      return `${value} days`;
    }

    if (valueMappings[key] && valueMappings[key][Number(value)]) {
      return valueMappings[key][Number(value)];
    }

    return String(value);
  };

  const getFieldCategory = (key: string) => {
    if (['HighBP', 'HighChol', 'CholCheck', 'BMI'].includes(key)) return 'Vital Signs';
    if (['Smoker', 'HvyAlcoholConsump'].includes(key)) return 'Lifestyle';
    if (['Stroke', 'HeartDiseaseorAttack', 'PhysActivity'].includes(key)) return 'Medical History';
    if (['Fruits', 'Veggies'].includes(key)) return 'Diet';
    if (['AnyHealthcare', 'NoDocbcCost'].includes(key)) return 'Healthcare Access';
    if (['GenHlth', 'MentHlth', 'PhysHlth'].includes(key)) return 'Health Status';
    if (['DiffWalk', 'Sex', 'Age', 'Education', 'Income'].includes(key)) return 'Demographics';
    return 'Other';
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Vital Signs':
        return (
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        );
      case 'Lifestyle':
        return (
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      case 'Medical History':
        return (
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        );
      case 'Diet':
        return (
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 3h2l.4 2M7 13h10l4-8H5.4m0 0L7 13m0 0l-2.5 5M7 13l2.5 5m6-5v6a2 2 0 01-2 2H9a2 2 0 01-2-2v-6m6 0V9a2 2 0 00-2-2H9a2 2 0 00-2 2v4.01" />
          </svg>
        );
      case 'Healthcare Access':
        return (
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
          </svg>
        );
      case 'Health Status':
        return (
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
          </svg>
        );
      case 'Demographics':
        return (
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
          </svg>
        );
      default:
        return (
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
    }
  };

  const categories = ['Vital Signs', 'Lifestyle', 'Medical History', 'Diet', 'Healthcare Access', 'Health Status', 'Demographics'];

  return (
    <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center p-4 z-50 backdrop-blur-sm">
      <div className="no-scrollbar bg-gradient-to-br from-gray-900 via-purple-900 to-blue-900 rounded-3xl border border-gray-700 shadow-2xl max-w-7xl w-full max-h-[95vh] overflow-y-auto transform transition-all duration-500 animate-gradient">
        <div className="p-8">
          <div className="flex justify-between items-center mb-8">
            <div className="flex items-center">
              <div className="w-4 h-4 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full mr-4 animate-pulse"></div>
              <h2 className="text-3xl font-bold text-white bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Patient Data Analysis
              </h2>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white transition-colors p-2 hover:bg-gray-700 rounded-full"
            >
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {categories.map((category, index) => {
              const categoryData = Object.entries(data).filter(([key]) => getFieldCategory(key) === category);
              if (categoryData.length === 0) return null;

              return (
                <div 
                  key={category} 
                  className="bg-gradient-to-br from-gray-800 to-gray-700 rounded-2xl p-6 border border-gray-600 shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-105"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <div className="flex items-center mb-4">
                    <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl mr-4">
                      {getCategoryIcon(category)}
                    </div>
                    <h3 className="text-xl font-bold text-blue-300">{category}</h3>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {categoryData.map(([key, value]) => (
                      <div key={key} className="group bg-gradient-to-r from-gray-700 to-gray-600 rounded-xl p-4 border border-gray-500 hover:border-blue-400 transition-all duration-300 hover:shadow-lg relative">
                        <div className="flex justify-between items-center mb-2">
                          <div className="text-xs text-gray-400 uppercase tracking-wide font-semibold">
                            {formatFieldName(key)}
                          </div>
                          <button 
                            onClick={() => handleDescriptionClick(key)} 
                            className="text-gray-500 hover:text-blue-400 transition-colors"
                          >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                          </button>
                        </div>
                        <div className="text-lg font-bold text-gray-200 group-hover:text-blue-300 transition-colors">
                          {formatValue(key, value)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-8 flex justify-end">
            <button
              onClick={onClose}
              className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:from-blue-700 hover:to-purple-700 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 font-semibold"
            >
              Close
            </button>
          </div>
        </div>
      </div>
      {description && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-70"
          onClick={handleCloseDescription}
        >
          <div className="bg-gray-800 p-8 rounded-2xl shadow-2xl max-w-lg w-full text-white text-center">
            <p>{description}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default PatientInputForm; 