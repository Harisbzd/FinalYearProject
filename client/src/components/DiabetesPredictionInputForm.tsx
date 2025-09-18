import React, { useState } from "react";
import StyledButton from "./StyledButton";

const initialState = {
  HighBP: '',
  HighChol: '',
  CholCheck: '',
  BMI: '',
  Smoker: '',
  Stroke: '',
  HeartDiseaseorAttack: '',
  PhysActivity: '',
  Fruits: '',
  Veggies: '',
  HvyAlcoholConsump: '',
  AnyHealthcare: '',
  NoDocbcCost: '',
  GenHlth: '',
  MentHlth: '',
  PhysHlth: '',
  DiffWalk: '',
  Sex: '',
  Age: '',
  Education: '',
  Income: '',
};

type FormState = typeof initialState;

const yesNoOptions = [
  { value: '1', label: 'Yes' },
  { value: '0', label: 'No' },
];

// Healthcare-focused form sections
const formSections = [
  {
    title: "Medical History",
    icon: "🏥",
    fields: [
      { key: 'HighBP', label: 'High Blood Pressure', type: 'select', description: 'Do you have high blood pressure?' },
      { key: 'HighChol', label: 'High Cholesterol', type: 'select', description: 'Do you have high cholesterol?' },
      { key: 'CholCheck', label: 'Cholesterol Check', type: 'select', description: 'Have you had a cholesterol check in the past 5 years?' },
      { key: 'Stroke', label: 'Stroke History', type: 'select', description: 'Have you ever had a stroke?' },
      { key: 'HeartDiseaseorAttack', label: 'Heart Disease', type: 'select', description: 'Do you have heart disease or have had a heart attack?' },
    ]
  },
  {
    title: "Physical Health",
    icon: "💪",
    fields: [
      { key: 'BMI', label: 'Body Mass Index (BMI)', type: 'number', description: 'Enter your BMI (10-60)', placeholder: 'e.g. 25.5' },
      { key: 'PhysActivity', label: 'Physical Activity', type: 'select', description: 'Do you engage in physical activity or exercise?' },
      { key: 'DiffWalk', label: 'Walking Difficulty', type: 'select', description: 'Do you have serious difficulty walking or climbing stairs?' },
      { key: 'PhysHlth', label: 'Physical Health Days', type: 'number', description: 'How many days in the past 30 days was your physical health not good?', placeholder: '0-30' },
    ]
  },
  {
    title: "Mental Health",
    icon: "🧠",
    fields: [
      { key: 'MentHlth', label: 'Mental Health Days', type: 'number', description: 'How many days in the past 30 days was your mental health not good?', placeholder: '0-30' },
      { key: 'GenHlth', label: 'General Health Rating', type: 'select', description: 'Rate your general health (1=Excellent, 5=Poor)' },
    ]
  },
  {
    title: "Lifestyle & Diet",
    icon: "🌱",
    fields: [
      { key: 'Fruits', label: 'Fruit Consumption', type: 'select', description: 'Do you consume fruit 1 or more times per day?' },
      { key: 'Veggies', label: 'Vegetable Consumption', type: 'select', description: 'Do you consume vegetables 1 or more times per day?' },
      { key: 'Smoker', label: 'Smoking Status', type: 'select', description: 'Have you smoked at least 100 cigarettes in your entire life?' },
      { key: 'HvyAlcoholConsump', label: 'Heavy Alcohol Use', type: 'select', description: 'Heavy drinkers (adult men having more than 14 drinks per week)' },
    ]
  },
  {
    title: "Healthcare Access",
    icon: "🏥",
    fields: [
      { key: 'AnyHealthcare', label: 'Healthcare Coverage', type: 'select', description: 'Do you have any kind of health care coverage?' },
      { key: 'NoDocbcCost', label: 'Cost Barrier', type: 'select', description: 'Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?' },
    ]
  },
  {
    title: "Demographics",
    icon: "👥",
    fields: [
      { key: 'Sex', label: 'Gender', type: 'select', description: 'Select your gender' },
      { key: 'Age', label: 'Age Category', type: 'select', description: 'Select your age category (1=18-24, 13=80+)' },
      { key: 'Education', label: 'Education Level', type: 'select', description: 'What is the highest grade or year of school you completed?' },
      { key: 'Income', label: 'Income Level', type: 'select', description: 'What is your annual household income?' },
    ]
  }
];

const DiabetesPredictionInputForm: React.FC<{ onSubmit?: (data: FormState) => void }> = ({ onSubmit }) => {
  const [form, setForm] = useState<FormState>(initialState);
  const [errors, setErrors] = useState<string[]>([]);
  const [currentSection, setCurrentSection] = useState(0);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setErrors([]); // Clear errors when user makes changes
  };

  const validateForm = () => {
    const newErrors: string[] = [];
    Object.entries(form).forEach(([key, value]) => {
      if (!value || value.trim() === '') {
        newErrors.push(`${key} is required`);
      }
    });
    setErrors(newErrors);
    return newErrors.length === 0;
  };

  const validateCurrentSection = () => {
    const currentSectionFields = formSections[currentSection].fields;
    const sectionErrors: string[] = [];
    
    currentSectionFields.forEach(field => {
      const value = form[field.key as keyof FormState];
      if (!value || value.toString().trim() === '') {
        sectionErrors.push(`${field.label} is required`);
      }
    });
    
    if (sectionErrors.length > 0) {
      setErrors(sectionErrors);
      return false;
    }
    
    setErrors([]);
    return true;
  };

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) {
      e.preventDefault();
    }
    if (validateForm() && onSubmit) {
      onSubmit(form);
    }
  };

  const getSelectOptions = (fieldKey: string) => {
    switch (fieldKey) {
      case 'GenHlth':
        return [
          { value: '1', label: '1 - Excellent' },
          { value: '2', label: '2 - Very Good' },
          { value: '3', label: '3 - Good' },
          { value: '4', label: '4 - Fair' },
          { value: '5', label: '5 - Poor' },
        ];
      case 'Sex':
        return [
          { value: '1', label: 'Male' },
          { value: '0', label: 'Female' },
        ];
      case 'Age':
        return [
          { value: '1', label: '1 - 18-24 years' },
          { value: '2', label: '2 - 25-29 years' },
          { value: '3', label: '3 - 30-34 years' },
          { value: '4', label: '4 - 35-39 years' },
          { value: '5', label: '5 - 40-44 years' },
          { value: '6', label: '6 - 45-49 years' },
          { value: '7', label: '7 - 50-54 years' },
          { value: '8', label: '8 - 55-59 years' },
          { value: '9', label: '9 - 60-64 years' },
          { value: '10', label: '10 - 65-69 years' },
          { value: '11', label: '11 - 70-74 years' },
          { value: '12', label: '12 - 75-79 years' },
          { value: '13', label: '13 - 80+ years' },
        ];
      case 'Education':
        return [
          { value: '1', label: '1 - Never attended school' },
          { value: '2', label: '2 - Elementary' },
          { value: '3', label: '3 - Some high school' },
          { value: '4', label: '4 - High school graduate' },
          { value: '5', label: '5 - Some college' },
          { value: '6', label: '6 - College graduate' },
        ];
      case 'Income':
        return [
          { value: '1', label: '1 - Less than $10,000' },
          { value: '2', label: '2 - $10,000-$15,000' },
          { value: '3', label: '3 - $15,000-$20,000' },
          { value: '4', label: '4 - $20,000-$25,000' },
          { value: '5', label: '5 - $25,000-$35,000' },
          { value: '6', label: '6 - $35,000-$50,000' },
          { value: '7', label: '7 - $50,000-$75,000' },
          { value: '8', label: '8 - $75,000 or more' },
        ];
      default:
        return yesNoOptions;
    }
  };

  const renderField = (field: any) => {
    const options = getSelectOptions(field.key);
    const value = form[field.key as keyof FormState];
    const isEmpty = !value || value.toString().trim() === '';
    const hasError = errors.some(error => error.includes(field.label));
    
    return (
      <div key={field.key} className="space-y-2">
        <label className="flex items-center text-white font-medium text-sm">
          {field.label}
          <span className="text-red-400 ml-2">*</span>
        </label>
        <p className="text-gray-300 text-xs mb-2">{field.description}</p>
        
        {field.type === 'select' ? (
          <select 
            name={field.key} 
            value={value} 
            onChange={handleChange}
            className={`w-full rounded-xl px-4 py-3 border-2 bg-white bg-opacity-10 text-white placeholder-gray-300 focus:outline-none transition-all duration-300 backdrop-blur-sm ${
              hasError 
                ? 'border-red-400 focus:border-red-500' 
                : isEmpty 
                  ? 'border-yellow-400 focus:border-yellow-500' 
                  : 'border-green-400 focus:border-green-500'
            }`}
          >
            <option value="" className="bg-gray-800 text-white">Select an option</option>
            {options.map(opt => (
              <option key={opt.value} value={opt.value} className="bg-gray-800 text-white">
                {opt.label}
              </option>
            ))}
          </select>
        ) : (
          <input 
            type="number" 
            name={field.key} 
            value={value} 
            onChange={handleChange}
            placeholder={field.placeholder}
            min={field.key === 'BMI' ? '10' : '0'}
            max={field.key === 'BMI' ? '60' : '30'}
            step={field.key === 'BMI' ? '0.1' : '1'}
            className={`w-full rounded-xl px-4 py-3 border-2 bg-white bg-opacity-10 text-white placeholder-gray-300 focus:outline-none transition-all duration-300 backdrop-blur-sm ${
              hasError 
                ? 'border-red-400 focus:border-red-500' 
                : isEmpty 
                  ? 'border-yellow-400 focus:border-yellow-500' 
                  : 'border-green-400 focus:border-green-500'
            }`}
          />
        )}
        
        {hasError && (
          <p className="text-red-400 text-xs flex items-center">
            <span className="mr-1">⚠</span>
            This field is required
          </p>
        )}
      </div>
    );
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full mb-4">
          <span className="text-3xl">🏥</span>
        </div>
        <h2 className="text-3xl font-bold text-white mb-2">Health Assessment Form</h2>
        <p className="text-gray-300 text-lg">Complete your health information for diabetes risk prediction</p>
      </div>

      {/* Progress Bar */}
      <div className="mb-8">
        <div className="flex justify-between items-center mb-2">
          <span className="text-white text-sm font-medium">Progress</span>
          <span className="text-white text-sm font-medium">{currentSection + 1} of {formSections.length}</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div 
            className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-500"
            style={{ width: `${((currentSection + 1) / formSections.length) * 100}%` }}
          ></div>
        </div>
      </div>

      {/* Error Display */}
      {errors.length > 0 && (
        <div className="bg-red-500 bg-opacity-20 border border-red-400 rounded-xl p-4 mb-6">
          <div className="flex items-center mb-2">
            <span className="text-red-400 text-xl mr-2">⚠</span>
            <h3 className="text-red-200 font-semibold">Please complete all required fields</h3>
          </div>
          <p className="text-red-100 text-sm">All fields are required for accurate prediction.</p>
        </div>
      )}

      {/* Form Sections - Show only current section */}
      <div className="min-h-[500px]">
        <div 
          className="bg-white bg-opacity-10 backdrop-blur-sm rounded-2xl p-8 border border-white border-opacity-20 transition-all duration-500 ring-2 ring-blue-400 ring-opacity-50"
        >
          {/* Section Header */}
          <div className="flex items-center mb-8">
            <div className="flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl mr-6">
              <span className="text-3xl">{formSections[currentSection].icon}</span>
            </div>
            <div>
              <h3 className="text-2xl font-bold text-white">{formSections[currentSection].title}</h3>
              <p className="text-gray-300">Complete the following health information</p>
            </div>
          </div>

          {/* Section Fields */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {formSections[currentSection].fields.map(renderField)}
          </div>
        </div>
      </div>

      {/* Navigation and Submit */}
      <div className="mt-8">
        {/* Section Indicators */}
        <div className="flex justify-center mb-6">
          <div className="flex space-x-3 bg-white bg-opacity-10 rounded-full p-2">
            {formSections.map((section, index) => (
              <button
                key={index}
                type="button"
                onClick={() => {
                  // Allow going back to previous sections or current section without validation
                  if (index <= currentSection) {
                    setCurrentSection(index);
                  } else {
                    // For future sections, validate current section first
                    if (validateCurrentSection()) {
                      setCurrentSection(index);
                    }
                  }
                }}
                className={`flex items-center justify-center w-10 h-10 rounded-full transition-all duration-300 ${
                  index === currentSection 
                    ? 'bg-blue-500 text-white scale-110' 
                    : index < currentSection 
                      ? 'bg-green-500 text-white' 
                      : 'bg-gray-600 text-gray-300 hover:bg-gray-500'
                }`}
                title={section.title}
              >
                <span className="text-lg">{section.icon}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Navigation Buttons */}
        <div className="flex justify-between items-center">
          <button
            type="button"
            onClick={() => setCurrentSection(Math.max(0, currentSection - 1))}
            disabled={currentSection === 0}
            className="px-8 py-4 bg-gradient-to-r from-gray-600 to-gray-700 text-white rounded-xl hover:from-gray-700 hover:to-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 flex items-center font-semibold text-lg shadow-lg"
          >
            <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Previous
          </button>

          <div className="text-center">
            <div className="text-white font-semibold text-lg">
              {formSections[currentSection].title}
            </div>
            <div className="text-gray-300 text-sm">
              Step {currentSection + 1} of {formSections.length}
            </div>
          </div>

          {currentSection < formSections.length - 1 ? (
            <button
              type="button"
              onClick={() => {
                if (validateCurrentSection()) {
                  setCurrentSection(Math.min(formSections.length - 1, currentSection + 1));
                }
              }}
              className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:from-blue-700 hover:to-purple-700 transition-all duration-300 flex items-center font-semibold text-lg shadow-lg"
            >
              Next
              <svg className="w-6 h-6 ml-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          ) : (
            <StyledButton
              type="submit"
              label="Get My Prediction"
              color="blue"
              icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7l5-4 5 4M9 7v10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2z" /></svg>}
              onClick={() => handleSubmit()}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default DiabetesPredictionInputForm; 