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

const DiabetesPredictionInputForm: React.FC<{ onSubmit?: (data: FormState) => void }> = ({ onSubmit }) => {
  const [form, setForm] = useState<FormState>(initialState);
  const [errors, setErrors] = useState<string[]>([]);

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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateForm() && onSubmit) {
      onSubmit(form);
    }
  };

  return (
    <div className="max-h-[700px] overflow-y-auto w-full max-w-xl mx-auto p-2 no-scrollbar rounded-2xl" style={{background: 'rgba(20, 20, 20, 0.7)'}}>
      <form onSubmit={handleSubmit} className="space-y-2">
        <h2 className="text-2xl font-bold mb-4 text-gray-100 text-center">Diabetes Prediction Input</h2>
        
        {errors.length > 0 && (
          <div className="bg-red-500 bg-opacity-20 p-4 rounded-lg mb-4">
            <h3 className="text-red-200 font-semibold mb-2">Please fix the following errors:</h3>
            <ul className="list-disc list-inside text-red-100">
              {errors.map((error, index) => (
                <li key={index}>{error}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* HighBP */}
          <div>
            <label className="block text-gray-200 mb-1">High Blood Pressure</label>
            <select name="HighBP" value={form.HighBP} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {yesNoOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
            </select>
          </div>
          {/* HighChol */}
          <div>
            <label className="block text-gray-200 mb-1">High Cholesterol</label>
            <select name="HighChol" value={form.HighChol} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {yesNoOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
            </select>
          </div>
          {/* CholCheck */}
          <div>
            <label className="block text-gray-200 mb-1">Cholesterol Check</label>
            <select name="CholCheck" value={form.CholCheck} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {yesNoOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
            </select>
          </div>
          {/* BMI */}
          <div>
            <label className="block text-gray-200 mb-1">BMI</label>
            <input type="number" name="BMI" value={form.BMI} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100 placeholder-gray-400" min="10" max="60" step="0.1" placeholder="e.g. 25.5" />
          </div>
          {/* Smoker */}
          <div>
            <label className="block text-gray-200 mb-1">Smoker</label>
            <select name="Smoker" value={form.Smoker} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {yesNoOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
            </select>
          </div>
          {/* Stroke */}
          <div>
            <label className="block text-gray-200 mb-1">Stroke</label>
            <select name="Stroke" value={form.Stroke} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {yesNoOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
            </select>
          </div>
          {/* HeartDiseaseorAttack */}
          <div>
            <label className="block text-gray-200 mb-1">Heart Disease or Attack</label>
            <select name="HeartDiseaseorAttack" value={form.HeartDiseaseorAttack} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {yesNoOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
            </select>
          </div>
          {/* PhysActivity */}
          <div>
            <label className="block text-gray-200 mb-1">Physical Activity</label>
            <select name="PhysActivity" value={form.PhysActivity} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {yesNoOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
            </select>
          </div>
          {/* Fruits */}
          <div>
            <label className="block text-gray-200 mb-1">Fruits Intake</label>
            <select name="Fruits" value={form.Fruits} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {yesNoOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
            </select>
          </div>
          {/* Veggies */}
          <div>
            <label className="block text-gray-200 mb-1">Vegetables Intake</label>
            <select name="Veggies" value={form.Veggies} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {yesNoOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
            </select>
          </div>
          {/* HvyAlcoholConsump */}
          <div>
            <label className="block text-gray-200 mb-1">Heavy Alcohol Consumption</label>
            <select name="HvyAlcoholConsump" value={form.HvyAlcoholConsump} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {yesNoOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
            </select>
          </div>
          {/* AnyHealthcare */}
          <div>
            <label className="block text-gray-200 mb-1">Any Healthcare</label>
            <select name="AnyHealthcare" value={form.AnyHealthcare} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {yesNoOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
            </select>
          </div>
          {/* NoDocbcCost */}
          <div>
            <label className="block text-gray-200 mb-1">No Doctor Because of Cost</label>
            <select name="NoDocbcCost" value={form.NoDocbcCost} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {yesNoOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
            </select>
          </div>
          {/* GenHlth */}
          <div>
            <label className="block text-gray-200 mb-1">General Health</label>
            <select name="GenHlth" value={form.GenHlth} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {[1,2,3,4,5].map(val => <option key={val} value={val}>{val}</option>)}
            </select>
          </div>
          {/* MentHlth */}
          <div>
            <label className="block text-gray-200 mb-1">Mental Health (days)</label>
            <input type="number" name="MentHlth" value={form.MentHlth} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100 placeholder-gray-400" min="0" max="30" placeholder="0-30" />
          </div>
          {/* PhysHlth */}
          <div>
            <label className="block text-gray-200 mb-1">Physical Health (days)</label>
            <input type="number" name="PhysHlth" value={form.PhysHlth} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100 placeholder-gray-400" min="0" max="30" placeholder="0-30" />
          </div>
          {/* DiffWalk */}
          <div>
            <label className="block text-gray-200 mb-1">Difficulty Walking</label>
            <select name="DiffWalk" value={form.DiffWalk} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {yesNoOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
            </select>
          </div>
          {/* Sex */}
          <div>
            <label className="block text-gray-200 mb-1">Sex</label>
            <select name="Sex" value={form.Sex} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              <option value="1">Male</option>
              <option value="0">Female</option>
            </select>
          </div>
          {/* Age */}
          <div>
            <label className="block text-gray-200 mb-1">Age Category</label>
            <select name="Age" value={form.Age} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {[1,2,3,4,5,6,7,8,9,10,11,12,13].map(val => <option key={val} value={val}>{val}</option>)}
            </select>
          </div>
          {/* Education */}
          <div>
            <label className="block text-gray-200 mb-1">Education Level</label>
            <select name="Education" value={form.Education} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {[1,2,3,4,5,6].map(val => <option key={val} value={val}>{val}</option>)}
            </select>
          </div>
          {/* Income */}
          <div>
            <label className="block text-gray-200 mb-1">Income Level</label>
            <select name="Income" value={form.Income} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
              <option value="">Select</option>
              {[1,2,3,4,5,6,7,8].map(val => <option key={val} value={val}>{val}</option>)}
            </select>
          </div>
        </div>
        <div className="pt-4">
          <StyledButton
            type="submit"
            label="Predict Diabetes"
            color="blue"
            icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7l5-4 5 4M9 7v10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2z" /></svg>}
          />
        </div>
      </form>
    </div>
  );
};

export default DiabetesPredictionInputForm; 