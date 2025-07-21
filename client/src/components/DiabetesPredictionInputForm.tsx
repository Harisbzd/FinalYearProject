import React, { useState } from "react";

const initialState = {
  HighBP: '',
  HighChol: '',
  CholCheck: '',
  BMI: '',
  Smoker: '',
  Stroke: '',
  HeartDisease: '',
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

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (onSubmit) onSubmit(form);
    // You can add validation or API call here
  };

  return (
    <div className="max-h-[700px] overflow-y-auto w-full max-w-xl mx-auto p-2 hide-scrollbar rounded-2xl" style={{background: 'rgba(20, 20, 20, 0.7)'}}>
      <form onSubmit={handleSubmit} className="space-y-2">
        <h2 className="text-2xl font-bold mb-4 text-gray-100 text-center">Diabetes Prediction Input</h2>
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
          {/* HeartDisease */}
          <div>
            <label className="block text-gray-200 mb-1">Heart Disease</label>
            <select name="HeartDisease" value={form.HeartDisease} onChange={handleChange} className="w-full rounded px-3 py-2 border border-gray-600 bg-transparent text-gray-100">
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
        <button type="submit" className="mt-4 w-full bg-gray-900 text-white py-2 rounded shadow hover:bg-gray-800 transition">
          Predict Diabetes
        </button>
      </form>
    </div>
  );
};

export default DiabetesPredictionInputForm; 