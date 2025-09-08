import React, { useState } from "react";
import { FaUser, FaEnvelope, FaLock, FaKey } from "react-icons/fa";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

interface SignUpFormProps {
  onSwitch: () => void;
}

function SignUpForm({ onSwitch }: SignUpFormProps): React.JSX.Element {
  // State for form fields
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [repeatPassword, setRepeatPassword] = useState("");
  const [agree, setAgree] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { register } = useAuth();

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!agree) {
      alert("You must agree to the terms of service.");
      return;
    }
    if (!name || !email || !password || !repeatPassword) {
      alert("Please fill in all fields.");
      return;
    }
    if (password !== repeatPassword) {
      alert("Passwords do not match.");
      return;
    }
    setLoading(true);
    
    try {
      const success = await register(name, email, password);
      if (success) {
        navigate("/dashboard");
      }
    } catch (error) {
      console.error('Registration error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-2xl shadow-lg w-full max-w-3xl p-8 border border-white border-opacity-20" style={{background: 'rgba(20, 20, 20, 0.7)'}}>
      <form onSubmit={handleSubmit}>
        <h2 className="text-3xl font-bold mb-8 text-white">Sign Up</h2>
        <div className="flex items-center mb-4 border border-white rounded px-3 py-2">
          <FaUser className="mr-3 text-white" />
          <input
            type="text"
            placeholder="Name"
            className="outline-none flex-1 bg-transparent text-white"
            value={name}
            onChange={e => setName(e.target.value)}
            required
          />
        </div>
        <div className="flex items-center mb-4 border border-white rounded px-3 py-2">
          <FaEnvelope className="mr-3 text-white" />
          <input
            type="email"
            placeholder="Email"
            className="outline-none flex-1 bg-transparent text-white"
            value={email}
            onChange={e => setEmail(e.target.value)}
            required
          />
        </div>
        <div className="flex items-center mb-4 border border-white rounded px-3 py-2">
          <FaLock className="mr-3 text-white" />
          <input
            type="password"
            placeholder="Password"
            className="outline-none flex-1 bg-transparent text-white"
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
          />
        </div>
        {/* Repeat Password Field */}
        <div className="flex items-center mb-4 border border-white rounded px-3 py-2">
          <FaKey className="mr-3 text-white" />
          <input
            type="password"
            placeholder="Repeat Password"
            className="outline-none flex-1 bg-transparent text-white"
            value={repeatPassword}
            onChange={e => setRepeatPassword(e.target.value)}
            required
          />
        </div>
        {/* Agree to Terms Checkbox */}
        <div className="flex items-center mb-6">
          <input
            type="checkbox"
            id="agree"
            checked={agree}
            onChange={e => setAgree(e.target.checked)}
            className="mr-2"
            required
          />
          <label htmlFor="agree" className="text-white text-sm select-none">
            I agree to the <a href="#" className="underline text-blue-300">terms of service</a>.
          </label>
        </div>
        <button type="submit" className="w-full bg-blue-600 text-white py-2 rounded shadow hover:bg-blue-700 transition" disabled={loading}>
          {loading ? "Registering..." : "SIGN UP"}
        </button>
        <p className="text-white mt-4 text-sm text-center">
          Already have an account?{" "}
          <button className="text-blue-400 underline" type="button" onClick={onSwitch}>
            Login here
          </button>
        </p>
      </form>
    </div>
  );
}

export default SignUpForm; 