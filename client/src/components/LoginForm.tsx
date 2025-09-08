import React, { useState } from "react";
import { FaEnvelope, FaLock } from "react-icons/fa";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

interface LoginFormProps {
  onSwitch: () => void;
}

function LoginForm({ onSwitch }: LoginFormProps): React.JSX.Element {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { login } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const success = await login(email, password);
      if (success) {
        navigate("/dashboard");
      }
    } catch (error) {
      console.error('Login error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-2xl shadow-lg w-full max-w-3xl p-8 border border-white border-opacity-20" style={{background: 'rgba(20, 20, 20, 0.7)'}}>
      <form onSubmit={handleSubmit}>
        <h2 className="text-3xl font-bold mb-8 text-white">Login</h2>
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
        <button type="submit" className="w-full bg-blue-600 text-white py-2 rounded shadow hover:bg-blue-700 transition" disabled={loading}>
          {loading ? "Logging in..." : "LOGIN"}
        </button>
        <p className="text-white mt-4 text-sm text-center">
          Don't have an account?{" "}
          <button className="text-blue-400 underline" type="button" onClick={onSwitch}>
            Register here
          </button>
        </p>
      </form>
    </div>
  );
}

export default LoginForm; 