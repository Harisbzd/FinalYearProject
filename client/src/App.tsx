import React, { useState } from "react";
import ChatWidget from "./components/ChatWidget";
import SignUpForm from "./components/SignUpForm";
import LoginForm from "./components/LoginForm";

function App(): React.JSX.Element {
  const [showRegister, setShowRegister] = useState<boolean>(false); // toggle between login and signup

  return (
    <div
      className="min-h-screen flex items-center justify-center bg-cover bg-center bg-opacity-40"
      style={{
        backgroundImage: "url('/1.jpg')",
      }}
    >
      {/* Overlay */}
      <div className="w-full min-h-screen bg-black bg-opacity-40 flex flex-col items-center justify-center px-4 py-10">
        {/* Heading */}
        <h1
          className="text-5xl md:text-6xl font-extrabold text-center mb-10 tracking-wide leading-tight"
          style={{
            background: 'linear-gradient(135deg, #ffffff 0%, #f0f8ff 50%, #e6f3ff 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            textShadow: '0 4px 8px rgba(0,0,0,0.3)',
            filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.5))',
            fontFamily: "'Playfair Display', 'Georgia', 'Times New Roman', serif"
          }}
        >
          Welcome to Health Portal
        </h1>

        {/* Login or Signup */}
        {showRegister ? (
          <SignUpForm onSwitch={() => setShowRegister(false)} />
        ) : (
          <LoginForm onSwitch={() => setShowRegister(true)} />
        )}

        {/* Chat Widget */}
        <div className="mt-12 w-full max-w-2xl">
          <ChatWidget />
        </div>
      </div>
    </div>
  );
}

export default App; 