import React from 'react';
import { useLocation } from 'react-router-dom';
import Navbar from './Navbar';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();
  
  // Don't show navbar on homepage (root path)
  const showNavbar = location.pathname !== '/';

  return (
    <div className="min-h-screen bg-gray-50">
      {showNavbar && <Navbar />}
      <main className={showNavbar ? 'pt-0' : ''}>
        {children}
      </main>
    </div>
  );
};

export default Layout; 