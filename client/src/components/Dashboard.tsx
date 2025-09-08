import React from 'react';
import { useAuth } from '../context/AuthContext';
import RoleSelection from './RoleSelection';
import PatientDashboard from './PatientDashboard';
import PredictionPage from './predictionPage';

const Dashboard: React.FC = () => {
  const { userRole, setUserRole } = useAuth();

  // If no role is selected, show role selection
  if (!userRole) {
    return <RoleSelection onRoleSelect={setUserRole} />;
  }

  // If patient role is selected, show patient dashboard
  if (userRole === 'patient') {
    return <PatientDashboard />;
  }

  // If data scientist role is selected, show prediction page with all models
  if (userRole === 'data-scientist') {
    return <PredictionPage />;
  }

  return null;
};

export default Dashboard;
