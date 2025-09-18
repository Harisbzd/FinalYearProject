import React from 'react';
import { useAuth } from '../context/AuthContext';
import RoleSelection from './RoleSelection';
import PatientDashboard from './PatientDashboard';
import DataScientistDashboard from './DataScientistDashboard';

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

  // If data scientist role is selected, show data scientist dashboard
  if (userRole === 'data-scientist') {
    return <DataScientistDashboard />;
  }

  return null;
};

export default Dashboard;
