import { RouteObject } from 'react-router-dom';
import App from '../App';
import ErrorPage from '../components/ErrorPage';
import Dashboard from '../components/Dashboard';
import PredictionPage from '../components/predictionPage';
import SettingsPage from '../components/SettingsPage';
import ProtectedRoute from '../components/ProtectedRoute';
import Layout from '../components/Layout';
import { AuthProvider } from '../context/AuthContext';

const routes: RouteObject[] = [
    {
        path: "/",
        element: (
            <AuthProvider>
                <App />
            </AuthProvider>
        ),
        errorElement: <ErrorPage />,
    },
    {
        path: "/dashboard",
        element: (
            <AuthProvider>
                <Layout>
                    <ProtectedRoute>
                        <Dashboard />
                    </ProtectedRoute>
                </Layout>
            </AuthProvider>
        ),
        errorElement: <ErrorPage />,
    },
    {
        path: "/prediction",
        element: (
            <AuthProvider>
                <Layout>
                    <ProtectedRoute>
                        <PredictionPage />
                    </ProtectedRoute>
                </Layout>
            </AuthProvider>
        ),
        errorElement: <ErrorPage />,
    },
    {
        path: "/settings",
        element: (
            <AuthProvider>
                <Layout>
                    <ProtectedRoute>
                        <SettingsPage />
                    </ProtectedRoute>
                </Layout>
            </AuthProvider>
        ),
        errorElement: <ErrorPage />,
    },
];

export default routes;
