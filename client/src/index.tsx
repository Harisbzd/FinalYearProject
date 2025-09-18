import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import routes from './routes/routes';
import { RouterProvider, createBrowserRouter } from 'react-router-dom';

const router = createBrowserRouter(routes);

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
); 