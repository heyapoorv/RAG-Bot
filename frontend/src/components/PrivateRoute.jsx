import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const PrivateRoute = ({ isAdmin = false }) => {
  const { isAuthenticated, user, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-bg-primary text-text-primary">
        <div className="animate-pulse">Loading Intelligence Suite...</div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (isAdmin && user?.role !== 'admin') {
    // Basic role check - current JWT logic in backend doesn't set role yet, 
    // but we can add it later. For now, let's just check if user exists.
    return <Navigate to="/user/chat" replace />;
  }

  return <Outlet />;
};

export default PrivateRoute;
