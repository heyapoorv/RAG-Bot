import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import UserLayout from './layouts/UserLayout';
import AdminLayout from './layouts/AdminLayout';

// Pages
import Login from './pages/Login';
import Register from './pages/Register';
import ChatInterface from './pages/ChatInterface';
import UserProfile from './pages/UserProfile';
import HelpDesk from './pages/HelpDesk';
import AdminDashboard from './pages/AdminDashboard';
import RAGTraceViewer from './pages/RAGTraceViewer';
import ConfigPanel from './pages/ConfigPanel';
import DocumentIntelligence from './pages/DocumentIntelligence';

import PrivateRoute from './components/PrivateRoute';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public Routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        
        {/* Protected User Routes */}
        <Route element={<PrivateRoute />}>
          <Route path="/user" element={<UserLayout />}>
            <Route path="chat" element={<ChatInterface />} />
            <Route path="documents" element={<DocumentIntelligence />} />
            <Route path="profile" element={<UserProfile />} />
            <Route path="help-desk" element={<HelpDesk />} />
            <Route index element={<Navigate to="/user/chat" replace />} />
          </Route>
        </Route>

        {/* Protected Admin Routes */}
        <Route element={<PrivateRoute isAdmin={true} />}>
          <Route path="/admin" element={<AdminLayout />}>
            <Route path="dashboard" element={<AdminDashboard />} />
            <Route path="rag-trace" element={<RAGTraceViewer />} />
            <Route path="settings" element={<ConfigPanel />} />
            <Route index element={<Navigate to="/admin/dashboard" replace />} />
          </Route>
        </Route>

        {/* Fallback Route */}
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
