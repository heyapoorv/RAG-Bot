import React from 'react';
import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { 
  LayoutDashboard, Activity, Database, Users, Settings, 
  LogOut, ArrowLeft, ShieldAlert
} from 'lucide-react';
import './Layout.css';

import { useAuth } from '../context/AuthContext';

const AdminLayout = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="layout-container admin-theme">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="flex-center" style={{ gap: '8px' }}>
            <ShieldAlert size={24} className="text-accent" />
            <h1 className="logo-text">Admin Panel</h1>
          </div>
        </div>

        <div className="sidebar-content">
          <nav className="nav-links mt-4">
            <NavLink to="/admin/dashboard" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <LayoutDashboard size={20} />
              <span>Dashboard</span>
            </NavLink>
            <NavLink to="/admin/rag-trace" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <Activity size={20} />
              <span>RAG Trace Viewer</span>
            </NavLink>
            <NavLink to="/admin/data" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <Database size={20} />
              <span>Vector DB Management</span>
            </NavLink>
            <NavLink to="/admin/users" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <Users size={20} />
              <span>User Access</span>
            </NavLink>
            <NavLink to="/admin/settings" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <Settings size={20} />
              <span>System Settings</span>
            </NavLink>
          </nav>
        </div>

        <div className="sidebar-footer border-t">
          <button className="nav-item return-btn" onClick={() => navigate('/user/chat')}>
            <ArrowLeft size={20} />
            <span>Back to Chat</span>
          </button>
          
          <div className="user-profile mt-2">
            <div className="avatar admin-avatar">{user?.username?.[0]?.toUpperCase() || 'A'}</div>
            <div className="user-info">
              <span className="user-name">{user?.username || 'Admin'}</span>
              <span className="user-role text-accent">Superadmin</span>
            </div>
            <button className="icon-btn logout" onClick={handleLogout}>
              <LogOut size={18} />
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
};

export default AdminLayout;
