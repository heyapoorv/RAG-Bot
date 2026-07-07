import React, { useState, useEffect } from 'react';
import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { 
  MessageSquare, Plus, Folder, User, Settings, 
  HelpCircle, Menu, X, LayoutDashboard, FileText, ChevronLeft, ChevronRight, LogOut, Sparkles
} from 'lucide-react';
import './Layout.css';

import { useAuth } from '../context/AuthContext';
import axios from 'axios';

const UserLayout = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [sessionId, setSessionId] = useState(`session_${Date.now()}`);
  const [pastChats, setPastChats] = useState([]);
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  useEffect(() => {
    if (user?.username) {
      axios.get(`${API_URL}/query/sessions?namespace=${user.username}`)
        .then(res => setPastChats(res.data))
        .catch(err => console.error("Failed to fetch sessions", err));
    }
  }, [user, sessionId]);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const toggleSidebar = () => setCollapsed(!collapsed);

  const handleNewChat = () => {
    setSessionId(`session_${Date.now()}`);
    navigate('/user/chat');
  };

  return (
    <div className="layout-container">
      {/* Sidebar */}
      <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
        <div className="sidebar-header">
          {!collapsed && <h1 className="logo-text">DocIntel AI</h1>}
          <button className="icon-btn" onClick={toggleSidebar}>
            {collapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
          </button>
        </div>

        <div className="sidebar-content">
          <button className="new-chat-btn" onClick={handleNewChat}>
            <Plus size={20} />
            {!collapsed && <span>New Chat</span>}
          </button>

          {!collapsed && (
            <div className="sidebar-section">
              <span className="section-label">Past Chats</span>
              <ul className="sidebar-list">
                {pastChats.map(chat => (
                  <li 
                    key={chat.session_id} 
                    className={chat.session_id === sessionId ? "active" : ""}
                    onClick={() => {
                      setSessionId(chat.session_id);
                      navigate('/user/chat');
                    }}
                    style={{ cursor: 'pointer' }}
                  >
                    <MessageSquare size={16}/> {chat.preview}
                  </li>
                ))}
                {pastChats.length === 0 && (
                  <li className="active"><MessageSquare size={16}/> Current Session</li>
                )}
              </ul>
            </div>
          )}

          {!collapsed && (
            <div className="sidebar-section">
              <span className="section-label">Uploaded Docs</span>
              <ul className="sidebar-list">
                <li><FileText size={16}/> Documents automatically embedded</li>
              </ul>
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          <nav className="nav-links">
            <NavLink to="/user/chat" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <MessageSquare size={20} />
              {!collapsed && <span>Chat</span>}
            </NavLink>
            <NavLink to="/user/documents" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <Sparkles size={20} className="text-accent" />
              {!collapsed && <span>Intelligence Workspace</span>}
            </NavLink>
            <NavLink to="/user/help-desk" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <HelpCircle size={20} />
              {!collapsed && <span>Help Desk</span>}
            </NavLink>
            <NavLink to="/user/profile" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <User size={20} />
              {!collapsed && <span>Profile</span>}
            </NavLink>
            <NavLink to="/admin/dashboard" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <LayoutDashboard size={20} />
              {!collapsed && <span>Admin View</span>}
            </NavLink>
          </nav>
          
          <div className="user-profile">
            <div className="avatar">{user?.username?.[0]?.toUpperCase() || 'U'}</div>
            {!collapsed && (
              <div className="user-info">
                <span className="user-name">{user?.username || 'User'}</span>
                <span className="user-role">Intelligence Node</span>
              </div>
            )}
            {!collapsed && (
              <button className="icon-btn logout" onClick={handleLogout} title="Logout">
                <LogOut size={18} />
              </button>
            )}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <Outlet context={{ sessionId, setSessionId }} />
      </main>
    </div>
  );
};

export default UserLayout;
