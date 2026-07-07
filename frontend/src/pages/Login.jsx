import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Activity, LogIn } from 'lucide-react';
import './Auth.css';
import { useAuth } from '../context/AuthContext';

const Login = () => {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setIsSubmitting(true);
    
    const result = await login(username, password);
    
    if (result.success) {
      navigate('/user/chat');
    } else {
      setError(result.error);
      setIsSubmitting(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="ambient-bg">
        <div className="ambient-blob-1"></div>
        <div className="ambient-blob-2"></div>
      </div>

      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-logo">
            <Activity size={32} />
            <span style={{ fontSize: '1.5rem', fontWeight: 600, color: 'var(--text-primary)' }}>DocIntel AI</span>
          </div>
          <h2 className="auth-title">Sign In</h2>
          <p className="auth-subtitle">Secure access to your intelligence node</p>
        </div>

        <div className="flex flex-col gap-3">
          <button className="sso-btn">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
            </svg>
            Continue with Google
          </button>

        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/20 text-red-500 p-3 rounded-md mb-4 text-sm">
            {error}
          </div>
        )}

        <form onSubmit={handleLogin} className="flex flex-col gap-4">
          <div className="form-group">
            <label className="form-label" htmlFor="username">Username</label>
            <input 
              type="text" 
              id="username" 
              className="form-input" 
              placeholder="Your username" 
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <div className="form-label">
              <label htmlFor="password">Password</label>
              <a href="#" className="text-accent-primary hover:underline">Forgot?</a>
            </div>
            <input 
              type="password" 
              id="password" 
              className="form-input" 
              placeholder="••••••••" 
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <button type="submit" className="auth-btn" disabled={isSubmitting}>
            <LogIn size={20} />
            {isSubmitting ? 'Authenticating...' : 'Enter Intelligence Suite'}
          </button>
        </form>

        <div className="auth-footer">
          Don't have an account? <Link to="/register" className="text-accent-primary hover:underline">Register</Link>
        </div>
      </div>
    </div>
  );
};

export default Login;
