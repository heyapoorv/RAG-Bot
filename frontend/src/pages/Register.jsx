import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Activity, UserPlus } from 'lucide-react';
import './Auth.css';
import { useAuth } from '../context/AuthContext';

const Register = () => {
  const navigate = useNavigate();
  const { register } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleRegister = async (e) => {
    e.preventDefault();
    setError('');
    setIsSubmitting(true);
    
    const result = await register(username, password);
    
    if (result.success) {
      navigate('/login');
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
          <h2 className="auth-title">Request Access</h2>
          <p className="auth-subtitle">Join your organization's intelligence node</p>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/20 text-red-500 p-3 rounded-md mb-4 text-sm">
            {error}
          </div>
        )}

        <form onSubmit={handleRegister} className="flex flex-col gap-4">
          <div className="form-group">
            <label className="form-label" htmlFor="username">Username</label>
            <input 
              type="text" 
              id="username" 
              className="form-input" 
              placeholder="Pick a username" 
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="password">Password</label>
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
            <UserPlus size={20} />
            {isSubmitting ? 'Creating Account...' : 'Register Account'}
          </button>
        </form>

        <div className="auth-footer">
          Already have an account? <Link to="/login" className="text-accent-primary hover:underline">Sign In</Link>
        </div>
      </div>
    </div>
  );
};

export default Register;
