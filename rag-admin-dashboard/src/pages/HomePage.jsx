import { Link } from "react-router-dom";

export default function HomePage() {
  return (
    <div className="home">
      <div className="hero">
        <h1>RAG Query System</h1>
        <p>Intelligent document analysis and query processing platform</p>
      </div>

      <div className="features-grid">
        <div className="feature-card">
          <div className="feature-icon">💬</div>
          <h3>Chat Interface</h3>
          <p>Interactive chat with AI-powered document analysis</p>
          <Link to="/chat" className="btn btn-primary">
            Start Chat
          </Link>
        </div>

        <div className="feature-card">
          <div className="feature-icon">�</div>
          <h3>Admin Access</h3>
          <p>Access administrative controls and analytics</p>
          <Link to="/login" className="btn">
            Admin Login
          </Link>
        </div>

        <div className="feature-card">
          <div className="feature-icon">📊</div>
          <h3>System Analytics</h3>
          <p>Monitor system performance and metrics</p>
          <Link to="/admin" className="btn">
            View Dashboard
          </Link>
        </div>
      </div>

      <div className="container">
        <div className="text-center">
          <h2 className="h3 mb-6">Quick Actions</h2>
          <div className="grid grid-cols-1 gap-4 max-w-2xl mx-auto">
            <Link to="/chat" className="card p-6">
              <div className="text-3xl mb-3">💬</div>
              <div className="font-medium">New Chat</div>
              <div className="text-sm text-muted mt-1">Start a conversation</div>
            </Link>
            <Link to="/login" className="card p-6">
              <div className="text-3xl mb-3">🔐</div>
              <div className="font-medium">Admin Login</div>
              <div className="text-sm text-muted mt-1">Access admin panel</div>
            </Link>
            <Link to="/admin" className="card p-6">
              <div className="text-3xl mb-3">📊</div>
              <div className="font-medium">Analytics</div>
              <div className="text-sm text-muted mt-1">View system metrics</div>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}