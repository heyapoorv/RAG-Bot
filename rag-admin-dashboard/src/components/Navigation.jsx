import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

export default function Navigation() {
  const location = useLocation();
  const navigate = useNavigate();
  const { isAuthenticated, user, logout } = useAuth();

  const handleLogout = () => {
    logout();
    navigate("/");
  };

  return (
    <nav className="main-nav">
      <div className="nav-container">
        <Link to="/" className="nav-brand">
          RAG Query
        </Link>
        <div className="nav-links">
          <Link to="/" className={location.pathname === "/" ? "active" : ""}>
            Home
          </Link>
          <Link to="/chat" className={location.pathname === "/chat" ? "active" : ""}>
            Chat
          </Link>
          {isAuthenticated && user?.role === 'admin' && (
            <Link to="/admin" className={location.pathname === "/admin" ? "active" : ""}>
              Admin
            </Link>
          )}
          {isAuthenticated ? (
            <div className="nav-auth-section">
              <span className="nav-user">
                Welcome, {user?.username}
              </span>
              <button onClick={handleLogout} className="btn btn-secondary">
                Logout
              </button>
            </div>
          ) : (
            <Link to="/login" className={location.pathname === "/login" ? "active" : ""}>
              Login
            </Link>
          )}
        </div>
      </div>
    </nav>
  );
}