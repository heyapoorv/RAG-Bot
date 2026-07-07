import { Link, useLocation } from "react-router-dom";

export default function Sidebar() {
  const location = useLocation();

  return (
    <div className="sidebar">
      <h3>RAG Admin</h3>
      <nav>
        <Link to="/" className={location.pathname === "/" ? "active" : ""}>
          🏠 Home
        </Link>
        <Link to="/chat" className={location.pathname === "/chat" ? "active" : ""}>
          💬 Chat
        </Link>
        <Link to="/admin" className={location.pathname === "/admin" ? "active" : ""}>
          📊 Dashboard
        </Link>
        <Link to="/login" className={location.pathname === "/login" ? "active" : ""}>
          🔐 Login
        </Link>
      </nav>
    </div>
  );
}