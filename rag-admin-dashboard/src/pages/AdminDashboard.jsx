import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { getAnalytics, getFailures } from "../api/client";
import Sidebar from "../components/Admin/Sidebar";
import StatCard from "../components/Admin/StatCard";
import "../styles/admin.css";

export default function AdminDashboard() {
  const [stats, setStats] = useState({});
  const [failures, setFailures] = useState([]);
  const [loading, setLoading] = useState(true);
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const loadData = async () => {
      try {
        const [analyticsRes, failuresRes] = await Promise.all([
          getAnalytics(),
          getFailures()
        ]);

        setStats(analyticsRes.data || {});
        setFailures(failuresRes.data || []);
      } catch (error) {
        console.error("Failed to load dashboard data:", error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const handleLogout = () => {
    logout();
    navigate("/");
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="loading-spinner mx-auto mb-4"></div>
        <p className="text-secondary">Loading dashboard...</p>
      </div>
    );
  }

  return (
    <div className="admin">
      <Sidebar />
      <div className="main-content">
          <div className="flex justify-between items-center mb-8">
            <div>
              <h1 className="text-3xl font-semibold mb-2">System Analytics</h1>
              <p className="text-secondary">Welcome back, {user?.username}!</p>
            </div>
            <button onClick={handleLogout} className="btn">
              Logout
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <StatCard
              label="Total Queries"
              value={stats.total_queries || 0}
              icon="📊"
            />
            <StatCard
              label="Average Latency"
              value={`${stats.avg_latency_ms || 0}ms`}
              icon="⚡"
            />
            <StatCard
              label="Cache Hit Rate"
              value={`${Math.round((stats.cache_hit_rate || 0) * 100)}%`}
              icon="🎯"
            />
            <StatCard
              label="Verification Rate"
              value={`${Math.round((stats.verification_rate || 0) * 100)}%`}
              icon="✅"
            />
          </div>

          <div className="card">
            <div className="card-header">
              <h2 className="text-xl font-semibold">Recent Failures</h2>
            </div>
            <div className="card-body">
              {failures.length === 0 ? (
                <div className="text-center py-8">
                  <p className="text-muted">No failures recorded</p>
                </div>
              ) : (
                <div className="gap-4">
                  {failures.map((f, i) => (
                    <div key={i} className="border rounded-lg p-4">
                      <div className="flex justify-between items-start mb-2">
                        <strong className="text-sm font-medium">{f.question}</strong>
                        <span className="text-xs text-muted">{f.timestamp || 'Recent'}</span>
                      </div>
                      <p className="text-sm text-secondary">{f.answer || f.error}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
   
  );
}