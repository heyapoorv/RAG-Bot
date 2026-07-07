import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, TrendingDown, Activity, Clock, Database, 
  ShieldCheck, AlertTriangle, ChevronDown, ChevronUp
} from 'lucide-react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar
} from 'recharts';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';
import './Admin.css';

const AdminDashboard = () => {
  const { token } = useAuth();
  const [expandedLog, setExpandedLog] = useState(null);
  const [metrics, setMetrics] = useState({
    total_queries: 0,
    avg_latency_ms: 0,
    cache_hit_rate: 0,
    verification_rate: 0
  });
  const [failureLogs, setFailureLogs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const [statsRes, logsRes] = await Promise.all([
          axios.get(`${API_URL}/analytics/summary`, {
            headers: { Authorization: `Bearer ${token}` }
          }),
          axios.get(`${API_URL}/analytics/failures`, {
            headers: { Authorization: `Bearer ${token}` }
          })
        ]);
        setMetrics(statsRes.data);
        setFailureLogs(logsRes.data);
      } catch (err) {
        console.error("Failed to fetch admin stats:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, [token]);

  const toggleLog = (id) => {
    if (expandedLog === id) {
      setExpandedLog(null);
    } else {
      setExpandedLog(id);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full min-h-[400px]">
        <Activity className="animate-spin text-accent-primary" size={48} />
      </div>
    );
  }

  return (
    <div className="admin-dashboard">
      <div className="dashboard-header">
        <h1 className="dashboard-title">System Overview</h1>
        <p className="text-muted">Real-time metrics and performance analytics</p>
      </div>

      {/* Metrics Cards */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-header">
            <span>Total Queries</span>
            <Activity size={18} className="text-accent" />
          </div>
          <div className="metric-value">{metrics.total_queries?.toLocaleString()}</div>
          <div className="metric-trend trend-up">
            <TrendingUp size={14} /> Live tracking
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span>Avg Latency</span>
            <Clock size={18} className="text-accent" />
          </div>
          <div className="metric-value">{metrics.avg_latency_ms}ms</div>
          <div className="metric-trend trend-down">
            <TrendingDown size={14} /> Optimized
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span>Cache Hit Rate</span>
            <Database size={18} className="text-accent" />
          </div>
          <div className="metric-value">{(metrics.cache_hit_rate * 100).toFixed(1)}%</div>
          <div className="metric-trend trend-up">
            <TrendingUp size={14} /> Efficient
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span>Verification Rate</span>
            <ShieldCheck size={18} className="text-accent" />
          </div>
          <div className="metric-value">{(metrics.verification_rate * 100).toFixed(1)}%</div>
          <div className="metric-trend trend-up">
            <TrendingUp size={14} /> Accurate
          </div>
        </div>
      </div>

      {/* Failure Logs */}
      <div className="failure-logs">
        <h3 className="logs-title">
          <AlertTriangle size={20} className="text-warning" />
          Recent Failure Logs
        </h3>
        <p className="text-muted mb-4 text-sm" style={{ marginBottom: '16px', fontSize: '0.875rem' }}>
          Queries flagged for incorrect generation or low confidence.
        </p>
        
        <div className="logs-list">
          {failureLogs.length === 0 ? (
            <p className="text-muted italic text-center py-8">No failures recorded yet.</p>
          ) : (
            failureLogs.map(log => (
              <div key={log.timestamp} className="log-item">
                <div className="log-header" onClick={() => toggleLog(log.timestamp)}>
                  <div className="flex gap-4 items-center">
                    <span className="log-question">"{log.question}"</span>
                  </div>
                  <div className="flex gap-4 items-center">
                    <span className="log-time">{new Date(log.timestamp * 1000).toLocaleTimeString()}</span>
                    {expandedLog === log.timestamp ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </div>
                </div>
                
                {expandedLog === log.timestamp && (
                  <div className="log-details animate-slide-in">
                    <div>
                      <div className="log-section-title">Generated Answer</div>
                      <div className="log-answer">{log.answer}</div>
                    </div>
                    <div>
                      <div className="log-section-title">Namespace / Model</div>
                      <div className="log-debug">
                        Namespace: {log.namespace}<br/>
                        Latency: {log.latency_ms.toFixed(2)}ms
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
