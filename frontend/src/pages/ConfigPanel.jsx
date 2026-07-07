import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import { 
  Settings, Save, RotateCcw, AlertTriangle, CheckCircle, 
  HelpCircle, Sliders, Server, Shield, History
} from 'lucide-react';
import './Admin.css';

const ConfigPanel = () => {
  const { user } = useAuth();
  const [config, setConfig] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  useEffect(() => {
    fetchConfig();
    fetchHistory();
  }, []);

  const fetchConfig = async () => {
    try {
      const token = localStorage.getItem('token');
      const res = await axios.get(`${API_URL}/config`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setConfig(res.data);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch system configuration.');
      setLoading(false);
    }
  };

  const fetchHistory = async () => {
    try {
      const token = localStorage.getItem('token');
      const res = await axios.get(`${API_URL}/config/history`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setHistory(res.data);
    } catch (err) {
      console.error("Failed to load config history", err);
    }
  };

  const handleChange = (key, val) => {
    setConfig(prev => ({
      ...prev,
      [key]: val
    }));
  };

  const handleSave = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError('');
    setSuccess('');
    try {
      const token = localStorage.getItem('token');
      await axios.post(`${API_URL}/config`, config, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setSuccess('Configuration updated successfully!');
      fetchConfig();
      fetchHistory();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to update configuration.');
    } finally {
      setSaving(false);
    }
  };

  const handleRollback = async (version) => {
    if (!window.confirm(`Are you sure you want to rollback to version ${version}?`)) return;
    setError('');
    setSuccess('');
    try {
      const token = localStorage.getItem('token');
      await axios.post(`${API_URL}/config/rollback/${version}`, {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setSuccess(`Successfully rolled back to version ${version}`);
      fetchConfig();
      fetchHistory();
    } catch (err) {
      setError(err.response?.data?.detail || 'Rollback failed.');
    }
  };

  if (loading) {
    return (
      <div className="flex-center" style={{ height: '70vh', flexDirection: 'column' }}>
        <div className="spinner mb-4"></div>
        <p className="text-secondary">Loading intelligence settings...</p>
      </div>
    );
  }

  return (
    <div className="admin-page">
      <header className="page-header mb-6">
        <div>
          <h1 className="page-title text-glow">System Settings</h1>
          <p className="page-subtitle">Configure the document processing, vector search, reranking and caching parameters.</p>
        </div>
      </header>

      {error && (
        <div className="alert alert-error mb-4 flex-center justify-start">
          <AlertTriangle size={18} className="mr-2" />
          <span>{error}</span>
        </div>
      )}

      {success && (
        <div className="alert alert-success mb-4 flex-center justify-start">
          <CheckCircle size={18} className="mr-2" />
          <span>{success}</span>
        </div>
      )}

      <form onSubmit={handleSave} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Main Settings Form */}
        <div className="lg:col-span-2 space-y-6">
          {/* Section: Document Ingestion */}
          <div className="panel card-glow">
            <div className="panel-header mb-4 border-b pb-2 flex-center justify-start">
              <Server className="text-accent mr-2" size={20} />
              <h2 className="panel-title">Ingestion & Chunking</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="form-label">Chunk Size (Words)</label>
                <input 
                  type="number" 
                  className="form-input"
                  value={config.CHUNK_SIZE || 180}
                  onChange={(e) => handleChange('CHUNK_SIZE', parseInt(e.target.value))}
                  min="50"
                  max="1000"
                />
                <span className="text-help">Target length for document child chunks.</span>
              </div>
              
              <div>
                <label className="form-label">Chunk Overlap (Sentences)</label>
                <input 
                  type="number" 
                  className="form-input"
                  value={config.CHUNK_OVERLAP || 2}
                  onChange={(e) => handleChange('CHUNK_OVERLAP', parseInt(e.target.value))}
                  min="0"
                  max="10"
                />
                <span className="text-help">Context overlap for sequential clauses.</span>
              </div>
            </div>
          </div>

          {/* Section: Retrieval & Reranker */}
          <div className="panel card-glow">
            <div className="panel-header mb-4 border-b pb-2 flex-center justify-start">
              <Sliders className="text-accent mr-2" size={20} />
              <h2 className="panel-title">Retrieval & Reranker</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="form-label">Base Retrieval Top-K</label>
                <input 
                  type="number" 
                  className="form-input"
                  value={config.RETRIEVAL_TOP_K || 10}
                  onChange={(e) => handleChange('RETRIEVAL_TOP_K', parseInt(e.target.value))}
                  min="1"
                  max="50"
                />
                <span className="text-help">Maximum chunks fetched from Pinecone before reranking.</span>
              </div>

              <div>
                <label className="form-label">Reranker Mode</label>
                <select 
                  className="form-input"
                  value={config.RERANKER_MODE || 'local'}
                  onChange={(e) => handleChange('RERANKER_MODE', e.target.value)}
                >
                  <option value="local">Local Cosine Similarity</option>
                  <option value="cross_encoder">Cross-Encoder (MS-Marco)</option>
                  <option value="hybrid">Hybrid (Score Combination)</option>
                </select>
                <span className="text-help">Reranking pipeline model selection.</span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="form-label">Reranker Model Name</label>
                <input 
                  type="text" 
                  className="form-input"
                  value={config.RERANKER_MODEL_NAME || ''}
                  onChange={(e) => handleChange('RERANKER_MODEL_NAME', e.target.value)}
                />
                <span className="text-help">HuggingFace cross-encoder endpoint model name.</span>
              </div>

              <div>
                <label className="form-label">Reranker Confidence Threshold</label>
                <input 
                  type="number" 
                  step="0.05"
                  className="form-input"
                  value={config.RERANKER_THRESHOLD || 0.3}
                  onChange={(e) => handleChange('RERANKER_THRESHOLD', parseFloat(e.target.value))}
                  min="0.0"
                  max="1.0"
                />
                <span className="text-help">Minimum relevance score to include in generation context.</span>
              </div>
            </div>
          </div>

          {/* Section: Cache & Verifier */}
          <div className="panel card-glow">
            <div className="panel-header mb-4 border-b pb-2 flex-center justify-start">
              <Shield className="text-accent mr-2" size={20} />
              <h2 className="panel-title">Semantic Cache & Verification</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
              <div className="flex-center justify-start mt-2">
                <input 
                  type="checkbox" 
                  id="cache_enabled"
                  className="form-checkbox mr-2"
                  checked={config.CACHE_ENABLED}
                  onChange={(e) => handleChange('CACHE_ENABLED', e.target.checked)}
                />
                <label htmlFor="cache_enabled" className="form-label mb-0 cursor-pointer">Enable Semantic Cache</label>
              </div>

              <div className="flex-center justify-start mt-2">
                <input 
                  type="checkbox" 
                  id="verification_enabled"
                  className="form-checkbox mr-2"
                  checked={config.VERIFICATION_ENABLED}
                  onChange={(e) => handleChange('VERIFICATION_ENABLED', e.target.checked)}
                />
                <label htmlFor="verification_enabled" className="form-label mb-0 cursor-pointer">Enable Answer Verification</label>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="form-label">Cache Similarity Threshold</label>
                <input 
                  type="number" 
                  step="0.01"
                  className="form-input"
                  value={config.CACHE_SIMILARITY_THRESHOLD || 0.9}
                  onChange={(e) => handleChange('CACHE_SIMILARITY_THRESHOLD', parseFloat(e.target.value))}
                  min="0.5"
                  max="1.0"
                />
                <span className="text-help">Threshold for returning cached answers semantic-matches.</span>
              </div>

              <div>
                <label className="form-label">Verification Mode</label>
                <select 
                  className="form-input"
                  value={config.VERIFICATION_MODE || 'strict'}
                  onChange={(e) => handleChange('VERIFICATION_MODE', e.target.value)}
                >
                  <option value="strict">Strict (Reject failed check)</option>
                  <option value="lenient">Lenient (Log check but return anyway)</option>
                </select>
                <span className="text-help">Response handling for negative verification results.</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Col: Operations Actions & Audit Trail */}
        <div className="space-y-6">
          <div className="panel card-glow">
            <h2 className="panel-title mb-4 border-b pb-2 flex-center justify-start">
              <Settings className="text-accent mr-2" size={18} />
              Operations
            </h2>
            
            <button 
              type="submit" 
              className="btn btn-primary w-full flex-center justify-center mb-3"
              disabled={saving}
            >
              <Save size={18} className="mr-2" />
              {saving ? 'Updating System...' : 'Commit Changes'}
            </button>

            <button 
              type="button" 
              onClick={fetchConfig}
              className="btn btn-outline w-full flex-center justify-center"
            >
              <RotateCcw size={18} className="mr-2" />
              Reset Form
            </button>
          </div>

          {/* Audit Logs */}
          <div className="panel card-glow">
            <h2 className="panel-title mb-4 border-b pb-2 flex-center justify-start">
              <History className="text-accent mr-2" size={18} />
              Audit Logs
            </h2>

            <div className="timeline-container" style={{ maxHeight: '350px', overflowY: 'auto' }}>
              {history.map((hist, i) => (
                <div key={i} className="timeline-item mb-4 pb-2 border-b last:border-0">
                  <div className="flex-center justify-between mb-1">
                    <span className="badge badge-accent">v{hist.version}</span>
                    <span className="text-help text-xs">{new Date(hist.updated_at).toLocaleString()}</span>
                  </div>
                  <p className="text-secondary text-xs mb-2">
                    Updated by: <strong>{hist.updated_by}</strong>
                  </p>
                  
                  {hist.version > 1 && (
                    <button 
                      type="button" 
                      onClick={() => handleRollback(hist.version)}
                      className="btn btn-outline text-xs px-2 py-1 h-auto flex-center inline-flex"
                      style={{ padding: '4px 8px', fontSize: '11px' }}
                    >
                      <RotateCcw size={12} className="mr-1" />
                      Restore Version
                    </button>
                  )}
                </div>
              ))}
              {history.length === 0 && (
                <p className="text-secondary text-center py-4 text-sm">No configuration history found.</p>
              )}
            </div>
          </div>
        </div>
      </form>
    </div>
  );
};

export default ConfigPanel;
