import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import { 
  FileText, Upload, Sparkles, AlertTriangle, ShieldCheck, 
  HelpCircle, CheckCircle, Split, ShieldAlert, BookOpen, Layers
} from 'lucide-react';
import './ChatInterface.css'; // Leverage existing premium styles

const DocumentIntelligence = () => {
  const { user } = useAuth();
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [docB, setDocB] = useState('');
  
  // UI Tabs & States
  const [activeTab, setActiveTab] = useState('summary'); // summary, clauses, risks, entities, compare
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [uploadSuccess, setUploadSuccess] = useState('');
  
  // Analysis Outputs
  const [summaryData, setSummaryData] = useState(null);
  const [clausesData, setClausesData] = useState(null);
  const [risksData, setRisksData] = useState(null);
  const [entitiesData, setEntitiesData] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const namespace = user?.username || 'global';

  useEffect(() => {
    fetchDocuments();
  }, [user]);

  const fetchDocuments = async () => {
    try {
      const token = localStorage.getItem('token');
      const res = await axios.get(`${API_URL}/documents?namespace=${namespace}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setDocuments(res.data);
      if (res.data.length > 0 && !selectedDoc) {
        setSelectedDoc(res.data[0].document_id);
      }
    } catch (err) {
      console.error("Failed to load documents", err);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setError('');
    setUploadSuccess('');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', namespace); // Scoped per user namespace

    try {
      const token = localStorage.getItem('token');
      await axios.post(`${API_URL}/upload/`, formData, {
        headers: { 
          'Content-Type': 'multipart/form-data',
          Authorization: `Bearer ${token}`
        }
      });
      setUploadSuccess('Document successfully indexed!');
      await fetchDocuments();
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed.');
    } finally {
      setUploading(false);
    }
  };

  const runAnalysis = async (tab = activeTab) => {
    if (!selectedDoc) {
      setError('Please upload and select a document first.');
      return;
    }
    setLoading(true);
    setError('');
    
    const token = localStorage.getItem('token');
    const headers = { Authorization: `Bearer ${token}` };

    try {
      if (tab === 'summary') {
        const res = await axios.post(`${API_URL}/documents/summarize`, {
          document_id: selectedDoc,
          namespace: namespace,
          max_length: 250
        }, { headers });
        setSummaryData(res.data);
      } else if (tab === 'clauses') {
        const res = await axios.post(
          `${API_URL}/documents/clauses?document_id=${selectedDoc}&namespace=${namespace}`,
          {}, { headers }
        );
        setClausesData(res.data);
      } else if (tab === 'risks') {
        const res = await axios.post(
          `${API_URL}/documents/risks?document_id=${selectedDoc}&namespace=${namespace}`,
          {}, { headers }
        );
        setRisksData(res.data);
      } else if (tab === 'entities') {
        const res = await axios.post(
          `${API_URL}/documents/entities?document_id=${selectedDoc}&namespace=${namespace}`,
          {}, { headers }
        );
        setEntitiesData(res.data);
      } else if (tab === 'compare') {
        if (!docB) {
          setError('Please select a second document for comparison.');
          setLoading(false);
          return;
        }
        const res = await axios.post(`${API_URL}/documents/compare`, {
          document_id_a: selectedDoc,
          document_id_b: docB,
          namespace: namespace
        }, { headers });
        setComparisonData(res.data);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis task failed.');
    } finally {
      setLoading(false);
    }
  };

  // Trigger analysis when document or tab changes
  useEffect(() => {
    if (selectedDoc && activeTab !== 'compare') {
      runAnalysis(activeTab);
    }
  }, [selectedDoc, activeTab]);

  return (
    <div className="chat-container" style={{ padding: '24px', overflowY: 'auto', display: 'block' }}>
      <header className="page-header mb-6">
        <div>
          <h1 className="page-title text-glow flex-center justify-start" style={{ gap: '8px' }}>
            <Sparkles className="text-accent" />
            AI Document Intelligence Workspace
          </h1>
          <p className="page-subtitle">Perform summarization, audit clauses, score severity risks, and compare document versions.</p>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar: Documents Selector & Upload */}
        <div className="space-y-6">
          <div className="panel card-glow">
            <h2 className="panel-title mb-4 border-b pb-2 flex-center justify-start">
              <FileText size={18} className="text-accent mr-2" />
              Source Documents
            </h2>
            
            {/* Upload Area */}
            <div className="upload-container mb-4">
              <label className="flex-center flex-col justify-center border-2 border-dashed border-gray-600 rounded-lg p-4 cursor-pointer hover:border-accent transition-colors">
                <Upload size={24} className="text-secondary mb-2" />
                <span className="text-xs text-secondary text-center">
                  {uploading ? 'Processing file...' : 'Upload PDF, DOCX or TXT'}
                </span>
                <input 
                  type="file" 
                  className="hidden" 
                  onChange={handleFileUpload}
                  disabled={uploading}
                  accept=".pdf,.docx,.txt"
                />
              </label>
            </div>

            {uploadSuccess && <div className="alert alert-success text-xs p-2 mb-2">{uploadSuccess}</div>}
            {error && <div className="alert alert-error text-xs p-2 mb-2">{error}</div>}

            {/* Documents List */}
            <div className="space-y-2 mt-4">
              {documents.map((doc) => (
                <div 
                  key={doc.document_id} 
                  className={`chat-item ${selectedDoc === doc.document_id ? 'active' : ''}`}
                  onClick={() => setSelectedDoc(doc.document_id)}
                  style={{ cursor: 'pointer', padding: '10px' }}
                >
                  <FileText size={16} className="mr-2 text-accent" />
                  <div className="truncate text-sm w-full">
                    <div>{doc.filename}</div>
                    <span className="text-help text-xs">{doc.chunk_count} clauses indexed</span>
                  </div>
                </div>
              ))}
              {documents.length === 0 && (
                <p className="text-secondary text-center text-xs py-4">No documents found. Please upload above.</p>
              )}
            </div>
          </div>
        </div>

        {/* Workspace: Tabs & Result Panels */}
        <div className="lg:col-span-3 space-y-6">
          <div className="panel card-glow" style={{ minHeight: '500px' }}>
            {/* Action Selection Tabs */}
            <div className="flex-center border-b pb-2 mb-4 justify-start overflow-x-auto" style={{ gap: '16px' }}>
              <button 
                onClick={() => setActiveTab('summary')}
                className={`tab-btn pb-2 border-b-2 text-sm flex-center ${activeTab === 'summary' ? 'border-accent text-accent' : 'border-transparent text-secondary'}`}
              >
                <BookOpen size={16} className="mr-1" /> Summary
              </button>
              <button 
                onClick={() => setActiveTab('clauses')}
                className={`tab-btn pb-2 border-b-2 text-sm flex-center ${activeTab === 'clauses' ? 'border-accent text-accent' : 'border-transparent text-secondary'}`}
              >
                <Layers size={16} className="mr-1" /> Clauses
              </button>
              <button 
                onClick={() => setActiveTab('risks')}
                className={`tab-btn pb-2 border-b-2 text-sm flex-center ${activeTab === 'risks' ? 'border-accent text-accent' : 'border-transparent text-secondary'}`}
              >
                <ShieldAlert size={16} className="mr-1" /> Risk Analysis
              </button>
              <button 
                onClick={() => setActiveTab('compare')}
                className={`tab-btn pb-2 border-b-2 text-sm flex-center ${activeTab === 'compare' ? 'border-accent text-accent' : 'border-transparent text-secondary'}`}
              >
                <Split size={16} className="mr-1" /> Comparison
              </button>
            </div>

            {loading ? (
              <div className="flex-center flex-col justify-center" style={{ height: '350px' }}>
                <div className="spinner mb-4"></div>
                <p className="text-secondary text-sm">AI Agent is running analysis models...</p>
              </div>
            ) : (
              <div className="tab-content">
                {/* ── Summary Tab ── */}
                {activeTab === 'summary' && summaryData && (
                  <div className="space-y-4">
                    <div className="glass-panel p-4 rounded">
                      <h3 className="text-glow text-base font-semibold mb-2">Executive Summary</h3>
                      <p className="text-secondary leading-relaxed text-sm">{summaryData.summary}</p>
                    </div>

                    <div>
                      <h4 className="text-sm font-semibold mb-2 text-accent">Key Topics Extracted</h4>
                      <div className="flex flex-wrap gap-2">
                        {summaryData.key_topics?.map((topic, i) => (
                          <span key={i} className="badge badge-accent text-xs">{topic}</span>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* ── Clauses Tab ── */}
                {activeTab === 'clauses' && clausesData && (
                  <div className="space-y-4">
                    <h3 className="text-glow text-base font-semibold">Extracting Key Agreement Terms</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left border-collapse text-xs">
                        <thead>
                          <tr className="border-b border-gray-700">
                            <th className="py-2 px-3">Title</th>
                            <th className="py-2 px-3">Type</th>
                            <th className="py-2 px-3">Text preview</th>
                            <th className="py-2 px-3">Risk Level</th>
                          </tr>
                        </thead>
                        <tbody>
                          {clausesData.clauses?.map((c, i) => (
                            <tr key={i} className="border-b border-gray-800 hover:bg-gray-900/50">
                              <td className="py-3 px-3 font-semibold text-accent">{c.title}</td>
                              <td className="py-3 px-3">{c.type}</td>
                              <td className="py-3 px-3 text-secondary max-w-xs truncate">{c.text}</td>
                              <td className="py-3 px-3">
                                <span className={`badge ${c.risk_level === 'high' ? 'badge-error' : c.risk_level === 'medium' ? 'badge-warning' : 'badge-success'}`}>
                                  {c.risk_level}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* ── Risk Tab ── */}
                {activeTab === 'risks' && risksData && (
                  <div className="space-y-6">
                    <div className="flex-center justify-between border-b pb-3">
                      <h3 className="text-glow text-base font-semibold">Identified Legal & Financial Vulnerabilities</h3>
                      <div className="flex-center">
                        <span className="text-sm text-secondary mr-2">Severity Rating:</span>
                        <span className={`text-base font-bold px-3 py-1 rounded ${risksData.overall_severity === 'high' ? 'bg-red-950 text-red-400 border border-red-500' : 'bg-green-950 text-green-400 border border-green-500'}`}>
                          {risksData.overall_severity?.toUpperCase()} ({risksData.risk_score}/100)
                        </span>
                      </div>
                    </div>

                    <div className="space-y-4">
                      {risksData.risks?.map((risk, i) => (
                        <div key={i} className="glass-panel p-4 rounded border-l-4 border-red-500">
                          <div className="flex-center justify-between mb-2">
                            <h4 className="font-semibold text-sm text-glow flex-center">
                              <ShieldAlert className="text-red-500 mr-2" size={16} />
                              {risk.risk_title || `Risk Item #${i+1}`}
                            </h4>
                            <span className="badge badge-error text-xs">severity: {risk.severity}</span>
                          </div>
                          <p className="text-secondary text-xs mb-2">{risk.description}</p>
                          <div className="mt-2 text-xs bg-gray-900/40 p-2 rounded">
                            <span className="font-semibold text-green-400">Mitigation: </span>
                            <span className="text-secondary">{risk.mitigation}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* ── Compare Tab ── */}
                {activeTab === 'compare' && (
                  <div className="space-y-4">
                    <div className="glass-panel p-4 rounded mb-4">
                      <h3 className="text-glow text-sm font-semibold mb-2">Select Target Document for Comparison</h3>
                      <div className="flex-center" style={{ gap: '16px' }}>
                        <select 
                          className="form-input" 
                          value={docB} 
                          onChange={(e) => setDocB(e.target.value)}
                        >
                          <option value="">-- Choose second document --</option>
                          {documents
                            .filter(d => d.document_id !== selectedDoc)
                            .map(d => (
                              <option key={d.document_id} value={d.document_id}>{d.filename}</option>
                            ))
                          }
                        </select>
                        <button onClick={() => runAnalysis('compare')} className="btn btn-primary">Compare Docs</button>
                      </div>
                    </div>

                    {comparisonData && (
                      <div className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div className="glass-panel p-4 rounded">
                            <h4 className="text-sm font-semibold text-accent mb-2">Key Modifications & Differences</h4>
                            <ul className="space-y-2 text-xs text-secondary list-disc pl-4">
                              {comparisonData.differences?.map((d, i) => <li key={i}>{d}</li>)}
                            </ul>
                          </div>
                          <div className="glass-panel p-4 rounded">
                            <h4 className="text-sm font-semibold text-accent mb-2">Severity Impact Shift</h4>
                            <p className="text-xs text-secondary mb-3">{comparisonData.severity_impact}</p>
                            <span className={`badge ${comparisonData.conflict_detected ? 'badge-error' : 'badge-success'}`}>
                              {comparisonData.conflict_detected ? '⚠️ Structural Conflicts Found' : '✅ Consistent Terms'}
                            </span>
                          </div>
                        </div>

                        <div>
                          <h4 className="text-sm font-semibold mb-2 text-glow">Shared Commonalities</h4>
                          <div className="space-y-2">
                            {comparisonData.similarities?.map((sim, i) => (
                              <div key={i} className="text-xs bg-gray-900/30 p-2 border border-gray-800 rounded text-secondary">
                                {sim}
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentIntelligence;
