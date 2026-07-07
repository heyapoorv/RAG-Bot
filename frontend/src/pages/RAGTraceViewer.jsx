import React, { useState, useEffect } from 'react';
import { 
  FileText, Search, Activity, Network, ChevronRight, SlidersHorizontal
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';
import './Admin.css';

const RAGTraceViewer = () => {
  const { token } = useAuth();
  const [traces, setTraces] = useState([]);
  const [activeTrace, setActiveTrace] = useState(null);
  const [loading, setLoading] = useState(true);
  const [detailsLoading, setDetailsLoading] = useState(false);

  useEffect(() => {
    const fetchTraces = async () => {
      try {
        const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const res = await axios.get(`${API_URL}/admin/queries`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setTraces(res.data);
        if (res.data.length > 0) {
          fetchTraceDetails(res.data[0]._id);
        } else {
          setLoading(false);
        }
      } catch (err) {
        console.error("Failed to fetch traces:", err);
        setLoading(false);
      }
    };

    fetchTraces();
  }, [token]);

  const fetchTraceDetails = async (id) => {
    setDetailsLoading(true);
    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const res = await axios.get(`${API_URL}/admin/trace/${id}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setActiveTrace(res.data);
    } catch (err) {
      console.error("Failed to fetch trace details:", err);
    } finally {
      setDetailsLoading(false);
      setLoading(false);
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
      <div className="dashboard-header flex justify-between items-center mb-6">
        <div>
          <h1 className="dashboard-title">RAG Trace Viewer</h1>
          <p className="text-muted">Inspect retrieved chunks and similarity scores</p>
        </div>
      </div>

      <div className="trace-grid">
        {/* Left Panel - Traces List */}
        <div className="trace-panel">
          <div className="panel-header">
            <span>Recent Queries</span>
          </div>
          <div className="panel-content p-0 overflow-y-auto max-h-[600px]">
            {traces.length === 0 ? (
              <p className="p-4 text-muted italic text-sm">No queries found.</p>
            ) : (
              traces.map(trace => (
                <div 
                  key={trace._id} 
                  className={`p-4 border-b border-border-color cursor-pointer transition-colors ${activeTrace?.id === trace._id ? 'bg-bg-tertiary border-l-4 border-l-accent-primary' : 'hover:bg-bg-tertiary/50'}`}
                  onClick={() => fetchTraceDetails(trace._id)}
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className="font-semibold text-xs text-text-primary uppercase">{trace._id.slice(-6)}</span>
                    <span className="text-xs text-muted">{trace.latency_ms?.toFixed(0)}ms</span>
                  </div>
                  <div className="text-sm text-text-secondary line-clamp-2">"{trace.question}"</div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Right Panel - Chunk Details */}
        <div className="trace-panel">
          <div className="panel-header">
            <div className="flex items-center gap-2">
              <Network size={18} className="text-accent-primary" />
              <span>Retrieval Context</span>
            </div>
            {activeTrace && (
              <span className="text-xs bg-bg-tertiary px-2 py-1 rounded-md">
                {activeTrace.reranked_chunks?.length || 0} Chunks
              </span>
            )}
          </div>
          <div className="panel-content overflow-y-auto max-h-[600px]">
            {detailsLoading ? (
              <div className="flex items-center justify-center h-full py-20">
                <Activity className="animate-spin text-accent-primary" size={32} />
              </div>
            ) : activeTrace ? (
              <>
                <div className="mb-6 bg-bg-primary/50 p-4 rounded-lg border border-border-color">
                  <h4 className="text-xs font-semibold text-accent-primary uppercase mb-2">Query</h4>
                  <p className="text-sm text-text-primary font-medium">"{activeTrace.question}"</p>
                  
                  <h4 className="text-xs font-semibold text-accent-primary uppercase mt-4 mb-2">Answer</h4>
                  <p className="text-sm text-text-secondary">{activeTrace.answer}</p>
                </div>

                <div className="space-y-4">
                  {(activeTrace.reranked_chunks || []).map((chunk, index) => (
                    <div key={index} className="chunk-card bg-bg-secondary/50 border border-border-color rounded-lg p-4">
                      <div className="chunk-header flex justify-between items-center mb-3">
                        <div className="chunk-source flex items-center gap-2 text-xs font-semibold text-text-primary">
                          <FileText size={14} className="text-accent-primary" />
                          {chunk.source}
                        </div>
                        <div className="score-badge bg-accent-primary/10 text-accent-primary px-2 py-0.5 rounded text-xs">
                          Score: {chunk.score?.toFixed(3) || 'N/A'}
                        </div>
                      </div>
                      <div className="chunk-text text-sm text-text-secondary leading-relaxed bg-bg-primary/30 p-3 rounded border border-border-color/50">
                        {chunk.chunk_text}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="flex flex-col items-center justify-center h-full py-20 text-muted">
                <Search size={48} className="mb-4 opacity-20" />
                <p>Select a query to view its RAG trace</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RAGTraceViewer;
