import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, Plus, Paperclip, FileText, X, ChevronDown, ChevronRight, 
  Copy, RotateCcw, AlertCircle, Check
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './ChatInterface.css';

const INITIAL_MESSAGES = [
  {
    id: 1,
    role: 'assistant',
    content: 'Hello. I am your DocIntel AI assistant. You can upload documents or ask me questions about your connected knowledge base.',
    citations: []
  }
];

import { useAuth } from '../context/AuthContext';
import { useOutletContext } from 'react-router-dom';
import axios from 'axios';

const ChatInterface = () => {
  const { user, token } = useAuth();
  const { sessionId } = useOutletContext() || { sessionId: `session_${Date.now()}` };
  
  const [messages, setMessages] = useState(INITIAL_MESSAGES);
  
  // Clear or load messages when sessionId changes
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await axios.get(`${API_URL}/query/history/${sessionId}`);
        if (res.data.messages && res.data.messages.length > 0) {
          const formatted = res.data.messages.map((m, i) => ({
            id: i + 2,
            role: m.role,
            content: m.content,
            citations: [] // we aren't saving citations in memory yet
          }));
          setMessages([...INITIAL_MESSAGES, ...formatted]);
        } else {
          setMessages(INITIAL_MESSAGES);
        }
      } catch (err) {
        setMessages(INITIAL_MESSAGES);
      }
    };
    fetchHistory();
  }, [sessionId]);

  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [attachments, setAttachments] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [expandedCitations, setExpandedCitations] = useState({});
  const messagesEndRef = useRef(null);

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFiles = async (files) => {
    const newAttachments = files.map(f => ({
      file: f,
      name: f.name,
      size: (f.size / 1024 / 1024).toFixed(2) + ' MB',
      progress: 0,
      status: 'uploading'
    }));
    
    setAttachments(prev => [...prev, ...newAttachments]);

    for (const att of newAttachments) {
      const formData = new FormData();
      formData.append('file', att.file);
      formData.append('session_id', user.username); // Using username as namespace

      try {
        await axios.post(`${API_URL}/upload/`, formData, {
          onUploadProgress: (progressEvent) => {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setAttachments(prev => prev.map(p => 
              p.name === att.name ? { ...p, progress } : p
            ));
          }
        });
        setAttachments(prev => prev.map(p => 
          p.name === att.name ? { ...p, status: 'complete', progress: 100 } : p
        ));
      } catch (error) {
        console.error('Upload failed:', error);
        setAttachments(prev => prev.map(p => 
          p.name === att.name ? { ...p, status: 'error' } : p
        ));
      }
    }
  };

  const removeAttachment = (name) => {
    setAttachments(prev => prev.filter(a => a.name !== name));
  };

  const toggleCitation = (msgId, citId) => {
    setExpandedCitations(prev => ({
      ...prev,
      [`${msgId}-${citId}`]: !prev[`${msgId}-${citId}`]
    }));
  };

  const fetchAIResponse = async (userMsg) => {
    setIsTyping(true);
    try {
      const response = await axios.post(`${API_URL}/query/`, {
        questions: [userMsg],
        namespace: user.username,
        session_id: sessionId
      });

      const data = response.data.answers[0];
      
      setIsTyping(false);
      setMessages(prev => [...prev, {
        id: Date.now(),
        role: 'assistant',
        content: data.answer,
        citations: data.citations ? data.citations.map((c, i) => ({
          id: i + 1,
          source: c.source,
          text: c.highlight
        })) : []
      }]);
    } catch (error) {
      console.error('Query failed:', error);
      setIsTyping(false);
      setMessages(prev => [...prev, {
        id: Date.now(),
        role: 'assistant',
        content: 'I encountered an error while processing your request. Please check if your documents are uploaded correctly.',
        citations: []
      }]);
    }
  };

  const handleSend = () => {
    if (!inputValue.trim() && attachments.length === 0) return;

    const newMsg = {
      id: Date.now(),
      role: 'user',
      content: inputValue,
      attachments: attachments.filter(a => a.status === 'complete').map(a => a.name)
    };

    setMessages(prev => [...prev, newMsg]);
    setInputValue('');
    setAttachments([]);
    fetchAIResponse(newMsg.content);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`message-wrapper ${msg.role}`}>
            {msg.role === 'assistant' && (
              <div className="assistant-avatar">
                <AlertCircle size={20} />
              </div>
            )}
            <div className="message-content" style={{ width: '100%' }}>
              <div className="message-bubble">
                {msg.attachments && msg.attachments.length > 0 && (
                  <div className="flex gap-2 mb-2">
                    {msg.attachments.map((att, i) => (
                      <div key={i} className="attachment-badge">
                        <FileText size={14} /> {att}
                      </div>
                    ))}
                  </div>
                )}
                <div className="markdown-body">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {msg.content}
                  </ReactMarkdown>
                </div>

                {msg.citations && msg.citations.length > 0 && (
                  <div className="citations-container">
                    {msg.citations.map((cit) => {
                      const isExpanded = expandedCitations[`${msg.id}-${cit.id}`];
                      return (
                        <div key={cit.id}>
                          <button 
                            className="citation-toggle"
                            onClick={() => toggleCitation(msg.id, cit.id)}
                          >
                            {isExpanded ? <ChevronDown size={14}/> : <ChevronRight size={14}/>}
                            [Citation {cit.id}] {cit.source}
                          </button>
                          {isExpanded && (
                            <div className="citation-box animate-slide-in">
                              <div className="citation-source">
                                <FileText size={14} /> {cit.source}
                              </div>
                              {cit.text}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
              
              {msg.role === 'assistant' && (
                <div className="message-actions">
                  <button className="action-btn" title="Copy"><Copy size={16}/></button>
                  <button className="action-btn" title="Regenerate"><RotateCcw size={16}/></button>
                </div>
              )}
            </div>
          </div>
        ))}
        
        {isTyping && (
          <div className="message-wrapper assistant">
            <div className="assistant-avatar"><AlertCircle size={20} /></div>
            <div className="message-bubble">
              <div className="typing-indicator">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area-wrapper">
        <div 
          className={`input-container ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {dragActive && (
            <div className="drag-overlay">
              <Plus size={24} /> Drop files to upload
            </div>
          )}

          {attachments.length > 0 && (
            <div className="attachments-preview">
              {attachments.map((file, idx) => (
                <div key={idx} className="attachment-badge relative">
                  <FileText size={14} />
                  <span>{file.name}</span>
                  <button 
                    className="remove-attachment ml-1"
                    onClick={() => removeAttachment(file.name)}
                  >
                    <X size={14} />
                  </button>
                  {file.progress < 100 && (
                    <div className="absolute bottom-0 left-0 w-full h-1 bg-bg-primary rounded-b-md overflow-hidden">
                      <div 
                        className="h-full bg-accent-primary transition-all duration-300"
                        style={{ width: `${file.progress}%` }}
                      />
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          <div className="input-row">
            <label className="attach-btn cursor-pointer">
              <Paperclip size={20} />
              <input 
                type="file" 
                multiple 
                className="hidden" 
                style={{ display: 'none' }}
                onChange={(e) => handleFiles(Array.from(e.target.files))}
              />
            </label>
            <textarea
              className="chat-input"
              placeholder="Message DocIntel AI..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
            />
            <button 
              className="send-btn" 
              onClick={handleSend}
              disabled={!inputValue.trim() && attachments.length === 0}
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
