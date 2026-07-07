import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { HelpCircle, FileText, MessageCircle, ExternalLink, ArrowLeft, Send, Paperclip } from 'lucide-react';
import { useTickets } from '../context/TicketContext';

const HelpDesk = () => {
  const [showTicketForm, setShowTicketForm] = useState(false);
  const [subject, setSubject] = useState('');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState('');
  const [priority, setPriority] = useState('Medium');
  
  const { addTicket } = useTickets();
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!subject.trim() || !description.trim() || !category) return;

    addTicket({
      subject,
      description,
      category,
      priority
    });

    // Reset and navigate
    setSubject('');
    setDescription('');
    setCategory('');
    setPriority('Medium');
    setShowTicketForm(false);
    navigate('/user/profile');
  };

  return (
    <div className="admin-dashboard">
      {!showTicketForm ? (
        <>
          <div className="dashboard-header mb-8 text-center max-w-2xl mx-auto">
            <div className="flex justify-center mb-4">
              <HelpCircle size={48} className="text-accent-primary" />
            </div>
            <h1 className="dashboard-title text-3xl">How can we help?</h1>
            <p className="text-muted mt-2">Search our knowledge base or get in touch with support.</p>
            
            <div className="mt-6 relative">
              <input 
                type="text" 
                placeholder="Search for articles..." 
                className="w-full h-12 bg-bg-secondary border border-border-color rounded-lg px-4 text-text-primary focus:outline-none focus:border-accent-primary shadow-sm"
              />
            </div>
          </div>

          <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-6 mt-12">
            <div className="bg-bg-secondary border border-border-color rounded-lg p-6 hover:border-accent-primary transition-colors cursor-pointer group shadow-sm hover:shadow-md">
              <FileText size={24} className="text-accent-primary mb-4" />
              <h3 className="text-lg font-semibold text-text-primary mb-2 group-hover:text-accent-primary transition-colors">Documentation</h3>
              <p className="text-sm text-text-muted mb-4">Read detailed guides on how to use DocIntel AI, connect your data sources, and manage your account.</p>
              <div className="text-sm text-accent-primary font-medium flex items-center gap-1">
                Browse Articles <ExternalLink size={14} />
              </div>
            </div>

            <div 
              className="bg-bg-secondary border border-border-color rounded-lg p-6 hover:border-accent-primary transition-colors cursor-pointer group shadow-sm hover:shadow-md"
              onClick={() => setShowTicketForm(true)}
            >
              <MessageCircle size={24} className="text-accent-primary mb-4" />
              <h3 className="text-lg font-semibold text-text-primary mb-2 group-hover:text-accent-primary transition-colors">Contact Support</h3>
              <p className="text-sm text-text-muted mb-4">Can't find what you're looking for? Our enterprise support team is available 24/7 to assist you.</p>
              <div className="text-sm text-accent-primary font-medium flex items-center gap-1">
                Open a Ticket <ExternalLink size={14} />
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="max-w-3xl mx-auto">
          <button 
            className="flex items-center gap-2 text-text-muted hover:text-text-primary mb-6 transition-colors font-medium text-sm"
            onClick={() => setShowTicketForm(false)}
          >
            <ArrowLeft size={16} /> Back to Help Desk
          </button>

          <div className="bg-bg-secondary border border-border-color rounded-lg p-8 shadow-lg">
            <div className="border-b border-border-color pb-6 mb-6">
              <h2 className="text-2xl font-semibold text-text-primary mb-2 flex items-center gap-3">
                <div className="bg-accent-primary/20 p-2 rounded-lg text-accent-primary">
                  <MessageCircle size={24} />
                </div>
                Submit a Support Request
              </h2>
              <p className="text-text-muted text-sm ml-12">
                Our team typically responds within 2-4 business hours for enterprise customers.
              </p>
            </div>

            <form onSubmit={handleSubmit} className="flex flex-col gap-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-semibold text-text-primary mb-2" htmlFor="category">
                    Category <span className="text-error">*</span>
                  </label>
                  <select 
                    id="category"
                    value={category}
                    onChange={(e) => setCategory(e.target.value)}
                    className="w-full h-11 bg-bg-primary border border-border-color rounded-md px-3 text-text-primary focus:outline-none focus:border-accent-primary shadow-sm transition-colors"
                    required
                  >
                    <option value="">Select a category...</option>
                    <option value="Technical Issue">Technical Issue</option>
                    <option value="Billing">Billing & Subscription</option>
                    <option value="Feature Request">Feature Request</option>
                    <option value="Account Access">Account Access</option>
                    <option value="Other">Other</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-text-primary mb-2" htmlFor="priority">
                    Priority <span className="text-error">*</span>
                  </label>
                  <select 
                    id="priority"
                    value={priority}
                    onChange={(e) => setPriority(e.target.value)}
                    className="w-full h-11 bg-bg-primary border border-border-color rounded-md px-3 text-text-primary focus:outline-none focus:border-accent-primary shadow-sm transition-colors"
                    required
                  >
                    <option value="Low">Low - Not blocking work</option>
                    <option value="Medium">Medium - Partially blocking work</option>
                    <option value="High">High - Blocking essential work</option>
                    <option value="Urgent">Urgent - Complete system outage</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-semibold text-text-primary mb-2" htmlFor="subject">
                  Subject <span className="text-error">*</span>
                </label>
                <input 
                  id="subject"
                  type="text" 
                  value={subject}
                  onChange={(e) => setSubject(e.target.value)}
                  className="w-full h-11 bg-bg-primary border border-border-color rounded-md px-3 text-text-primary focus:outline-none focus:border-accent-primary shadow-sm transition-colors"
                  placeholder="Brief summary of the issue"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-text-primary mb-2" htmlFor="description">
                  Description <span className="text-error">*</span>
                </label>
                <textarea 
                  id="description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="w-full min-h-[200px] h-48 bg-bg-primary border border-border-color rounded-md p-4 text-text-primary focus:outline-none focus:border-accent-primary shadow-sm transition-colors"
                  placeholder="Please provide steps to reproduce, error messages, and any other relevant context..."
                  required
                />
              </div>

              <div className="border border-dashed border-border-color rounded-md p-4 bg-bg-primary flex flex-col items-center justify-center gap-2 text-text-muted hover:border-accent-primary transition-colors cursor-pointer group">
                <Paperclip size={20} className="group-hover:text-accent-primary transition-colors" />
                <span className="text-sm">Click to attach files or drag and drop</span>
                <span className="text-xs">Max file size: 25MB</span>
              </div>

              <div className="flex justify-end mt-2 pt-6 border-t border-border-color">
                <button 
                  type="button"
                  onClick={() => setShowTicketForm(false)}
                  className="px-6 py-2 rounded-md text-text-primary hover:bg-bg-tertiary transition-colors mr-3 font-medium text-sm"
                >
                  Cancel
                </button>
                <button 
                  type="submit" 
                  className="bg-accent-primary hover:bg-accent-hover text-bg-primary font-semibold px-6 py-2 rounded-md transition-all flex items-center gap-2 shadow-md hover:shadow-lg active:scale-[0.98]"
                >
                  <Send size={16} />
                  Submit Request
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default HelpDesk;
