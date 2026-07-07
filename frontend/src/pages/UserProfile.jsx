import React, { useState } from 'react';
import { User, Mail, Shield, Key, Edit2, Check, X, Ticket } from 'lucide-react';
import { useTickets } from '../context/TicketContext';
import { useAuth } from '../context/AuthContext';
import './Admin.css';

const UserProfile = () => {
  const { user } = useAuth();
  const { tickets } = useTickets();
  
  const [editing, setEditing] = useState(null);
  const [tempValue, setTempValue] = useState('');

  const handleEdit = (field, value) => {
    setEditing(field);
    setTempValue(value);
  };

  const handleSave = (field) => {
    // In a real app, you'd call an API here
    setEditing(null);
  };

  const handleCancel = () => {
    setEditing(null);
  };

  return (
    <div className="admin-dashboard">
      <div className="dashboard-header mb-8">
        <h1 className="dashboard-title">User Profile</h1>
        <p className="text-muted">Manage your personal information and support tickets</p>
      </div>

      <div className="max-w-3xl mx-auto flex flex-col gap-8">
        {/* Profile Details */}
        <div className="bg-bg-secondary border border-border-color rounded-lg p-6">
          <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
            <User size={20} className="text-accent-primary" />
            Personal Information
          </h3>

          <div className="flex flex-col gap-4">
            <div className="flex justify-between items-center py-3 border-b border-border-color">
              <div className="flex-1">
                <div className="font-medium text-text-primary">Username</div>
                <div className="text-sm text-text-muted mt-1">{user?.username || 'Guest'}</div>
              </div>
            </div>

            <div className="flex justify-between items-center py-3 border-b border-border-color">
              <div className="flex-1">
                <div className="font-medium text-text-primary">Email Address</div>
                <div className="text-sm text-text-muted mt-1 flex items-center gap-2">
                  {user?.email || `${user?.username}@company.com`}
                  <span className="text-xs bg-success/20 text-success px-2 py-0.5 rounded">Verified</span>
                </div>
              </div>
            </div>

            <div className="flex justify-between items-center py-3">
              <div className="flex-1">
                <div className="font-medium text-text-primary">Password</div>
                <div className="text-sm text-text-muted mt-1">••••••••</div>
              </div>
              <button className="bg-bg-tertiary hover:bg-border-hover text-text-primary px-4 py-2 rounded-md text-sm font-medium transition-colors">
                Change Password
              </button>
            </div>
          </div>
        </div>

        {/* Support Tickets */}
        <div className="bg-bg-secondary border border-border-color rounded-lg p-6">
          <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
            <Ticket size={20} className="text-accent-primary" />
            Support Tickets
          </h3>

          <div className="flex flex-col gap-4">
            {tickets.length === 0 ? (
              <p className="text-text-muted text-sm italic">No support tickets raised yet.</p>
            ) : (
              tickets.map((ticket) => (
                <div key={ticket.id} className="border border-border-color rounded-lg p-4 bg-bg-primary">
                  <div className="flex justify-between items-center mb-2">
                    <div className="font-semibold text-text-primary">{ticket.subject}</div>
                    <span className={`text-xs px-2 py-1 rounded font-medium ${ticket.status === 'Open' ? 'bg-warning/20 text-warning' : 'bg-success/20 text-success'}`}>
                      {ticket.status}
                    </span>
                  </div>
                  <div className="text-sm text-text-secondary mb-3">{ticket.description}</div>
                  <div className="flex justify-between items-center text-xs text-text-muted">
                    <span>ID: {ticket.id}</span>
                    <span>Date: {ticket.date}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

export default UserProfile;
