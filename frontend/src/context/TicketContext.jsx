import React, { createContext, useState, useContext } from 'react';

const TicketContext = createContext();

export const useTickets = () => useContext(TicketContext);

export const TicketProvider = ({ children }) => {
  const [tickets, setTickets] = useState([
    {
      id: 'TKT-1042',
      subject: 'Issue with connecting to Pinecone DB',
      status: 'Open',
      date: '2024-03-15',
      description: 'Unable to connect to the production vector database instance.'
    }
  ]);

  const addTicket = (ticket) => {
    setTickets([{
      ...ticket,
      id: `TKT-${Math.floor(1000 + Math.random() * 9000)}`,
      status: 'Open',
      date: new Date().toISOString().split('T')[0]
    }, ...tickets]);
  };

  return (
    <TicketContext.Provider value={{ tickets, addTicket }}>
      {children}
    </TicketContext.Provider>
  );
};
