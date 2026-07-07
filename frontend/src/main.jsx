import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { TicketProvider } from './context/TicketContext.jsx'
import { AuthProvider } from './context/AuthContext.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <AuthProvider>
      <TicketProvider>
        <App />
      </TicketProvider>
    </AuthProvider>
  </StrictMode>,
)
