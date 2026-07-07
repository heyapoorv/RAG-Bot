import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navigation from "../components/Navigation";
import ProtectedRoute from "../components/ProtectedRoute";
import HomePage from "../pages/HomePage";
import ChatPage from "../pages/ChatPage";
import AdminDashboard from "../pages/AdminDashboard";
import Login from "../pages/Login";

export default function AppRouter() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-secondary">
        <Navigation />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/login" element={<Login />} />
            <Route
              path="/admin"
              element={
                <ProtectedRoute>
                  <AdminDashboard />
                </ProtectedRoute>
              }
            />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}