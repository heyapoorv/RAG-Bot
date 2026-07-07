import AppRouter from "./routes/AppRouter";
import { AuthProvider } from "./contexts/AuthContext";
import "./index.css";
import "./styles/chat.css";
import "./styles/admin.css";

export default function App() {
  return (
    <AuthProvider>
      <AppRouter />
    </AuthProvider>
  );
}