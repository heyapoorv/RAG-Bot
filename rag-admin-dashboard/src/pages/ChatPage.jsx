import { useState } from "react";
import ChatWindow from "../components/Chat/ChatWindow";
import InputBox from "../components/Chat/InputBox";
import { sendQuery } from "../api/client";

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [sessionId] = useState("user-1");

  const handleSend = async (text, file) => {
    if (!text && !file) return;

    const userMsg = { role: "user", text };
    setMessages((prev) => [...prev, userMsg]);

    try {
      const res = await sendQuery({
        questions: [text],
        session_id: sessionId,
      });

      const answer = res.data.answers?.[0];

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: answer?.answer || "No response",
          sources: answer?.sources || [],
        },
      ]);
    } catch (err) {
      console.error("Chat error:", err);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Error generating response" },
      ]);
    }
  };

  return (
    <div className="chat-container">
      <ChatWindow messages={messages} />
      <InputBox onSend={handleSend} />
    </div>
  );
}