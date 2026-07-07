import { useState } from "react";
import { sendQuery } from "../api/client";

export default function useStreamChat() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  const typeEffect = async (text, callback) => {
    let output = "";

    for (let i = 0; i < text.length; i++) {
      output += text[i];
      callback(output);
      await sleep(10); // typing speed
    }
  };

  const sendMessage = async (text, file) => {
    if (!text && !file) return;

    setMessages((prev) => [...prev, { role: "user", text }]);
    setLoading(true);

    try {
      const res = await sendQuery({
        questions: [text],
        session_id: "user-1",
      });

      const answer = res.data?.answers?.[0]?.answer || "No response";

      let streamedText = "";

      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "" },
      ]);

      await typeEffect(answer, (val) => {
        streamedText = val;

        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1].text = streamedText;
          return updated;
        });
      });
    } catch (err) {
      console.error("Stream chat error:", err);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Error occurred" },
      ]);
    }

    setLoading(false);
  };

  return { messages, sendMessage, loading };
}