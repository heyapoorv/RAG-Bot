import { useRef, useState } from "react";
import UploadAttachment from "../Upload/UploadAttachment.jsx";

export default function InputBox({ onSend }) {
  const [text, setText] = useState("");
  const fileRef = useRef(null);

  const handleSend = () => {
    onSend(text, fileRef.current?.files?.[0]);
    setText("");
    if (fileRef.current) fileRef.current.value = null;
  };

  return (
    <div className="input-box">

      <UploadAttachment fileRef={fileRef} />

      <input
        value={text}
        placeholder="Ask anything..."
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && handleSend()}
      />

      <button className="send-btn" onClick={handleSend}>Send</button>
    </div>
  );
}