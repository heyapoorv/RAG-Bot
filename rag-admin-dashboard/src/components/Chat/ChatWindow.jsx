import Message from "./Message";

export default function ChatWindow({ messages }) {
  return (
    <div className="messages">
      {messages.map((m, i) => (
        <Message key={i} msg={m} />
      ))}
    </div>
  );
}