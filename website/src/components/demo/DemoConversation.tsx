import { useEffect, useRef } from "react";

export interface DemoMessage {
  id: string;
  role: "akane" | "you";
  text: string;
  time: string;
  preview?: boolean;
}

interface DemoConversationProps {
  messages: readonly DemoMessage[];
  generating: boolean;
}

export function DemoConversation({ messages, generating }: DemoConversationProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest" });
  }, [messages]);

  return <section className="demo-conversation demo-panel" aria-labelledby="demo-conversation-title">
    <header className="demo-sidebar-header">
      <div className="demo-sidebar-identity"><span className="demo-sidebar-avatar" aria-hidden="true">A</span><div><strong>Akane</strong><span><i className="demo-status-dot live" aria-hidden="true" />{generating ? "Responding" : "Online"}</span></div></div>
    </header>
    <div className="demo-sidebar-section">
      <p className="demo-sidebar-label">Chat</p>
      <strong id="demo-conversation-title">Conversation history</strong>
    </div>
    <div
      className="demo-messages"
      role="log"
      aria-live="polite"
      aria-relevant="additions text"
      aria-busy={generating}
    >
      {messages.length === 0 && <div className="demo-history-empty"><span aria-hidden="true">✦</span><strong>A quiet place to begin.</strong><p>Your temporary conversation will appear here. Nothing in a guest session touches the owner profile.</p></div>}
      {messages.map((message, index) => <article className={`demo-message ${message.role} ${index === messages.length - 1 ? "current" : ""}`} key={message.id}>
        <div className="demo-message-avatar" aria-hidden="true">{message.role === "akane" ? "A" : "You"}</div>
        <div className="demo-message-body">
          <div className="demo-message-meta">
            <strong>{message.role === "akane" ? "Akane" : "You"}</strong>
            {message.preview && <span className="demo-preview-label">Preview</span>}
          </div>
          {message.text
            ? <p>{message.text}</p>
            : <p className="demo-thinking"><span aria-hidden="true"><i /><i /><i /></span><span className="sr-only">Akane is thinking</span></p>}
        </div>
      </article>)}
      <div ref={endRef} />
    </div>
    <div className="demo-sidebar-footer"><span>Guest Session</span><span>Private</span><small>Temporary memory</small></div>
  </section>;
}
