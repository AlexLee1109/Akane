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
    <header className="demo-panel-header">
      <h2 id="demo-conversation-title">Current conversation</h2>
      {generating && <span className="demo-streaming-label">Streaming</span>}
    </header>
    <div
      className="demo-messages"
      role="log"
      aria-live="polite"
      aria-relevant="additions text"
      aria-busy={generating}
    >
      {messages.map(message => <article className={`demo-message ${message.role}`} key={message.id}>
        <div className="demo-message-avatar" aria-hidden="true">{message.role === "akane" ? "A" : "Y"}</div>
        <div className="demo-message-body">
          <div className="demo-message-meta">
            <strong>{message.role === "akane" ? "Akane" : "You"}</strong>
            {message.preview && <span className="demo-preview-label">Prerecorded preview</span>}
            <time>{message.time}</time>
          </div>
          {message.text
            ? <p>{message.text}</p>
            : <p className="demo-thinking"><span aria-hidden="true"><i /><i /><i /></span><span className="sr-only">Akane is thinking</span></p>}
        </div>
      </article>)}
      <div ref={endRef} />
    </div>
  </section>;
}
