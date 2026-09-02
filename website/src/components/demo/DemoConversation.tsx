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
  connection: "connecting" | "live" | "offline" | "preview";
}

export function DemoConversation({ messages, generating, connection }: DemoConversationProps) {
  const endRef = useRef<HTMLDivElement>(null);
  const previewMode = connection === "preview";
  const emptyTitle = connection === "offline"
    ? "Akane is offline right now."
    : connection === "connecting"
      ? "Checking live availability…"
      : previewMode
        ? "Try the conversation layout."
        : "A quiet place to begin.";
  const emptyText = connection === "offline"
    ? "You can try Preview without sending anything to Akane."
    : connection === "connecting"
      ? "This usually takes only a moment."
      : previewMode
        ? "Preview responses are simulated and are not sent to Akane."
        : "Your first message automatically begins an isolated guest conversation.";

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest" });
  }, [messages]);

  return <section className="demo-conversation demo-panel" aria-labelledby="demo-conversation-title">
    <header className="demo-sidebar-header">
      <div className="demo-sidebar-identity"><span className="demo-sidebar-avatar" aria-hidden="true">A</span><div><strong id="demo-conversation-title">Akane</strong><span>Conversation</span></div></div>
    </header>
    <div
      className="demo-messages"
      role="log"
      aria-live="polite"
      aria-relevant="additions text"
      aria-busy={generating}
    >
      {messages.length === 0 && <div className="demo-history-empty"><span aria-hidden="true">✦</span><strong>{emptyTitle}</strong><p>{emptyText}</p></div>}
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
  </section>;
}
