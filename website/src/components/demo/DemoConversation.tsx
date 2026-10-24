import { useLayoutEffect, useRef, useState } from "react";
import { projectConfig } from "../../config/project";

export interface DemoMessage {
  id: string;
  role: "akane" | "you";
  text: string;
  time: string;
  preview?: boolean;
  interrupted?: boolean;
}

interface DemoConversationProps {
  messages: readonly DemoMessage[];
  generating: boolean;
  connection: "connecting" | "live" | "offline" | "preview";
  onSuggestion: (message: string) => void;
}

export function DemoConversation({ messages, generating, connection, onSuggestion }: DemoConversationProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const followRef = useRef(true);
  const lastUserRef = useRef<string>();
  const [showLatest, setShowLatest] = useState(false);
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

  function scrollToLatest() {
    const container = scrollRef.current;
    if (container) container.scrollTop = container.scrollHeight;
    followRef.current = true;
    setShowLatest(false);
  }

  useLayoutEffect(() => {
    const lastUserId = messages.filter(message => message.role === "you").at(-1)?.id;
    if (lastUserId !== lastUserRef.current) followRef.current = true;
    lastUserRef.current = lastUserId;
    if (followRef.current) scrollToLatest();
    else setShowLatest(true);
  }, [messages]);

  return <section className="demo-conversation demo-panel" aria-labelledby="demo-conversation-title">
    <header className="demo-sidebar-header">
      <div className="demo-sidebar-identity"><img className="demo-sidebar-avatar" src={`${projectConfig.basePath}assets/akane-logo-192.png`} width="36" height="36" alt="" /><div><strong id="demo-conversation-title">Akane</strong><span>{previewMode ? "Simulated preview" : "Your guest conversation"}</span></div></div>
      <span className={`demo-connection-label ${connection}`}><i aria-hidden="true" />{connection === "live" ? "Live Akane" : connection === "preview" ? "Simulated preview" : connection === "connecting" ? "Connecting" : "Offline"}</span>
    </header>
    <div
      className="demo-messages"
      ref={scrollRef}
      tabIndex={0}
      aria-label="Conversation history"
      onScroll={event => {
        const container = event.currentTarget;
        followRef.current = container.scrollHeight - container.scrollTop - container.clientHeight < 48;
        setShowLatest(!followRef.current);
      }}
      role="log"
      aria-live="polite"
      aria-relevant="additions text"
      aria-busy={generating}
    >
      {messages.length === 0 && <div className="demo-history-empty"><img className="demo-welcome-avatar" src={`${projectConfig.basePath}assets/akane-logo-192.png`} width="64" height="64" alt="" /><strong>{emptyTitle}</strong><p>{emptyText}</p>{(connection === "live" || previewMode) && <div className="demo-suggestions">{["Hi Akane. What’s on your mind?", "What makes a good rainy day?"].map(text => <button type="button" key={text} onClick={() => { onSuggestion(text); document.getElementById("demo-message")?.focus(); }}>{text}<span aria-hidden="true">↗</span></button>)}</div>}</div>}
      {messages.map((message, index) => <article className={`demo-message ${message.role} ${index === messages.length - 1 ? "current" : ""}`} key={message.id}>
        <div className="demo-message-avatar" aria-hidden="true">{message.role === "akane" ? <img src={`${projectConfig.basePath}assets/akane-logo-192.png`} width="30" height="30" alt="" /> : "You"}</div>
        <div className="demo-message-body">
          <div className="demo-message-meta">
            <strong>{message.role === "akane" ? "Akane" : "You"}</strong>
            {message.preview && <span className="demo-preview-label">Preview</span>}
            {message.interrupted && <span className="demo-preview-label">Stopped · partial reply</span>}
            <span className="demo-message-time">{message.time}</span>
          </div>
          {message.text
            ? <p>{message.text}</p>
            : <p className="demo-thinking"><span aria-hidden="true"><i /><i /><i /></span><span>Thinking…</span></p>}
        </div>
      </article>)}
    </div>
    {showLatest && <button className="demo-latest" type="button" onClick={scrollToLatest}>Latest messages <span aria-hidden="true">↓</span></button>}
  </section>;
}
