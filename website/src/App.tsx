import { FormEvent, KeyboardEvent, ReactNode, useEffect, useRef, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { AkaneStage } from "./components/AkaneStage";
import { projectConfig } from "./config/project";
import { akaneClient, PublicApiError, type PublicHealth, type PublicSession } from "./lib/akaneClient";
import { canUseSpeech, speak, stopSpeech } from "./lib/speech";
import { clearGuestToken, getGuestToken, storeGuestToken } from "./lib/session";
import type { AkanePresentationState } from "./presentation";

const asset = `${projectConfig.basePath}assets/akane-hero.png`;
const logo = `${projectConfig.basePath}assets/akane-logo.png`;
const github = projectConfig.githubUrl;
const features = [
  ["✦", "Memory", "Preserves meaningful facts, preferences, conversations, and shared events."],
  ["♡", "Emotion", "Maintains a grounded emotional state that develops through conversation and experience."],
  ["◔", "Offscreen Life", "Chooses persistent activities and thoughts while you are away."],
  ["↗", "Autonomous Initiative", "Can decide when it has a grounded reason to begin a conversation."],
  ["◌", "Multi-Interface Companion", "Shares one identity and state across the popup, Discord, and web demo."],
];
const suggestions = ["Tell me about yourself.", "What are you doing right now?", "What is something you have been thinking about?", "Tell me an opinion you genuinely have."];
const stack = [
  ["Inference", "Gemma 4 E4B", "llama.cpp · GGUF"], ["Backend", "Python", "FastAPI streaming"], ["Persistence", "Atomic JSON", "state and history"], ["Interfaces", "React + Vite", "Popup · Discord"], ["Deployment", "Raspberry Pi 5", "local runtime"],
];

function Mark() { return <span className="mark" aria-hidden="true">✦</span>; }
function Logo() { return <img className="logo" src={logo} alt="Akane logo" />; }
function GithubLink({ children, className = "button secondary" }: { children: ReactNode; className?: string }) {
  return github ? <a className={className} href={github} target="_blank" rel="noreferrer">{children} <span aria-hidden="true">↗</span></a> : <span className={`${className} disabled`} aria-disabled="true">{children}</span>;
}

function Navbar() {
  const [open, setOpen] = useState(false);
  const links = [["/", "Home"], ["/demo", "Demo"], ["/technology", "Technology"]] as const;
  return <header className="site-header"><nav className="nav shell" aria-label="Primary navigation">
    <Link className="brand" to="/" onClick={() => setOpen(false)}><Logo /><strong>Akane</strong><span>AI COMPANION</span></Link>
    <button className="menu-button" aria-expanded={open} aria-controls="nav-links" onClick={() => setOpen(!open)}>{open ? "Close" : "Menu"}</button>
    <div id="nav-links" className={`nav-links ${open ? "open" : ""}`}>{links.map(([to, label]) => <NavLink key={to} to={to} end={to === "/"} onClick={() => setOpen(false)}>{label}</NavLink>)}<GithubLink className="nav-github">GitHub</GithubLink></div>
  </nav></header>;
}

function Footer() { return <footer className="footer shell"><div className="footer-brand"><Logo /><div><strong>Akane</strong><p>Local-first privacy: the static site does not bundle a model or personal Akane state.</p></div></div><div className="footer-links"><Link to="/">Home</Link><Link to="/demo">Demo</Link><Link to="/technology">Technology</Link><GithubLink className="plain-link">GitHub</GithubLink></div><small>© {new Date().getFullYear()} Akane</small></footer>; }

function Character({ compact = false }: { compact?: boolean }) { return <img className={`akane-image ${compact ? "compact" : ""}`} src={asset} alt="Akane, a blue-haired anime-style companion in a white jacket" />; }
function Eyebrow({ children }: { children: ReactNode }) { return <p className="eyebrow"><Mark /> {children}</p>; }

function HomePage() { return <main>
  <section className="hero shell"><div className="hero-copy"><Eyebrow>Open source · Local first · Privacy minded</Eyebrow><h1>Your local<br />AI companion,<br /><em>brought to life.</em></h1><p className="lead">Akane is a locally hosted AI companion with persistent memory, emotional continuity, autonomous activities, and natural conversation.</p><div className="actions"><Link className="button primary" to="/demo">Try Demo <span>→</span></Link><Link className="button secondary" to="/technology">View Technology</Link></div><Link className="message-teaser" to="/demo"><Mark /><span>Message Akane…<small>Try the browser demo</small></span><b>→</b></Link></div><div className="hero-art"><div className="speech-bubble">Good morning!<br /><strong>What would you like to talk about?</strong></div><Character /></div></section>
  <section className="section shell"><div className="section-heading"><Eyebrow>Built to be more than a chatbot</Eyebrow><h2>Features that make Akane special</h2></div><div className="feature-grid">{features.map(([icon, title, text]) => <article className="feature-card" key={title}><i>{icon}</i><h3>{title}</h3><p>{text}</p><Link to="/technology">Learn more →</Link></article>)}</div></section>
  <section className="technology-strip shell"><div><h2>Local-first. Powerful. Yours.</h2><p>Akane runs as a small, focused project with a shared local runtime.</p></div><div className="tech-pills">{["Gemma 4 E4B", "llama.cpp", "Python", "Atomic JSON", "Discord"].map(item => <span key={item}>{item}</span>)}</div></section>
  <section className="architecture-teaser shell"><div><Eyebrow>One companion, several ways to connect</Eyebrow><h2>A shared backend, with a familiar presence.</h2><p>Web Demo, Popup, and Discord each send requests through the same Akane runtime.</p><Link className="text-link" to="/technology">Explore the architecture →</Link></div><div className="architecture-lines" aria-label="Web Demo, Popup, and Discord connect to the shared Akane backend, then Gemma through llama.cpp"><span>Web Demo</span><span>Popup</span><span>Discord</span><b>Shared Akane Backend</b><strong>Gemma 4 via llama.cpp</strong></div></section>
  <section className="home-cta shell"><Logo /><div><h2>Open source. Built with care.</h2><p>Akane is a local-first companion project made for thoughtful experimentation.</p></div><GithubLink>View on GitHub</GithubLink><Link className="button primary" to="/demo">Try Demo →</Link></section>
</main>; }

type Message = { id: string; role: "akane" | "you"; text: string; time: string; preview?: boolean };
type ConnectionState = "connecting" | "live" | "showcase";

const greeting: Message = {
  id: "welcome",
  role: "akane",
  text: "I'm here. Public profiles stay separate from the owner's personal Akane state.",
  time: "Now",
};
const previewReplies = [
  "This is a prerecorded preview. Live Akane can remember the active public profile and stream a response when the Raspberry Pi is available.",
  "In live guest mode, memory and relationship continuity are real for this tab, but the temporary profile expires.",
  "Akane's real conversation path uses the same prompt, memory, emotion, relationship, and shared model coordinator as her other interfaces.",
];
const retryDelays = [5_000, 15_000, 30_000, 60_000];

function currentTime() {
  return new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function sessionGreeting(): Message {
  return {
    ...greeting,
    id: crypto.randomUUID(),
    text: "Guest mode is live. Our memory and relationship continuity are real, but this temporary profile will expire.",
  };
}

function DemoPage() {
  const initialConnection: ConnectionState =
    projectConfig.demoMode === "showcase" || !projectConfig.apiUrl ? "showcase" : "connecting";
  const [connection, setConnection] = useState<ConnectionState>(initialConnection);
  const [health, setHealth] = useState<PublicHealth | null>(null);
  const [messages, setMessages] = useState<Message[]>([greeting]);
  const [input, setInput] = useState("");
  const [activeSession, setActiveSession] = useState<PublicSession | null>(null);
  const activeSessionRef = useRef<PublicSession | null>(null);
  const [backendPresentation, setBackendPresentation] = useState<AkanePresentationState>();
  const [generating, setGenerating] = useState(false);
  const [actionPending, setActionPending] = useState(false);
  const [error, setError] = useState("");
  const [retryExhausted, setRetryExhausted] = useState(false);
  const [reconnectKey, setReconnectKey] = useState(0);
  const [tts, setTts] = useState(false);
  const [volume, setVolume] = useState(.8);
  const aborter = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  function activateSession(session: PublicSession | null) {
    activeSessionRef.current = session;
    setActiveSession(session);
  }

  useEffect(() => { endRef.current?.scrollIntoView({ block: "nearest" }); }, [messages]);

  useEffect(() => {
    if (projectConfig.demoMode === "showcase" || !projectConfig.apiUrl) {
      setConnection("showcase");
      setHealth(null);
      return;
    }

    const controller = new AbortController();
    let stopped = false;
    let timer: number | undefined;
    setConnection("connecting");
    setRetryExhausted(false);

    async function probe(attempt: number) {
      try {
        const result = await akaneClient.health(controller.signal);
        if (stopped) return;
        if (result.status === "offline" || !result.streaming) {
          throw new PublicApiError("model_unavailable", "Live Akane is still starting up.");
        }

        const current = activeSessionRef.current;
        if (current) {
          try {
            const resumed = await akaneClient.revalidateSession(current.sessionToken);
            activateSession(resumed);
          } catch (cause) {
            if (cause instanceof PublicApiError && ["session_expired", "unauthorized"].includes(cause.code)) {
              clearGuestToken();
              activateSession(null);
              setError("That temporary guest session expired. Start a new guest session when Akane reconnects.");
            } else {
              throw cause;
            }
          }
        }

        setHealth(result);
        setConnection("live");
        setRetryExhausted(false);
      } catch (cause) {
        if (stopped || (cause as Error).name === "AbortError") return;
        setHealth(null);
        setConnection("showcase");
        if (attempt < retryDelays.length) {
          timer = window.setTimeout(() => { void probe(attempt + 1); }, retryDelays[attempt]);
        } else {
          setRetryExhausted(true);
        }
      }
    }

    void probe(0);
    return () => {
      stopped = true;
      controller.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [reconnectKey]);

  function reconnect() {
    if (projectConfig.demoMode !== "live" || !projectConfig.apiUrl) return;
    setError("");
    setReconnectKey(current => current + 1);
  }

  function availabilityFailed(cause: unknown) {
    return cause instanceof PublicApiError
      && !["busy", "queue_full", "rate_limited", "generation_timeout"].includes(cause.code)
      && (["offline", "model_unavailable", "invalid_response"].includes(cause.code) || cause.status >= 500);
  }

  function openPreview() {
    setConnection("showcase");
    setMessages([greeting]);
    setError("");
  }

  async function startGuest() {
    if (connection !== "live" || generating || actionPending) return;
    setActionPending(true);
    setError("");
    try {
      const storedToken = getGuestToken();
      const session = storedToken
        ? await akaneClient.revalidateSession(storedToken)
        : await akaneClient.createSession();
      storeGuestToken(session.sessionToken);
      activateSession(session);
      setMessages([sessionGreeting()]);
    } catch (cause) {
      if (cause instanceof PublicApiError && ["session_expired", "unauthorized"].includes(cause.code)) {
        clearGuestToken();
        try {
          const session = await akaneClient.createSession();
          storeGuestToken(session.sessionToken);
          activateSession(session);
          setMessages([sessionGreeting()]);
          return;
        } catch (renewalError) {
          cause = renewalError;
        }
      }
      setError((cause as Error).message);
      if (availabilityFailed(cause)) reconnect();
    } finally {
      setActionPending(false);
    }
  }

  async function resetConversation() {
    if (generating) return;
    aborter.current?.abort();
    stopSpeech();
    setError("");
    if (!activeSession) {
      setMessages([greeting]);
      return;
    }
    if (connection !== "live") {
      setError("Reconnect to live Akane before resetting this conversation.");
      return;
    }
    setActionPending(true);
    try {
      await akaneClient.resetConversation(activeSession.sessionToken);
      setMessages([sessionGreeting()]);
    } catch (cause) {
      setError((cause as Error).message);
      if (availabilityFailed(cause)) reconnect();
    } finally {
      setActionPending(false);
    }
  }

  async function endGuestSession() {
    if (!activeSession || actionPending) return;
    setActionPending(true);
    setError("");
    try {
      if (connection === "live") await akaneClient.deleteSession(activeSession.sessionToken);
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      clearGuestToken();
      activateSession(null);
      setMessages([greeting]);
      setActionPending(false);
    }
  }

  function sendPreview(message: string) {
    const reply = previewReplies[Math.abs([...message].reduce((sum, character) => sum + character.charCodeAt(0), 0)) % previewReplies.length];
    const now = currentTime();
    setMessages(current => [
      ...current,
      { id: crypto.randomUUID(), role: "you", text: message, time: now },
      { id: crypto.randomUUID(), role: "akane", text: reply, time: now, preview: true },
    ]);
    if (tts) speak(reply, volume);
  }

  async function renewExpiredGuest() {
    clearGuestToken();
    const session = await akaneClient.createSession();
    storeGuestToken(session.sessionToken);
    activateSession(session);
    setError("The temporary guest session expired, so a fresh guest session is ready. Please send that message again.");
  }

  async function send(text = input) {
    const message = text.trim();
    if (!message || generating) return;
    if (message.length > 750) {
      setError("Keep messages at or under 750 characters.");
      return;
    }
    if (connection === "connecting") {
      setError("Akane is still connecting. You can send once live mode or Preview Mode is ready.");
      return;
    }
    if (connection === "showcase") {
      setInput("");
      setError("");
      sendPreview(message);
      return;
    }
    if (!activeSession) {
      setError("Start a temporary guest session before sending a live message.");
      return;
    }

    setInput("");
    setError("");
    setGenerating(true);
    const now = currentTime();
    const replyId = crypto.randomUUID();
    setMessages(current => [
      ...current,
      { id: crypto.randomUUID(), role: "you", text: message, time: now },
      { id: replyId, role: "akane", text: "", time: now },
    ]);
    const controller = new AbortController();
    aborter.current = controller;
    let completed = "";
    try {
      await akaneClient.streamChat(activeSession.sessionToken, message, {
        onDelta: delta => {
          completed += delta;
          setMessages(current => current.map(item => item.id === replyId ? { ...item, text: completed } : item));
        },
        onDone: () => { if (tts) speak(completed, volume); },
        onPresentation: setBackendPresentation,
      }, controller.signal);
    } catch (cause) {
      setMessages(current => current.filter(item => item.id !== replyId));
      if ((cause as Error).name !== "AbortError") {
        if (cause instanceof PublicApiError && cause.code === "session_expired") {
          try { await renewExpiredGuest(); }
          catch (renewalError) {
            clearGuestToken();
            activateSession(null);
            setError((renewalError as Error).message);
          }
        } else {
          setError((cause as Error).message);
          if (availabilityFailed(cause)) reconnect();
        }
      }
    } finally {
      aborter.current = null;
      setGenerating(false);
      setBackendPresentation(undefined);
    }
  }

  function keyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void send();
    }
  }

  const previewMode = connection === "showcase";
  const needsSession = connection === "live" && !activeSession;
  const inputDisabled = generating || actionPending || connection === "connecting" || needsSession;
  const sessionLabel = activeSession ? "Temporary guest profile" : "No live profile selected";
  const connectionLabel = connection === "connecting"
    ? "Connecting"
    : connection === "showcase" ? "Preview Mode" : health?.status === "busy" ? "Live · busy" : "Live";

  return <main className="demo shell">
    <div className="demo-intro"><Eyebrow>Browser interface</Eyebrow><h1>Meet Akane</h1><p>Use an isolated temporary guest profile, or explore the clearly labeled offline preview while the Raspberry Pi is unavailable.</p>
      <div className={`mode-banner ${connection}`} role="status">
        <strong>{connectionLabel}</strong>
        <span>{connection === "connecting" ? "Checking the Raspberry Pi connection…" : previewMode ? "Replies are prerecorded, are not sent to Akane, and are not saved." : "Responses stream from the real Akane runtime through your isolated public profile."}</span>
      </div>
    </div>
    <div className="demo-grid">
      <section className="conversation panel" aria-label="Conversation history"><div className="panel-title"><h2>Conversation</h2><span className={`status ${generating ? "active" : ""}`}>{generating ? "Streaming" : connectionLabel}</span></div><div className="messages" aria-live="polite">{messages.map(message => <article className={`message ${message.role}`} key={message.id}><div className="avatar">{message.role === "akane" ? "A" : "Y"}</div><div><div className="message-meta"><strong>{message.role === "akane" ? "Akane" : "You"}</strong>{message.preview && <span className="preview-label">Prerecorded preview</span>}<time>{message.time}</time></div><p>{message.text || <span className="streaming-cursor">Thinking</span>}</p></div></article>)}<div ref={endRef} /></div><button className="quiet-button" disabled={actionPending || generating} onClick={() => { if (previewMode) openPreview(); else void resetConversation(); }}>{previewMode || !activeSession ? "Clear preview" : "Reset conversation"}</button></section>
      <AkaneStage
        imageSrc={asset}
        bubbleText={messages.at(-1)?.role === "akane" && messages.at(-1)?.text ? messages.at(-1)!.text : "I'm all ears. What’s on your mind?"}
        connection={connection}
        connectionLabel={previewMode ? "Preview Mode" : connection === "connecting" ? "Connecting" : generating ? "Streaming live" : "Live Akane"}
        generating={generating}
        hasResponseText={Boolean(generating && messages.at(-1)?.role === "akane" && messages.at(-1)?.text)}
        modelName={projectConfig.modelName}
        sessionLabel={previewMode ? "Prerecorded showcase" : sessionLabel}
        backendPresentation={backendPresentation}
      />
      <aside className="controls panel"><h2>Demo controls</h2>
        <fieldset><legend>Voice</legend><label className="toggle"><span>Browser TTS <small>{canUseSpeech() ? "Optional" : "Unavailable"}</small></span><input type="checkbox" checked={tts} disabled={!canUseSpeech()} onChange={event => setTts(event.target.checked)} /></label><label>Volume<input type="range" min="0" max="1" step=".1" value={volume} disabled={!tts} onChange={event => setVolume(Number(event.target.value))} /></label></fieldset>
        <fieldset><legend>Session</legend>
          {connection === "connecting" && <p>Checking live availability…</p>}
          {previewMode && <><p className="preview-note"><strong>Preview Mode</strong> is prerecorded and never persisted.</p><button className="quiet-button" onClick={openPreview}>Clear preview</button></>}
          {connection === "live" && !activeSession && <div className="profile-choice"><p>Guest memory and relationship continuity are real but temporary and isolated from every other profile.</p><button className="quiet-button" disabled={actionPending || health?.guestEnabled !== true} onClick={() => void startGuest()}>Start guest session</button><button className="quiet-button" disabled={actionPending} onClick={openPreview}>Offline Preview</button></div>}
          {activeSession && <div className="session-actions"><p className="session-id">Temporary guest continuity active</p><button className="quiet-button" disabled={actionPending || generating} onClick={() => void resetConversation()}>Reset conversation</button><button className="danger-button" disabled={actionPending || generating} onClick={() => void endGuestSession()}>End guest session</button></div>}
        </fieldset>
        <fieldset><legend>Connection</legend><p><span className={`dot ${previewMode ? "bad" : ""}`} />{connectionLabel}</p>{previewMode && projectConfig.demoMode === "live" && projectConfig.apiUrl && <button className="quiet-button" onClick={reconnect}>Reconnect now</button>}{retryExhausted && <p className="retry-note">Automatic retries finished. Manual reconnect remains available.</p>}</fieldset>
      </aside>
    </div>
    <section className="suggestions panel"><h2>{previewMode ? "Try the prerecorded preview…" : "Try saying something…"}</h2><div>{suggestions.map(prompt => <button key={prompt} onClick={() => void send(prompt)} disabled={inputDisabled}>{prompt}</button>)}</div></section>
    <form className="composer panel" onSubmit={(event: FormEvent) => { event.preventDefault(); void send(); }}><Mark /><label className="sr-only" htmlFor="message">Message Akane</label><textarea id="message" value={input} onChange={event => setInput(event.target.value)} onKeyDown={keyDown} maxLength={750} disabled={inputDisabled} placeholder={needsSession ? "Start a guest session first…" : connection === "connecting" ? "Connecting to Akane…" : previewMode ? "Try the prerecorded preview…" : "Message Akane…"} rows={2} /><div><small>{input.length}/750 · Enter to send, Shift + Enter for a new line</small>{error && <p className="form-error" role="alert">{error}</p>}</div>{generating ? <button className="button secondary" type="button" onClick={() => { aborter.current?.abort(); stopSpeech(); }}>Stop</button> : <button className="button primary" disabled={inputDisabled || !input.trim()}>Send →</button>}</form>
    <section className="runtime panel" aria-label="Runtime status"><span>◈ {connectionLabel}</span><span>▣ Model: {projectConfig.modelName}</span><span>◌ {tts ? "Browser TTS on" : "Voice off"}</span><span>◉ {previewMode ? "Nothing saved" : sessionLabel}</span></section>
  </main>;
}

function TechnologyPage() { const architecture: Array<[string, string[]]> = [["Interfaces", ["Web Demo", "Desktop Popup", "Discord"]], ["Shared Akane Backend", ["Prompt Compiler", "Conversation Memory", "Canonical Profile State", "Emotion System", "Offscreen Life", "Autonomous Initiative", "Time Context", "Inference Coordinator"]], ["Model Runtime", ["Gemma 4 E4B", "llama.cpp", "Quantized GGUF"]], ["Persistence", ["Atomic state writes", "Conversation history", "Validated profile state"]]]; return <main>
  <section className="technology-hero shell"><div><Eyebrow>Local first · Privacy minded · Always yours</Eyebrow><h1>How <em>Akane</em> works</h1><p className="lead">Akane combines a local language model with persistent state, memory, autonomous life, emotional continuity, and multiple user interfaces.</p><p className="detail">Akane’s primary backend runs locally on a Raspberry Pi 5. This static frontend connects to that backend when an endpoint is configured.</p><div className="actions"><Link className="button primary" to="/demo">Try Demo →</Link><GithubLink>View on GitHub</GithubLink></div></div><div className="tech-hero-art"><div className="speech-bubble">A local companion<br /><strong>with a shared runtime.</strong></div><Character /></div></section>
  <section className="architecture shell panel"><div className="section-heading left"><Eyebrow>System architecture</Eyebrow><h2>One coordinated companion</h2></div><div className="architecture-grid">{architecture.map(([title, items], index) => <article key={title} className="architecture-column"><h3>{title}</h3><div>{items.map(item => <span key={item}>{item}</span>)}</div>{index < architecture.length - 1 && <b className="flow-arrow" aria-hidden="true">↓</b>}</article>)}</div><p className="flow-caption">Interfaces → shared request handling → prompt and context compilation → serialized inference → response streaming → state persistence</p></section>
  <section className="section shell"><div className="section-heading"><Eyebrow>Core capabilities</Eyebrow><h2>Built for continuity, not just replies</h2></div><div className="feature-grid">{[["◈", "Local Model Inference", "A single llama.cpp runtime serializes visible and background work on constrained hardware."], ["✦", "Persistent Memory and State", "Atomic JSON writes preserve validated conversations, memory, emotion, and presence."], ["♡", "Emotional Continuity", "Recent context informs a grounded emotional state without a second model pass."], ["◔", "Autonomous Life and Initiative", "A process-owned worker maintains activities and avoids duplicate autonomous messages."], ["↗", "Streaming Multi-Interface UI", "The popup, Discord adapter, and web demo all use the shared request path."]].map(([i, t, p]) => <article className="feature-card" key={t}><i>{i}</i><h3>{t}</h3><p>{p}</p><span className="implemented">Implemented</span></article>)}</div></section>
  <section className="flow-section shell panel"><Eyebrow>System flow</Eyebrow><div className="system-flow">{["User interface", "Shared backend", "Relevant context", "Gemma inference", "Streamed response", "Validated state update"].map((item, index) => <div key={item}><span>{item}</span>{index < 5 && <b>→</b>}</div>)}</div></section>
  <section className="stack-section shell"><div className="section-heading left"><Eyebrow>Technology stack</Eyebrow><h2>Small, confirmed building blocks</h2></div><div className="stack-grid">{stack.map(([category, name, detail]) => <article key={category}><span>{category}</span><h3>{name}</h3><p>{detail}</p></article>)}</div></section>
  <section className="engineering shell panel"><Eyebrow>Engineering focus</Eyebrow><h2>The constraints shape the work.</h2><div>{["Serializing conversation and autonomous inference on one Pi", "Preserving state safely across restarts", "Keeping companion context distinct from the user", "Coordinating popup, Discord, and web interfaces"].map(item => <p key={item}>✦ {item}</p>)}</div></section>
  <section className="home-cta shell"><Logo /><div><h2>Build with Akane.</h2><p>Explore the browser demo or see the local-first project on GitHub.</p></div><GithubLink>View on GitHub</GithubLink><Link className="button primary" to="/demo">Try Demo →</Link></section>
</main>; }

function App() { const location = useLocation(); useEffect(() => { window.scrollTo(0, 0); }, [location.pathname]); return <><Navbar /><Routes><Route path="/" element={<HomePage />} /><Route path="/demo" element={<DemoPage />} /><Route path="/technology" element={<TechnologyPage />} /><Route path="*" element={<HomePage />} /></Routes><Footer /></>; }
export default App;
