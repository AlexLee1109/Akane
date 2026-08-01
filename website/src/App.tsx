import { FormEvent, KeyboardEvent, ReactNode, useEffect, useRef, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { AkaneStage } from "./components/AkaneStage";
import { projectConfig } from "./config/project";
import { akaneClient, PublicApiError, type PublicHealth, type PublicSession } from "./lib/akaneClient";
import { clearGuestToken, getGuestToken, storeGuestToken } from "./lib/session";
import type { AkanePresentationState } from "./presentation";

const asset = `${projectConfig.basePath}assets/akane-hero.png`;
const logo = `${projectConfig.basePath}assets/akane-logo.png`;
const homepageImage = `${projectConfig.basePath}assets/homepage-image.png`;
const github = projectConfig.githubUrl;
const features = [
  ["✦", "Memory", "Remembers meaningful conversations, preferences, shared context, and important details over time."],
  ["♡", "Emotional continuity", "Maintains grounded emotion, mood, and relationship context without forcing it into every reply."],
  ["◔", "Ambient presence", "Keeps a quiet sense of focus between conversations without inventing a complete physical world."],
  ["⌁", "Natural conversation", "Responds with direct judgment, personality, continuity, and context-aware reasoning."],
  ["◎", "Companion interfaces", "Shares one backend across the desktop popup, Discord, and the web."],
];
const continuitySteps = [
  ["01", "Remember", "Meaningful facts, preferences, decisions, and shared experiences."],
  ["02", "Understand", "Relevant context influences her judgment without overwhelming the reply."],
  ["03", "Continue", "Future conversations build from established memory and relationship state."],
];
const availableNow = [
  "Local Gemma inference",
  "Discord conversation",
  "Desktop popup",
  "Streaming text generation",
  "Persistent owner memory",
  "Emotion and relationship continuity",
  "Ambient presence",
  "Read-only VS Code context",
];
const plannedWork = [
  "Speech output",
  "Live2D presentation layer",
  "Expression synchronization",
];
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
  const location = useLocation();
  const links = [["/", "Home"], ["/demo", "Demo"], ["/technology", "Technology"]] as const;
  return <header className={`site-header ${location.pathname === "/" ? "home-header" : ""}`}><nav className="nav shell" aria-label="Primary navigation">
    <Link className="brand" to="/" onClick={() => setOpen(false)}><Logo /><strong>Akane</strong><span>AI COMPANION</span></Link>
    <button className="menu-button" aria-expanded={open} aria-controls="nav-links" onClick={() => setOpen(!open)}>{open ? "Close" : "Menu"}</button>
    <div id="nav-links" className={`nav-links ${open ? "open" : ""}`}>{links.map(([to, label]) => <NavLink key={to} to={to} end={to === "/"} onClick={() => setOpen(false)}>{label}</NavLink>)}<GithubLink className="nav-github">GitHub</GithubLink></div>
  </nav></header>;
}

function Footer() { return <footer className="footer shell"><div className="footer-brand"><Logo /><div><strong>Akane</strong><p>A local-first AI companion built around memory, continuity, and personal presence.</p><small>Open source under the MIT License.</small></div></div><div className="footer-links"><Link to="/">Home</Link><Link to="/demo">Demo</Link><Link to="/technology">Technology</Link><GithubLink className="plain-link">GitHub</GithubLink></div><small>© {new Date().getFullYear()} Akane</small></footer>; }

function Character({ compact = false }: { compact?: boolean }) { return <img className={`akane-image ${compact ? "compact" : ""}`} src={asset} alt="Akane, a blue-haired anime-style companion in a white jacket" />; }
function Eyebrow({ children }: { children: ReactNode }) { return <p className="eyebrow"><Mark /> {children}</p>; }

function StatusBadge({ kind = "available" }: { kind?: "available" | "planned" }) {
  return <span className={`status-badge ${kind}`}><span aria-hidden="true">{kind === "available" ? "✓" : "○"}</span>{kind === "available" ? "Available" : "Planned"}</span>;
}

function HomePage() { return <main className="home-page">
  <section className="home-hero shell" aria-labelledby="home-title">
    <picture className="home-hero-media">
      <source
        type="image/jpeg"
        srcSet={`${projectConfig.basePath}assets/homepage-image-720.jpg 720w, ${projectConfig.basePath}assets/homepage-image-1100.jpg 1100w, ${projectConfig.basePath}assets/homepage-image-1448.jpg 1448w`}
        sizes="100vw"
      />
      <img src={homepageImage} width="1448" height="1086" fetchPriority="high" decoding="async" alt="Akane, a blue-haired AI companion, standing in a bright room overlooking a city." />
    </picture>
    <div className="hero-atmosphere" aria-hidden="true"><span /><span /><span /></div>
    <div className="home-hero-copy">
      <p className="hero-badge"><span aria-hidden="true">♢</span> Local-first <b>•</b> Private <b>•</b> Always yours</p>
      <h1 id="home-title">Your local<br />AI companion.</h1>
      <p className="hero-accent">Always by your side.</p>
      <p className="home-lead">Akane remembers what matters, develops through conversation, and stays present across your desktop, Discord, and the web.</p>
      <div className="actions hero-actions"><Link className="button primary" to="/demo"><span aria-hidden="true">✦</span> Try the Demo <span aria-hidden="true">→</span></Link><GithubLink>View on GitHub</GithubLink></div>
      <ul className="trust-list" aria-label="Project facts"><li><span aria-hidden="true">▣</span> Runs on Raspberry Pi</li><li><span aria-hidden="true">♢</span> Private owner profile</li><li><span aria-hidden="true">〈/〉</span> Open source</li></ul>
    </div>
  </section>

  <section className="home-section capability-section shell" aria-labelledby="capability-title">
    <div className="section-heading"><Eyebrow>Made to understand you</Eyebrow><h2 id="capability-title">Features that make Akane <em>feel real</em></h2></div>
    <div className="feature-grid">{features.map(([icon, title, text]) => <article className="feature-card" key={title}><i aria-hidden="true">{icon}</i><h3>{title}</h3><p>{text}</p><StatusBadge /></article>)}</div>
  </section>

  <section className="home-section continuity-section shell" aria-labelledby="continuity-title">
    <div className="continuity-intro"><Eyebrow>Built through continuity</Eyebrow><h2 id="continuity-title">A companion that develops with every meaningful interaction</h2><p>Akane does more than retain isolated facts. Conversations can shape what she remembers, how she understands you, and how the relationship develops over time.</p></div>
    <ol className="continuity-flow">{continuitySteps.map(([number, title, text]) => <li key={title}><span className="step-number">{number}</span><div><h3>{title}</h3><p>{text}</p></div></li>)}</ol>
  </section>

  <section className="home-section interface-section shell" aria-labelledby="interface-title">
    <div className="section-heading"><Eyebrow>One shared companion</Eyebrow><h2 id="interface-title">Meet Akane wherever the conversation happens</h2><p>Discord, the desktop popup, and the website connect to the same conversation architecture while keeping personal and guest profiles isolated.</p></div>
    <div className="interface-diagram" role="group" aria-label="Desktop popup and Discord use the owner profile, while the website uses a temporary guest profile. All connect to the shared Akane backend and Gemma on the Raspberry Pi.">
      <div className="interface-sources">
        <article><div><span className="interface-icon" aria-hidden="true">▣</span><h3>Desktop popup</h3><StatusBadge /></div><p>Akane’s primary companion experience, designed for an always-present desktop form.</p><small>Owner profile</small></article>
        <article><div><span className="interface-icon" aria-hidden="true">⌁</span><h3>Discord</h3><StatusBadge /></div><p>Remote and mobile conversation through the same owner profile and shared backend.</p><small>Owner profile</small></article>
        <article><div><span className="interface-icon" aria-hidden="true">◎</span><h3>Website</h3><StatusBadge /></div><p>A browser-based guest experience connected to the real model running on the Raspberry Pi.</p><small>Temporary guest profile</small></article>
      </div>
      <div className="diagram-arrow" aria-hidden="true">→</div>
      <article className="backend-node"><span aria-hidden="true">✦</span><h3>Shared Akane backend</h3><ul><li>Conversation</li><li>Memory</li><li>Emotion</li><li>Relationship</li><li>Inference</li></ul></article>
      <div className="diagram-arrow" aria-hidden="true">→</div>
      <article className="runtime-node"><span aria-hidden="true">◈</span><h3>Gemma</h3><p>Runs on the owner’s Raspberry Pi</p></article>
    </div>
    <p className="profile-note"><strong>Private continuity:</strong> Desktop and Discord share the owner profile. Website guests receive a separate temporary profile.</p>
  </section>

  <section className="home-section development-section shell" aria-labelledby="development-title">
    <div className="section-heading"><Eyebrow>Current development</Eyebrow><h2 id="development-title">What Akane can do today—and what comes next</h2></div>
    <div className="development-grid">
      <article><div className="status-heading"><span aria-hidden="true">✓</span><div><h3>Available now</h3><p>Implemented in the current repository.</p></div></div><ul>{availableNow.map(item => <li key={item}><span>{item}</span><StatusBadge /></li>)}</ul></article>
      <article><div className="status-heading planned"><span aria-hidden="true">○</span><div><h3>Being developed</h3><p>Planned directions, without promised dates.</p></div></div><ul>{plannedWork.map(item => <li key={item}><span>{item}</span><StatusBadge kind="planned" /></li>)}</ul></article>
    </div>
  </section>

  <section className="home-cta shell" aria-labelledby="cta-title"><Mark /><div><h2 id="cta-title">Ready to meet Akane?</h2><p>Experience the real companion pipeline through the live demo, or explore the project on GitHub.</p></div><div className="actions"><Link className="button cta-primary" to="/demo">Try the Demo <span aria-hidden="true">→</span></Link><GithubLink className="button cta-secondary">View on GitHub</GithubLink></div></section>
</main>; }

type Message = { id: string; role: "akane" | "you"; text: string; time: string; preview?: boolean };
type ConnectionState = "connecting" | "live" | "showcase";

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
    id: crypto.randomUUID(),
    role: "akane",
    text: "Guest mode is live. Our memory and relationship continuity are real, but this temporary profile will expire.",
    time: "Now",
  };
}

function DemoPage() {
  const initialConnection: ConnectionState =
    projectConfig.demoMode === "showcase" || !projectConfig.apiUrl ? "showcase" : "connecting";
  const [connection, setConnection] = useState<ConnectionState>(initialConnection);
  const [health, setHealth] = useState<PublicHealth | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [activeSession, setActiveSession] = useState<PublicSession | null>(null);
  const activeSessionRef = useRef<PublicSession | null>(null);
  const [backendPresentation, setBackendPresentation] = useState<AkanePresentationState>();
  const [generating, setGenerating] = useState(false);
  const [actionPending, setActionPending] = useState(false);
  const [error, setError] = useState("");
  const [retryExhausted, setRetryExhausted] = useState(false);
  const [reconnectKey, setReconnectKey] = useState(0);
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
    setMessages([]);
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
    setError("");
    if (!activeSession) {
      setMessages([]);
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
      setMessages([]);
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
        <fieldset><legend>Session</legend>
          {connection === "connecting" && <p>Checking live availability…</p>}
          {previewMode && <><p className="preview-note"><strong>Preview Mode</strong> is prerecorded and never persisted.</p><button className="quiet-button" onClick={openPreview}>Clear preview</button></>}
          {connection === "live" && !activeSession && <div className="profile-choice"><p>Guest memory and relationship continuity are real but temporary and isolated from every other profile.</p><button className="quiet-button" disabled={actionPending || health?.guestEnabled !== true} onClick={() => void startGuest()}>Start guest session</button><button className="quiet-button" disabled={actionPending} onClick={openPreview}>Offline Preview</button></div>}
          {activeSession && <div className="session-actions"><p className="session-id">Temporary guest continuity active</p><button className="quiet-button" disabled={actionPending || generating} onClick={() => void resetConversation()}>Reset conversation</button><button className="danger-button" disabled={actionPending || generating} onClick={() => void endGuestSession()}>End guest session</button></div>}
        </fieldset>
        <fieldset><legend>Connection</legend><p><span className={`dot ${previewMode ? "bad" : ""}`} />{connectionLabel}</p>{previewMode && projectConfig.demoMode === "live" && projectConfig.apiUrl && <button className="quiet-button" onClick={reconnect}>Reconnect now</button>}{retryExhausted && <p className="retry-note">Automatic retries finished. Manual reconnect remains available.</p>}</fieldset>
      </aside>
    </div>
    <form className="composer panel" onSubmit={(event: FormEvent) => { event.preventDefault(); void send(); }}><Mark /><label className="sr-only" htmlFor="message">Message Akane</label><textarea id="message" value={input} onChange={event => setInput(event.target.value)} onKeyDown={keyDown} maxLength={750} disabled={inputDisabled} placeholder={needsSession ? "Start a guest session first…" : connection === "connecting" ? "Connecting to Akane…" : previewMode ? "Try the prerecorded preview…" : "Message Akane…"} rows={2} /><div><small>{input.length}/750 · Enter to send, Shift + Enter for a new line</small>{error && <p className="form-error" role="alert">{error}</p>}</div>{generating ? <button className="button secondary" type="button" onClick={() => { aborter.current?.abort(); }}>Stop</button> : <button className="button primary" disabled={inputDisabled || !input.trim()}>Send →</button>}</form>
    <section className="runtime panel" aria-label="Runtime status"><span>◈ {connectionLabel}</span><span>▣ Model: {projectConfig.modelName}</span><span>◉ {previewMode ? "Nothing saved" : sessionLabel}</span></section>
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
