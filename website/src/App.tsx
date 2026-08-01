import { FormEvent, KeyboardEvent, ReactNode, useEffect, useRef, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { AkaneStage } from "./components/AkaneStage";
import { projectConfig } from "./config/project";
import { akaneClient, PublicApiError, type PublicHealth, type PublicSession } from "./lib/akaneClient";
import { clearGuestToken, getGuestToken, storeGuestToken } from "./lib/session";
import { TechnologyPage } from "./pages/TechnologyPage";
import type { AkanePresentationState } from "./presentation";

const asset = `${projectConfig.basePath}assets/akane-hero.png`;
const logo = `${projectConfig.basePath}assets/akane-logo.png`;
const homepageImage = `${projectConfig.basePath}assets/homepage-image.png`;
const github = projectConfig.githubUrl;
const companionPillars = [
  ["memory", "Remembers you", "Meaningful facts, decisions, preferences, and shared experiences remain part of future conversations instead of disappearing after each session."],
  ["growth", "Develops with you", "Akane maintains her own opinions, emotional continuity, and an evolving relationship rather than resetting into a generic assistant."],
  ["presence", "Stays present", "Talk through the desktop popup, Discord, or the web while sharing the same underlying companion intelligence."],
] as const;
const availableNow = [
  "Local conversation",
  "Persistent memory",
  "Discord",
  "Desktop popup",
  "Web guest demo",
];
const plannedWork = [
  "Voice",
  "Live2D",
  "Expression synchronization",
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

type HomeIconName = "memory" | "growth" | "presence" | "desktop" | "discord" | "website" | "pi";

function HomeIcon({ name }: { name: HomeIconName }) {
  const paths: Record<HomeIconName, ReactNode> = {
    memory: <><path d="M5 5.5h14v10H9l-4 3v-13Z" /><path d="M8.5 9h7M8.5 12h4.5" /></>,
    growth: <><path d="M12 20V10" /><path d="M12 12c-4 0-6-2.5-6-6 4 0 6 2.5 6 6ZM12 15c4 0 6-2.5 6-6-4 0-6 2.5-6 6Z" /></>,
    presence: <><circle cx="12" cy="12" r="2.5" /><path d="M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4M4.7 4.7a10.3 10.3 0 0 0 0 14.6M19.3 4.7a10.3 10.3 0 0 1 0 14.6" /></>,
    desktop: <><rect x="3.5" y="4.5" width="17" height="12" rx="1.5" /><path d="M8.5 20h7M12 16.5V20" /></>,
    discord: <><path d="M5 6.5c4.5-2 9.5-2 14 0l1.2 9.2a14 14 0 0 1-4.2 2.1l-1-1.5M19 6.5l-1.2 9.2A14 14 0 0 1 13.5 18" /><circle cx="9" cy="12" r="1" /><circle cx="15" cy="12" r="1" /></>,
    website: <><circle cx="12" cy="12" r="8.5" /><path d="M3.7 12h16.6M12 3.5c2.2 2.4 3.3 5.2 3.3 8.5S14.2 18.1 12 20.5C9.8 18.1 8.7 15.3 8.7 12S9.8 5.9 12 3.5Z" /></>,
    pi: <><rect x="5" y="5" width="14" height="14" rx="2" /><path d="M9 2.5v2.3M15 2.5v2.3M9 19.2v2.3M15 19.2v2.3M2.5 9h2.3M2.5 15h2.3M19.2 9h2.3M19.2 15h2.3" /><circle cx="12" cy="12" r="3" /></>,
  };
  return <svg className="home-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">{paths[name]}</svg>;
}

function HomeHero() { return <section className="home-hero shell" aria-labelledby="home-title">
  <picture className="home-hero-media">
    <source type="image/jpeg" srcSet={`${projectConfig.basePath}assets/homepage-image-720.jpg 720w, ${projectConfig.basePath}assets/homepage-image-1100.jpg 1100w, ${projectConfig.basePath}assets/homepage-image-1448.jpg 1448w`} sizes="100vw" />
    <img src={homepageImage} width="1448" height="1086" fetchPriority="high" decoding="async" alt="Akane, a blue-haired AI companion, standing in a bright room overlooking a city." />
  </picture>
  <div className="home-hero-copy">
    <p className="hero-badge"><span aria-hidden="true">♢</span> Local-first <b>•</b> Private <b>•</b> Always yours</p>
    <h1 id="home-title">A local AI companion<br />that remembers you.</h1>
    <p className="hero-accent">Always by your side.</p>
    <p className="home-lead">Akane develops through conversation, remembers meaningful experiences, and stays present across your desktop, Discord, and the web—all powered by your Raspberry Pi.</p>
    <div className="actions hero-actions"><Link className="button primary" to="/demo"><span aria-hidden="true">✦</span> Try the Demo <span aria-hidden="true">→</span></Link><GithubLink>View on GitHub</GithubLink></div>
    <ul className="trust-list" aria-label="Project facts"><li><span aria-hidden="true">▣</span> Runs on Raspberry Pi</li><li><span aria-hidden="true">♢</span> Your memories stay private</li><li><span aria-hidden="true">〈/〉</span> Open source</li></ul>
  </div>
</section>; }

function ProductDemo() { return <section className="home-section product-demo shell" aria-labelledby="product-demo-title">
  <div className="home-editorial-heading"><Eyebrow>See Akane in action</Eyebrow><h2 id="product-demo-title">More than a chat window</h2><p>Akane combines streamed conversation, persistent memory, emotional continuity, and an always-present desktop form.</p></div>
  <figure className="product-demo-figure" data-media-status="awaiting-real-popup-recording">
    <div className="product-demo-poster">
      <div className="product-demo-copy"><span>Desktop companion preview</span><h3>A companion designed to stay close.</h3><p>This space is ready for a real popup recording. No simulated conversation is shown.</p><small>Real demonstration recording coming soon</small></div>
      <img src={asset} loading="lazy" decoding="async" alt="Akane’s desktop character artwork" />
    </div>
    <figcaption className="product-demo-facts"><span><b>01</b> Persistent continuity</span><span><b>02</b> Real local inference</span><span><b>03</b> One Akane across interfaces</span></figcaption>
  </figure>
</section>; }

function CompanionPillars() { return <section className="home-section pillars-section shell" aria-labelledby="pillars-title">
  <div className="section-heading"><Eyebrow>Made to feel present</Eyebrow><h2 id="pillars-title">What makes Akane different</h2></div>
  <div className="pillar-grid">{companionPillars.map(([icon, title, text]) => <article className="pillar-card" key={title}><HomeIcon name={icon} /><h3>{title}</h3><p>{text}</p></article>)}</div>
</section>; }

function ConversationCard({ label, question, answer }: { label: string; question: string; answer: string }) { return <article className="conversation-example"><h3>{label}</h3><div><b>You</b><p>{question}</p></div><div className="akane-reply"><b>Akane</b><p>{answer}</p></div></article>; }

function ContinuityStory() { return <section className="home-section continuity-story shell" aria-labelledby="continuity-title">
  <div className="continuity-story-heading"><Eyebrow>Built through continuity</Eyebrow><h2 id="continuity-title">A conversation that does not start over</h2><p>Akane does not remember everything indiscriminately. She keeps what matters and brings it back when it genuinely changes the conversation.</p></div>
  <div className="continuity-story-flow">
    <ConversationCard label="Earlier conversation" question="I’m prioritizing the popup before I work on Live2D." answer="Good. A moving model will not help much if the actual companion experience still feels unfinished." />
    <div className="memory-retained"><HomeIcon name="memory" /><span aria-hidden="true">↓</span><strong>Meaningful memory retained</strong><span aria-hidden="true">↓</span></div>
    <ConversationCard label="Later conversation" question="What should I work on next?" answer="You were focusing on the popup first. Finish the streaming and window behavior, then Live2D will have something solid to build on." />
  </div>
  <small className="illustrative-note">Illustrative example of continuity, not a hardcoded production response.</small>
</section>; }

function InterfaceOverview() { const interfaces: Array<[HomeIconName, string, string]> = [["desktop", "Desktop popup", "Akane’s main desktop companion experience."], ["discord", "Discord", "Remote and mobile access to the same personal continuity."], ["website", "Website", "A temporary guest experience powered by the real model on the Raspberry Pi."]]; return <section className="home-section interface-overview shell" aria-labelledby="interface-title">
  <div className="interface-heading"><Eyebrow>One shared companion</Eyebrow><h2 id="interface-title">The same Akane, wherever you talk</h2><p>The desktop popup, Discord, and the website connect to the same conversation architecture while keeping personal and guest continuity isolated.</p></div>
  <div className="interface-simple" role="group" aria-label="Desktop popup, Discord, and Website connect to one Akane, powered by Gemma on a Raspberry Pi">
    <div className="interface-options">{interfaces.map(([icon, title, text]) => <article key={title}><HomeIcon name={icon} /><h3>{title}</h3><p>{text}</p></article>)}</div>
    <div className="interface-join" aria-hidden="true"><span /><span /><span /></div>
    <div className="interface-runtime"><strong>One Akane</strong><span aria-hidden="true">↓</span><div><HomeIcon name="pi" /><b>Gemma on Raspberry Pi</b></div></div>
  </div>
  <p className="interface-privacy"><strong>Private continuity:</strong> Desktop and Discord share your private continuity. Website visitors receive isolated temporary guest sessions.</p>
</section>; }

function DevelopmentStrip() { return <section className="home-section development-strip shell" aria-labelledby="development-title">
  <div className="development-title"><Eyebrow>Current development</Eyebrow><Link to="/technology">Explore the technology <span aria-hidden="true">→</span></Link></div>
  <h2 id="development-title" className="sr-only">Current development status</h2>
  <div className="development-columns"><div><h3>Available now</h3><ul>{availableNow.map(item => <li key={item}><span>{item}</span><StatusBadge /></li>)}</ul></div><div><h3>In development</h3><ul>{plannedWork.map(item => <li key={item}><span>{item}</span><StatusBadge kind="planned" /></li>)}</ul></div></div>
</section>; }

function HomeCallToAction() { return <section className="home-cta shell" aria-labelledby="cta-title"><Mark /><div><h2 id="cta-title">Ready to meet Akane?</h2><p>Experience the real companion pipeline through the live demo, or explore how she works on GitHub.</p></div><div className="actions"><Link className="button cta-primary" to="/demo">Try the Demo <span aria-hidden="true">→</span></Link><GithubLink className="button cta-secondary">View on GitHub</GithubLink></div></section>; }

function HomePage() { return <main className="home-page"><HomeHero /><ProductDemo /><CompanionPillars /><ContinuityStory /><InterfaceOverview /><DevelopmentStrip /><HomeCallToAction /></main>; }

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

function App() { const location = useLocation(); useEffect(() => { window.scrollTo(0, 0); }, [location.pathname]); return <><Navbar /><Routes><Route path="/" element={<HomePage />} /><Route path="/demo" element={<DemoPage />} /><Route path="/technology" element={<TechnologyPage />} /><Route path="*" element={<HomePage />} /></Routes><Footer /></>; }
export default App;
