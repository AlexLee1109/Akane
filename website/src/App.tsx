import { ReactNode, useEffect, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { projectConfig } from "./config/project";
import { DemoPage } from "./pages/DemoPage";
import { TechnologyPage } from "./pages/TechnologyPage";

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

function App() { const location = useLocation(); useEffect(() => { window.scrollTo(0, 0); }, [location.pathname]); return <><Navbar /><Routes><Route path="/" element={<HomePage />} /><Route path="/demo" element={<DemoPage />} /><Route path="/technology" element={<TechnologyPage />} /><Route path="*" element={<HomePage />} /></Routes><Footer /></>; }
export default App;
