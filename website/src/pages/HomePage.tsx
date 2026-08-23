import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { projectConfig } from "../config/project";
import { useInView } from "../lib/useInView";
import "./home.css";

const homepageImage = `${projectConfig.basePath}assets/homepage-image.png`;
const characterImage = `${projectConfig.basePath}assets/akane-hero.png`;
type HomeIconName = "desktop" | "discord" | "growth" | "memory" | "pi" | "presence" | "shield" | "web";
type CapabilityId = "memory" | "growth" | "presence";
type InterfaceId = "desktop" | "discord" | "web";

const capabilities: readonly { id: CapabilityId; icon: HomeIconName; label: string; title: string; text: string; note: string }[] = [
  { id: "memory", icon: "memory", label: "Memory", title: "Meaning can return.", text: "Akane can retain selected experiences and useful details, then bring them back when they genuinely matter to a later conversation.", note: "Selected context, not a transcript dump" },
  { id: "growth", icon: "growth", label: "Development / Self", title: "A Self that can develop.", text: "Preferences, interests, and opinions can take shape over time instead of disappearing when a conversation ends.", note: "Continuity that develops gradually" },
  { id: "presence", icon: "presence", label: "Presence", title: "There is a sense of between.", text: "Lightweight InnerLife gives Akane a small amount of offscreen continuity while keeping conversation grounded in real stored state.", note: "Quiet continuity between conversations" },
];

const interfaces: readonly { id: InterfaceId; icon: HomeIconName; label: string; eyebrow: string; title: string; text: string }[] = [
  { id: "desktop", icon: "desktop", label: "Desktop", eyebrow: "Personal popup", title: "A quiet window on your desktop", text: "The owner interface keeps Akane close and uses the same private continuity as Discord." },
  { id: "discord", icon: "discord", label: "Discord", eyebrow: "Owner continuity", title: "The same Akane in Discord", text: "Discord adapts the conversation to the platform without creating a separate companion or profile." },
  { id: "web", icon: "web", label: "Web", eyebrow: "Temporary guest", title: "A safe way to meet Akane", text: "Every visitor receives an isolated temporary session that never touches the owner’s private continuity." },
];

function HomeIcon({ name }: { name: HomeIconName }) {
  const paths: Record<HomeIconName, ReactNode> = {
    desktop: <><rect x="3.5" y="4.5" width="17" height="12" rx="1.5" /><path d="M8.5 20h7M12 16.5V20" /></>,
    discord: <><path d="M5 6.5c4.5-2 9.5-2 14 0l1.2 9.2a14 14 0 0 1-4.2 2.1l-1-1.5M19 6.5l-1.2 9.2A14 14 0 0 1 13.5 18" /><circle cx="9" cy="12" r="1" /><circle cx="15" cy="12" r="1" /></>,
    growth: <><path d="M12 20V10" /><path d="M12 12c-4 0-6-2.5-6-6 4 0 6 2.5 6 6ZM12 15c4 0 6-2.5 6-6-4 0-6 2.5-6 6Z" /></>,
    memory: <><path d="M5 5.5h14v10H9l-4 3v-13Z" /><path d="M8.5 9h7M8.5 12h4.5" /></>,
    pi: <><rect x="5" y="5" width="14" height="14" rx="2" /><path d="M9 2.5v2.3M15 2.5v2.3M9 19.2v2.3M15 19.2v2.3M2.5 9h2.3M2.5 15h2.3M19.2 9h2.3M19.2 15h2.3" /><circle cx="12" cy="12" r="3" /></>,
    presence: <><circle cx="12" cy="12" r="2.5" /><path d="M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4M4.7 4.7a10.3 10.3 0 0 0 0 14.6M19.3 4.7a10.3 10.3 0 0 1 0 14.6" /></>,
    shield: <><path d="M12 3 19 6v5c0 4.8-3 8.2-7 10-4-1.8-7-5.2-7-10V6l7-3Z" /><path d="m9 12 2 2 4-5" /></>,
    web: <><circle cx="12" cy="12" r="8.5" /><path d="M3.7 12h16.6M12 3.5c2.2 2.4 3.3 5.2 3.3 8.5S14.2 18.1 12 20.5C9.8 18.1 8.7 15.3 8.7 12S9.8 5.9 12 3.5Z" /></>,
  };
  return <svg className="home-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">{paths[name]}</svg>;
}

function HomeHero() {
  return <section className="home-hero" aria-labelledby="home-title">
    <img className="home-hero-media" src={homepageImage} fetchPriority="high" decoding="async" alt="Akane, a blue-haired AI companion, standing in a bright room overlooking a city" />
    <div className="home-hero-wash" aria-hidden="true" />
    <div className="home-hero-copy shell">
      <p className="eyebrow home-hero-eyebrow">Local AI companion</p>
      <h1 id="home-title">Meet Akane.</h1>
      <p className="home-lead">A personal AI companion with memory, continuity, and a life that carries on between conversations.</p>
      <div className="actions home-hero-actions"><Link className="button primary" to="/demo">Talk to Akane<span aria-hidden="true">→</span></Link><Link className="home-text-link" to="/technology">How Akane works<span aria-hidden="true">↗</span></Link></div>
      <ul className="home-trust-list" aria-label="Project facts"><li><HomeIcon name="pi" />Runs locally</li><li><HomeIcon name="shield" />Private memory</li><li><HomeIcon name="presence" />Persistent continuity</li></ul>
    </div>
    <div className="home-hero-detail" aria-hidden="true"><span className="home-detail-dot" /><div><small>Continuity</small><strong>Picks up where you left off</strong></div></div>
  </section>;
}

function CompanionPillars() {
  const [activeId, setActiveId] = useState<CapabilityId>("memory");
  const active = capabilities.find(item => item.id === activeId) || capabilities[0];

  return <section className="home-section home-pillars shell" aria-labelledby="pillars-title">
    <div className="section-heading"><p className="eyebrow">Made to feel present</p><h2 id="pillars-title">Not another blank chat window.</h2><p>Akane is designed around one relationship that develops through time.</p></div>
    <div className="capability-experience">
      <div className="capability-selector" role="tablist" aria-label="Explore Akane’s continuity">
        {capabilities.map((item, index) => <button key={item.id} type="button" role="tab" aria-selected={activeId === item.id} aria-controls="capability-panel" onClick={() => setActiveId(item.id)}>
          <span>0{index + 1}</span><HomeIcon name={item.icon} /><strong>{item.label}</strong>
        </button>)}
      </div>
      <div id="capability-panel" className={`capability-stage ${active.id}`} role="tabpanel" aria-live="polite">
        <div className="capability-copy" key={active.id}><p className="eyebrow">{active.label}</p><h3>{active.title}</h3><p>{active.text}</p><span>{active.note}</span></div>
        <div className={`capability-visual ${active.id}`} aria-hidden="true">
          {active.id === "memory" && <><i>Conversation</i><b>Selected meaning</b><i>Later context</i></>}
          {active.id === "growth" && <><i>Interest</i><i>Preference</i><b>Self</b><i>Opinion</i></>}
          {active.id === "presence" && <><i>Conversation</i><b>InnerLife</b><i>Return</i></>}
        </div>
        <img className="capability-akane" src={characterImage} alt="Akane standing against a soft blue celestial backdrop" loading="lazy" decoding="async" />
      </div>
    </div>
  </section>;
}

function ContinuityStory() {
  const { ref, visible } = useInView<HTMLElement>(0.28);
  return <section ref={ref} className={`home-section home-continuity ${visible ? "is-visible" : ""}`} aria-labelledby="continuity-title">
    <div className="shell home-continuity-heading"><p className="eyebrow">A conversation with a past</p><h2 id="continuity-title">The thread carries forward.</h2><p>Conversation becomes selected context, then returns naturally when it can help.</p></div>
    <div className="shell continuity-scene">
      <div className="continuity-chat earlier"><span>Earlier</span><div className="bubble you">I’m finishing the popup before Live2D.</div></div>
      <div className="continuity-connector" aria-hidden="true"><i /><span>Meaning retained</span><i /></div>
      <div className="continuity-memory"><HomeIcon name="memory" /><small>Remembered</small><strong>Popup first.<br />Live2D later.</strong><span>Relevant context</span></div>
      <div className="continuity-connector" aria-hidden="true"><i /><span>Later context</span><i /></div>
      <div className="continuity-chat later"><span>Later</span><div className="bubble akane">Let’s finish streaming and window behavior first. Then we can move to Live2D.</div><small>Akane</small></div>
    </div>
    <p className="home-illustrative-note shell">Illustrative flow—not a real stored conversation or user record.</p>
  </section>;
}

function InterfaceOverview() {
  const [activeId, setActiveId] = useState<InterfaceId>("desktop");
  const active = interfaces.find(item => item.id === activeId) || interfaces[0];

  return <section className="home-section home-interfaces" aria-labelledby="interfaces-title">
    <div className="shell"><div className="section-heading center"><p className="eyebrow">One shared companion</p><h2 id="interfaces-title">Different places. The same Akane.</h2><p>Each interface is a window into one coordinated local runtime, with clear privacy boundaries.</p></div>
      <div className="interface-showcase">
        <div className="interface-selector" role="tablist" aria-label="Akane interfaces">{interfaces.map(item => <button key={item.id} type="button" role="tab" aria-selected={activeId === item.id} aria-controls="interface-panel" onClick={() => setActiveId(item.id)}><HomeIcon name={item.icon} />{item.label}</button>)}</div>
        <div id="interface-panel" className={`interface-stage ${active.id}`} role="tabpanel" aria-live="polite">
          <div className="interface-stage-copy" key={active.id}><p className="eyebrow">{active.eyebrow}</p><h3>{active.title}</h3><p>{active.text}</p><small>Stylized interface preview</small></div>
          <div className={`interface-window ${active.id}`} aria-hidden="true">
            <header><i /><i /><i /><span>{active.label}</span></header>
            {active.id === "desktop" && <div className="desktop-preview"><b>A</b><div><span>Akane</span><p>Owner conversation ready</p></div></div>}
            {active.id === "discord" && <div className="discord-preview"><aside><i /><i /><i /></aside><div><span># akane</span><p><b>A</b> Same owner continuity</p></div></div>}
            {active.id === "web" && <div className="web-preview"><span>Temporary guest</span><b>Isolated session</b><i>Owner profile untouched</i></div>}
          </div>
        </div>
        <div className="interface-runtime"><span className="runtime-pulse" aria-hidden="true" /><div><small>Shared local runtime</small><strong>Akane · {projectConfig.modelName}</strong></div><Link to="/technology">See architecture<span aria-hidden="true">→</span></Link></div>
      </div>
    </div>
  </section>;
}

function LocalPrivate() {
  return <section className="home-local" aria-labelledby="local-title"><div className="shell home-local-inner"><div className="local-chip"><HomeIcon name="pi" /><span>Raspberry Pi 5</span></div><div><p className="eyebrow">Local by default</p><h2 id="local-title">Her core stays close.</h2><p>Core inference runs on a local model. Private owner continuity persists on personal hardware, without a cloud-first model dependency.</p></div><dl><div><dt>Model</dt><dd>{projectConfig.modelName}</dd></div><div><dt>Persistence</dt><dd>Private local state</dd></div><div><dt>Purpose</dt><dd>One personal companion</dd></div></dl></div></section>;
}

function HomeCallToAction() {
  return <section className="home-cta" aria-labelledby="home-cta-title"><div className="shell home-cta-inner"><div><p className="eyebrow">Say hello</p><h2 id="home-cta-title">Ready to meet Akane?</h2><p>Try a temporary, isolated guest conversation—or a clearly labeled prerecorded preview when the Pi is unavailable.</p></div><Link className="button home-cta-button" to="/demo">Try the Demo<span aria-hidden="true">→</span></Link></div></section>;
}

export function HomePage() { return <main className="home-page"><HomeHero /><CompanionPillars /><ContinuityStory /><InterfaceOverview /><LocalPrivate /><HomeCallToAction /></main>; }
