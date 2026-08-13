import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { projectConfig } from "../config/project";
import "./home.css";

const homepageImage = `${projectConfig.basePath}assets/homepage-image.png`;

type HomeIconName = "desktop" | "discord" | "growth" | "memory" | "pi" | "presence" | "shield" | "web";

const companionPillars = [
  {
    icon: "memory",
    title: "Remembers you",
    text: "Meaningful memories and shared experiences carry into future conversations.",
  },
  {
    icon: "growth",
    title: "Develops with you",
    text: "Akane maintains opinions, emotion, and relationship continuity instead of resetting.",
  },
  {
    icon: "presence",
    title: "Stays present",
    text: "One companion architecture connects the desktop, Discord, and web.",
  },
] as const;

const continuitySteps = [
  { label: "Earlier", text: "“I’m finishing the popup before Live2D.”" },
  { label: "Remembered", text: "Popup first. Live2D later." },
  { label: "Later", text: "“Finish streaming and window behavior first. Then move to Live2D.”" },
] as const;

const interfaces = [
  { icon: "desktop", label: "Desktop" },
  { icon: "discord", label: "Discord" },
  { icon: "web", label: "Web" },
] as const;

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

function Eyebrow({ children }: { children: ReactNode }) {
  return <p className="home-eyebrow">{children}</p>;
}

function HomeHero() {
  const trustItems: readonly [HomeIconName, string][] = [
    ["pi", "Runs on Raspberry Pi"],
    ["shield", "Private personal memory"],
  ];

  return <section className="home-hero" aria-labelledby="home-title">
    <img
      className="home-hero-media"
      src={homepageImage}
      fetchPriority="high"
      decoding="async"
      alt="Akane, a blue-haired AI companion, standing in a bright room overlooking a city"
    />
    <div className="home-hero-copy shell">
      <p className="home-hero-eyebrow">Local-first <span>•</span> Private <span>•</span> Always yours</p>
      <h1 id="home-title">A local AI companion<br />that remembers you.</h1>
      <p className="home-hero-accent">Always by your side.</p>
      <p className="home-lead">Akane develops through conversation, remembers meaningful experiences, and stays present across the desktop, Discord, and the web—powered by a local model running on a Raspberry Pi 5.</p>
      <div className="actions home-hero-actions">
        <Link className="button primary" to="/demo">Try the Demo<span aria-hidden="true">→</span></Link>
      </div>
      <ul className="home-trust-list" aria-label="Project facts">
        {trustItems.map(([icon, label]) => <li key={label}><HomeIcon name={icon} />{label}</li>)}
      </ul>
    </div>
  </section>;
}

function CompanionPillars() {
  return <section className="home-section home-pillars shell" aria-labelledby="pillars-title">
    <div className="home-section-heading">
      <Eyebrow>Made to feel present</Eyebrow>
      <h2 id="pillars-title">What makes Akane different</h2>
    </div>
    <div className="home-pillar-grid">
      {companionPillars.map(item => <article className="home-pillar-card" key={item.title}>
        <HomeIcon name={item.icon} />
        <h3>{item.title}</h3>
        <p>{item.text}</p>
      </article>)}
    </div>
  </section>;
}

function ContinuityStory() {
  return <section className="home-section home-continuity" aria-labelledby="continuity-title">
    <div className="shell home-continuity-inner">
      <div className="home-continuity-copy">
        <Eyebrow>Built through continuity</Eyebrow>
        <h2 id="continuity-title">A conversation that does not start over</h2>
        <p>Akane keeps what matters and brings it back when it genuinely changes a future conversation.</p>
      </div>
      <div>
        <ol className="home-continuity-flow">
          {continuitySteps.map((step, index) => <li key={step.label}>
            <span aria-hidden="true">0{index + 1}</span>
            <div><h3>{step.label}</h3><p>{step.text}</p></div>
          </li>)}
        </ol>
        <p className="home-illustrative-note">Illustrative example—not a real stored conversation.</p>
      </div>
    </div>
  </section>;
}

function InterfaceOverview() {
  return <section className="home-section home-interfaces" aria-labelledby="interfaces-title">
    <div className="shell">
      <div className="home-interface-heading">
        <Eyebrow>One shared companion</Eyebrow>
        <h2 id="interfaces-title">The same Akane, wherever you talk</h2>
      </div>
      <div className="home-interface-flow" role="group" aria-label="Desktop, Discord, and Web connect to one Akane">
        <div className="home-interface-options">
          {interfaces.map(item => <div key={item.label}><HomeIcon name={item.icon} /><strong>{item.label}</strong></div>)}
        </div>
        <span className="home-interface-arrow" aria-hidden="true">→</span>
        <div className="home-interface-akane"><strong>One Akane</strong><span>Local {projectConfig.modelName} runtime</span></div>
      </div>
      <div className="home-interface-notes">
        <p>Desktop and Discord share private owner continuity. Web visitors receive isolated temporary guest sessions.</p>
        <p><strong>Available now:</strong> local conversation, persistent memory, popup, Discord, and web demo.</p>
        <p>Voice and Live2D remain planned.</p>
        <Link to="/technology">View development details <span aria-hidden="true">→</span></Link>
      </div>
    </div>
  </section>;
}

function HomeCallToAction() {
  return <section className="home-cta" aria-labelledby="home-cta-title">
    <div className="shell home-cta-inner">
      <div>
        <h2 id="home-cta-title">Ready to meet Akane?</h2>
        <p>Talk through the live guest demo and experience the real local companion pipeline.</p>
      </div>
      <Link className="button home-cta-button" to="/demo">Try the Demo</Link>
    </div>
  </section>;
}

export function HomePage() {
  return <main className="home-page">
    <HomeHero />
    <CompanionPillars />
    <ContinuityStory />
    <InterfaceOverview />
    <HomeCallToAction />
  </main>;
}
