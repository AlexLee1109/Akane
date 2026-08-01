import { type ReactNode, useEffect, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { projectConfig } from "./config/project";
import { DemoPage } from "./pages/DemoPage";
import { TechnologyPage } from "./pages/TechnologyPage";

const logo = `${projectConfig.basePath}assets/akane-logo.png`;
const homepageImage = `${projectConfig.basePath}assets/homepage-image.png`;

type HomeIconName =
  | "code"
  | "desktop"
  | "discord"
  | "growth"
  | "memory"
  | "pi"
  | "presence"
  | "shield"
  | "website";

interface HomeCard {
  icon: HomeIconName;
  title: string;
  text: string;
}

interface RouteMetadata {
  title: string;
  description: string;
}

const routeMetadata: Record<string, RouteMetadata> = {
  "/": {
    title: "Akane · Local AI Companion",
    description: "Meet Akane, a private local AI companion that remembers meaningful experiences and stays consistent across desktop, Discord, and the web.",
  },
  "/demo": {
    title: "Meet Akane · Live Demo",
    description: "Talk to Akane through an isolated temporary guest session, with a clearly labeled prerecorded preview when the Raspberry Pi is unavailable.",
  },
  "/technology": {
    title: "How Akane Works · Technology",
    description: "Explore Akane’s local Raspberry Pi runtime, persistent continuity, profile isolation, streaming pipeline, and interface architecture.",
  },
};

const companionPillars: readonly HomeCard[] = [
  {
    icon: "memory",
    title: "Remembers you",
    text: "Persistent memories, meaningful decisions, preferences, and shared experiences can influence future conversations.",
  },
  {
    icon: "growth",
    title: "Develops with you",
    text: "Relationship, emotion, opinions, and established continuity help Akane respond as the same companion instead of resetting each session.",
  },
  {
    icon: "presence",
    title: "Stays present",
    text: "The desktop popup, Discord, and website use the same underlying companion architecture while keeping private and guest continuity isolated.",
  },
];

const continuitySteps = [
  { label: "Earlier", text: "“I’m finishing the popup before Live2D.”" },
  { label: "Remembered", text: "Popup first. Live2D later." },
  {
    label: "Later",
    text: "“Finish the streaming and window behavior first. Then Live2D will have something solid to build on.”",
  },
] as const;

const interfaces: readonly HomeCard[] = [
  { icon: "desktop", title: "Desktop popup", text: "Akane’s primary desktop companion interface." },
  { icon: "discord", title: "Discord", text: "Remote and mobile access to the owner’s private continuity." },
  {
    icon: "website",
    title: "Website",
    text: "An isolated temporary guest experience using the real model on the Pi.",
  },
];

const developmentGroups = [
  {
    status: "Available",
    items: ["Local conversation", "Persistent memory", "Desktop popup", "Discord", "Web guest demo"],
  },
  {
    status: "In development",
    items: ["Expression synchronization"],
  },
  {
    status: "Planned",
    items: ["Voice", "Live2D"],
  },
] as const;

function Logo() {
  return <img className="logo" src={logo} alt="Akane logo" />;
}

function GithubLink({ children, className }: { children: ReactNode; className: string }) {
  return projectConfig.githubUrl
    ? <a className={className} href={projectConfig.githubUrl} target="_blank" rel="noreferrer">{children}<span aria-hidden="true">↗</span></a>
    : <span className={`${className} disabled`} aria-disabled="true">{children}</span>;
}

function Navbar() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const links = [["/", "Home"], ["/demo", "Demo"], ["/technology", "Technology"]] as const;

  useEffect(() => setOpen(false), [location.pathname]);

  return <header className={`site-header ${location.pathname === "/" ? "home-header" : ""}`}>
    <nav className="nav shell" aria-label="Primary navigation">
      <Link className="brand" to="/">
        <Logo />
        <strong>Akane</strong>
        <span>AI COMPANION</span>
      </Link>
      <button
        className="menu-button"
        type="button"
        aria-expanded={open}
        aria-controls="nav-links"
        aria-label={`${open ? "Close" : "Open"} navigation menu`}
        onClick={() => setOpen(current => !current)}
      >
        {open ? "Close" : "Menu"}
      </button>
      <div id="nav-links" className={`nav-links ${open ? "open" : ""}`}>
        {links.map(([to, label]) => <NavLink key={to} to={to} end={to === "/"}>{label}</NavLink>)}
        <GithubLink className="nav-github">GitHub</GithubLink>
      </div>
    </nav>
  </header>;
}

function Footer() {
  return <footer className="footer shell">
    <div className="footer-brand">
      <Logo />
      <div>
        <strong>Akane</strong>
        <p>A local-first AI companion built around memory, continuity, and personal presence.</p>
        <small>Open source under the MIT License.</small>
      </div>
    </div>
    <div className="footer-creator">
      <span>Designed and built by</span>
      <a href={projectConfig.githubUrl} target="_blank" rel="noreferrer">Alexander Lee</a>
    </div>
    <div className="footer-links">
      <Link to="/">Home</Link>
      <Link to="/demo">Demo</Link>
      <Link to="/technology">Technology</Link>
      <GithubLink className="plain-link">GitHub</GithubLink>
    </div>
    <small className="footer-copyright">© {new Date().getFullYear()} Akane</small>
  </footer>;
}

function Eyebrow({ children }: { children: ReactNode }) {
  return <p className="eyebrow">{children}</p>;
}

function HomeIcon({ name }: { name: HomeIconName }) {
  const paths: Record<HomeIconName, ReactNode> = {
    code: <><path d="m8.5 8-4 4 4 4M15.5 8l4 4-4 4M14 5l-4 14" /></>,
    desktop: <><rect x="3.5" y="4.5" width="17" height="12" rx="1.5" /><path d="M8.5 20h7M12 16.5V20" /></>,
    discord: <><path d="M5 6.5c4.5-2 9.5-2 14 0l1.2 9.2a14 14 0 0 1-4.2 2.1l-1-1.5M19 6.5l-1.2 9.2A14 14 0 0 1 13.5 18" /><circle cx="9" cy="12" r="1" /><circle cx="15" cy="12" r="1" /></>,
    growth: <><path d="M12 20V10" /><path d="M12 12c-4 0-6-2.5-6-6 4 0 6 2.5 6 6ZM12 15c4 0 6-2.5 6-6-4 0-6 2.5-6 6Z" /></>,
    memory: <><path d="M5 5.5h14v10H9l-4 3v-13Z" /><path d="M8.5 9h7M8.5 12h4.5" /></>,
    pi: <><rect x="5" y="5" width="14" height="14" rx="2" /><path d="M9 2.5v2.3M15 2.5v2.3M9 19.2v2.3M15 19.2v2.3M2.5 9h2.3M2.5 15h2.3M19.2 9h2.3M19.2 15h2.3" /><circle cx="12" cy="12" r="3" /></>,
    presence: <><circle cx="12" cy="12" r="2.5" /><path d="M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4M4.7 4.7a10.3 10.3 0 0 0 0 14.6M19.3 4.7a10.3 10.3 0 0 1 0 14.6" /></>,
    shield: <><path d="M12 3 19 6v5c0 4.8-3 8.2-7 10-4-1.8-7-5.2-7-10V6l7-3Z" /><path d="m9 12 2 2 4-5" /></>,
    website: <><circle cx="12" cy="12" r="8.5" /><path d="M3.7 12h16.6M12 3.5c2.2 2.4 3.3 5.2 3.3 8.5S14.2 18.1 12 20.5C9.8 18.1 8.7 15.3 8.7 12S9.8 5.9 12 3.5Z" /></>,
  };

  return <svg className="home-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">{paths[name]}</svg>;
}

function HomeHero() {
  const trustItems: readonly [HomeIconName, string][] = [
    ["pi", "Runs on Raspberry Pi"],
    ["shield", "Your memories stay private"],
    ["code", "Open source"],
  ];

  return <section className="home-hero" aria-labelledby="home-title">
    <picture className="home-hero-media">
      <source
        type="image/jpeg"
        srcSet={`${projectConfig.basePath}assets/homepage-image-720.jpg 720w, ${projectConfig.basePath}assets/homepage-image-1100.jpg 1100w, ${projectConfig.basePath}assets/homepage-image-1448.jpg 1448w`}
        sizes="100vw"
      />
      <img
        src={homepageImage}
        width="1448"
        height="1086"
        fetchPriority="high"
        decoding="async"
        alt="Akane, a blue-haired AI companion, standing in a bright room overlooking a city"
      />
    </picture>
    <div className="home-hero-copy shell">
      <p className="hero-eyebrow">Local-first <span>•</span> Private <span>•</span> Always yours</p>
      <h1 id="home-title">A local AI companion<br />that remembers you.</h1>
      <p className="hero-accent">Always by your side.</p>
      <p className="home-lead">Akane develops through conversation, remembers meaningful experiences, and stays present across the desktop, Discord, and the web—powered by a local model running on a Raspberry Pi 5.</p>
      <div className="actions hero-actions">
        <Link className="button primary" to="/demo">Try the Demo<span aria-hidden="true">→</span></Link>
      </div>
      <ul className="trust-list" aria-label="Project facts">
        {trustItems.map(([icon, label]) => <li key={label}><HomeIcon name={icon} />{label}</li>)}
      </ul>
    </div>
  </section>;
}

function CompanionPillars() {
  return <section className="home-section pillars-section shell" aria-labelledby="pillars-title">
    <div className="section-heading">
      <Eyebrow>Made to feel present</Eyebrow>
      <h2 id="pillars-title">What makes Akane different</h2>
    </div>
    <div className="pillar-grid">
      {companionPillars.map(item => <article className="pillar-card" key={item.title}>
        <HomeIcon name={item.icon} />
        <h3>{item.title}</h3>
        <p>{item.text}</p>
      </article>)}
    </div>
  </section>;
}

function ContinuityStory() {
  return <section className="home-section continuity-story" aria-labelledby="continuity-title">
    <div className="shell continuity-inner">
      <div className="continuity-story-heading">
        <Eyebrow>Built through continuity</Eyebrow>
        <h2 id="continuity-title">A conversation that does not start over</h2>
        <p>Akane keeps what matters and brings it back when it genuinely changes a future conversation.</p>
      </div>
      <ol className="continuity-story-flow">
        {continuitySteps.map((step, index) => <li key={step.label}>
          <span className="continuity-step-number" aria-hidden="true">0{index + 1}</span>
          <div><h3>{step.label}</h3><p>{step.text}</p></div>
        </li>)}
      </ol>
      <small className="illustrative-note">Illustrative example—not a real stored user conversation.</small>
    </div>
  </section>;
}

function InterfaceOverview() {
  return <section className="home-section interface-overview" aria-labelledby="interface-title">
    <div className="shell interface-inner">
      <div className="interface-heading">
        <Eyebrow>One shared companion</Eyebrow>
        <h2 id="interface-title">The same Akane, wherever you talk</h2>
      </div>
      <div className="interface-simple" role="group" aria-label="Desktop popup, Discord, and Website connect to one Akane, with Gemma running on a Raspberry Pi">
        <div className="interface-options">
          {interfaces.map(item => <article key={item.title}>
            <HomeIcon name={item.icon} />
            <h3>{item.title}</h3>
            <p>{item.text}</p>
          </article>)}
        </div>
        <div className="interface-join" aria-hidden="true"><span /><span /><span /></div>
        <div className="interface-runtime">
          <strong>One Akane</strong>
          <span aria-hidden="true">↓</span>
          <div><HomeIcon name="pi" /><b>Gemma on Raspberry Pi</b></div>
        </div>
      </div>
      <p className="interface-privacy"><strong>Private continuity:</strong> Desktop and Discord share the owner’s private continuity. Website visitors receive separate temporary guest sessions.</p>
    </div>
  </section>;
}

function DevelopmentStrip() {
  return <section className="home-section development-strip shell" aria-labelledby="development-title">
    <div className="development-title">
      <div><Eyebrow>Current development</Eyebrow><h2 id="development-title">What works today</h2></div>
      <Link to="/technology">Explore the technology <span aria-hidden="true">→</span></Link>
    </div>
    <div className="development-columns">
      {developmentGroups.map(group => <article key={group.status}>
        <h3>{group.status}</h3>
        <ul>{group.items.map(item => <li key={item}>{item}</li>)}</ul>
      </article>)}
    </div>
  </section>;
}

function HomeCallToAction() {
  return <section className="home-cta" aria-labelledby="cta-title">
    <div className="shell home-cta-inner">
      <div>
        <h2 id="cta-title">Ready to meet Akane?</h2>
        <p>Talk through the live guest demo or explore how the project works.</p>
      </div>
      <div className="actions">
        <Link className="button cta-primary" to="/demo">Try the Demo</Link>
      </div>
    </div>
  </section>;
}

function HomePage() {
  return <main className="home-page">
    <HomeHero />
    <CompanionPillars />
    <ContinuityStory />
    <InterfaceOverview />
    <DevelopmentStrip />
    <HomeCallToAction />
  </main>;
}

function updateMetadata(metadata: RouteMetadata) {
  document.title = metadata.title;
  const values = [
    ["meta[name='description']", metadata.description],
    ["meta[property='og:title']", metadata.title],
    ["meta[property='og:description']", metadata.description],
    ["meta[name='twitter:title']", metadata.title],
    ["meta[name='twitter:description']", metadata.description],
  ] as const;
  values.forEach(([selector, value]) => document.querySelector<HTMLMetaElement>(selector)?.setAttribute("content", value));
}

function App() {
  const location = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
    updateMetadata(routeMetadata[location.pathname] || routeMetadata["/"]);
  }, [location.pathname]);

  return <>
    <Navbar />
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/demo" element={<DemoPage />} />
      <Route path="/technology" element={<TechnologyPage />} />
      <Route path="*" element={<HomePage />} />
    </Routes>
    <Footer />
  </>;
}

export default App;
