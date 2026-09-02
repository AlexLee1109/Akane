import { type ReactNode, useEffect, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { projectConfig } from "./config/project";
import { DemoPage } from "./pages/DemoPage";
import { HomePage } from "./pages/HomePage";
import { AboutPage } from "./pages/AboutPage";
import { TechnologyPage } from "./pages/TechnologyPage";

const logo = `${projectConfig.basePath}assets/akane-logo-192.png`;

interface RouteMetadata {
  title: string;
  description: string;
}

const routeMetadata: Record<string, RouteMetadata> = {
  "/": {
    title: "Akane · Local AI Companion",
    description: "Meet Akane, a local AI companion who remembers what matters, develops her own preferences, and stays the same person across conversations.",
  },
  "/demo": {
    title: "Meet Akane · Live Demo",
    description: "Talk to Akane through an isolated temporary guest session, or try a clearly labeled simulated preview when she is offline.",
  },
  "/technology": {
    title: "How Akane Works · Technology",
    description: "See how one local generation produces Akane’s reply and grounded evidence for a persistent, developing Self.",
  },
  "/about": {
    title: "About Akane · Project Story",
    description: "Why Alexander Lee started Akane, chose local inference, and is building one companion who can develop over time.",
  },
};

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
  const links = [["/", "Home"], ["/demo", "Demo"], ["/technology", "Technology"], ["/about", "About"]] as const;

  useEffect(() => setOpen(false), [location.pathname]);

  return <header className={`site-header ${location.pathname === "/" ? "home-header" : ""}`}>
    <nav className="nav shell" aria-label="Primary navigation">
      <Link className="brand" to="/" aria-label="Akane home">
        <Logo />
        <strong>Akane</strong>
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
  return <footer className="footer">
    <div className="shell footer-inner">
      <Link className="footer-brand" to="/"><Logo /><strong>Akane</strong></Link>
      <span>Built by Alexander Lee · <GithubLink className="plain-link">GitHub</GithubLink></span>
      <span>© {new Date().getFullYear()}</span>
    </div>
  </footer>;
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
    <div className="route-view" key={location.pathname}>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="/technology" element={<TechnologyPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="*" element={<HomePage />} />
      </Routes>
    </div>
    {location.pathname !== "/demo" && <Footer />}
  </>;
}

export default App;
