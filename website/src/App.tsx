import { type ReactNode, useEffect, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { projectConfig } from "./config/project";
import { DemoPage } from "./pages/DemoPage";
import { HomePage } from "./pages/HomePage";
import { TechnologyPage } from "./pages/TechnologyPage";

const logo = `${projectConfig.basePath}assets/akane-logo.png`;

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
      <div className="footer-row footer-primary">
        <Link className="footer-brand" to="/">
          <Logo />
          <strong>Akane</strong>
        </Link>
        <nav className="footer-links" aria-label="Footer navigation">
          <Link to="/">Home</Link>
          <Link to="/demo">Demo</Link>
          <Link to="/technology">Technology</Link>
          <GithubLink className="plain-link">GitHub</GithubLink>
        </nav>
      </div>
      <div className="footer-row footer-meta">
        <span>Built by <a href={projectConfig.githubUrl} target="_blank" rel="noreferrer">Alexander Lee</a></span>
        <span>MIT License</span>
        <span>© {new Date().getFullYear()} Akane</span>
      </div>
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
