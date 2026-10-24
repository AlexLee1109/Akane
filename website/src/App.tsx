import { lazy, Suspense, type ReactNode, useEffect, useRef, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { projectConfig } from "./config/project";
const HomePage = lazy(() => import("./pages/HomePage").then(module => ({ default: module.HomePage })));
const DemoPage = lazy(() => import("./pages/DemoPage").then(module => ({ default: module.DemoPage })));
const AboutPage = lazy(() => import("./pages/AboutPage").then(module => ({ default: module.AboutPage })));
const TechnologyPage = lazy(() => import("./pages/TechnologyPage").then(module => ({ default: module.TechnologyPage })));

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
    description: "Talk to Akane in a private temporary conversation, or try a clearly labeled simulated preview when she is offline.",
  },
  "/technology": {
    title: "How Akane Works · Technology",
    description: "Explore Akane’s local inference, grounded memory, developing Self, and isolated public sessions.",
  },
  "/about": {
    title: "About Akane · Project Story",
    description: "Why Alexander Lee started Akane, chose local inference, and is building one companion who can develop over time.",
  },
};

function Logo() {
  return <img className="logo" src={logo} width="32" height="32" alt="" />;
}

function GithubLink({ children, className }: { children: ReactNode; className: string }) {
  return projectConfig.githubUrl
    ? <a className={className} href={projectConfig.githubUrl} target="_blank" rel="noreferrer">{children}<span aria-hidden="true">↗</span></a>
    : <span className={`${className} disabled`} aria-disabled="true">{children}</span>;
}

function Navbar() {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();
  const headerRef = useRef<HTMLElement>(null);
  const menuRef = useRef<HTMLButtonElement>(null);
  const links = [["/", "Home"], ["/demo", "Demo"], ["/technology", "Technology"], ["/about", "About"]] as const;

  useEffect(() => setOpen(false), [location.pathname]);
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") { setOpen(false); menuRef.current?.focus(); }
    };
    const onPointerDown = (event: PointerEvent) => {
      if (!headerRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("pointerdown", onPointerDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("pointerdown", onPointerDown);
    };
  }, [open]);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return <header ref={headerRef} className={`site-header ${scrolled ? "is-scrolled" : ""}`}>
    <nav className="nav shell" aria-label="Primary navigation">
      <Link className="brand" to="/" aria-label="Akane home">
        <Logo />
        <strong>Akane</strong>
        <span className="brand-subtitle">A continuing companion</span>
      </Link>
      <button
        className="menu-button"
        ref={menuRef}
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
        <Link className="button primary nav-cta" to="/demo">Talk to Akane <span aria-hidden="true">↗</span></Link>
      </div>
    </nav>
  </header>;
}

function Footer() {
  return <footer className="footer">
    <div className="shell">
      <div className="footer-top"><div><Link className="footer-brand" to="/"><Logo /><strong>Akane</strong></Link><p>A continuing companion. Built close to home.</p></div>
        <nav aria-label="Footer navigation"><Link to="/demo">Meet Akane</Link><Link to="/technology">Technology</Link><Link to="/about">Our story</Link><GithubLink className="plain-link">GitHub</GithubLink></nav>
      </div>
      <div className="footer-bottom"><span>© {new Date().getFullYear()} · Built by Alexander Lee</span><span>Local-first AI · An independent project</span></div>
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
    <a className="skip-link" href="#main-content" onClick={event => {
      event.preventDefault();
      document.getElementById("main-content")?.focus();
    }}>Skip to content</a>
    <Navbar />
    <div className="route-view" key={location.pathname}>
      <Suspense fallback={<div className="route-loading" role="status">Opening Akane’s world…</div>}><Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="/technology" element={<TechnologyPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="*" element={<HomePage />} />
      </Routes></Suspense>
    </div>
    {location.pathname !== "/demo" && <Footer />}
  </>;
}

export default App;
