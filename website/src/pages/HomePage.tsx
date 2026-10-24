import { Link } from "react-router-dom";
import { Architecture } from "../components/Architecture";
import { Closing } from "../components/Closing";
import { projectConfig } from "../config/project";
import "./home.css";

const assets = `${projectConfig.basePath}assets/`;

function Hero() {
  return <section className="home-hero shell" aria-labelledby="home-title">
    <div className="hero-copy">
      <p className="eyebrow"><span className="eyebrow-line" /> A little more familiar, every day</p>
      <h1 id="home-title">An AI companion<br />who <em>grows<br />with you.</em></h1>
      <p className="page-lead">Meet Akane. Lasting memory, a developing perspective, and conversations that carry forward. All built on a local-first foundation.</p>
      <div className="actions"><Link className="button primary" to="/demo">Talk to Akane <span aria-hidden="true">↗</span></Link><Link className="text-link" to="/technology">Explore the technology <span aria-hidden="true">→</span></Link></div>
      <div className="hero-footnote"><span className="small-orbit" aria-hidden="true" /><span>One character. A continuing story.</span></div>
    </div>
    <figure className="hero-portrait">
      <picture><source media="(max-width: 600px)" srcSet={`${assets}akane-night-mobile.webp`} /><img src={`${assets}akane-night-hero-1280.webp`} srcSet={`${assets}akane-night-hero-1280.webp 1280w, ${assets}akane-night-hero.webp 1672w`} sizes="(max-width: 800px) 100vw, 55vw" width="1672" height="941" alt="Akane beside a window, her blue hair and white jacket illuminated by the city at night" fetchPriority="high" /></picture>
      <figcaption><div><span className="portrait-kicker">MEET YOUR COMPANION</span><strong>Akane <span lang="ja">茜</span></strong></div><span>A familiar presence.<br />A perspective of her own.</span></figcaption>
    </figure>
  </section>;
}

function Continuity() {
  return <section className="continuity light-section section" aria-labelledby="continuity-title"><div className="shell">
    <div className="section-intro split-intro"><div><p className="eyebrow">01 / Built for continuity</p><h2 id="continuity-title">More than a chat.<br /><em>Someone to meet again.</em></h2></div><p>Getting to know someone takes time. Akane is an experiment in giving that time a place to live.</p></div>
    <div className="continuity-grid">
      <article className="memory-story"><span className="feature-index">PERSISTENT MEMORY</span><h3>Some things are<br />worth keeping.</h3><p>The details that matter can find their way into the next conversation. A shared plan. A favorite song. Something to pick up next time.</p><div className="memory-example"><div className="memory-example-top"><span className="notebook-mark" aria-hidden="true">↳</span><span>From a previous conversation</span></div><p>“You were working on that<br />rainy-day playlist.”</p><div className="memory-example-bottom"><span className="diagram-dot" /> Relevant memory <span>Illustrative example</span></div></div></article>
      <article className="identity-story"><span className="feature-index">DEVELOPING IDENTITY</span><h3>A perspective.<br />With room to change.</h3><p>Akane begins with a recognizable character and voice. Grounded experience can shape her interests, preferences, and opinions over time.</p><div className="identity-path" aria-label="Experience informs perspective, which can be revised"><div><span>01</span><strong>Experience</strong><small>Something to draw on</small></div><b aria-hidden="true">→</b><div><span>02</span><strong>Perspective</strong><small>A point of view</small></div><b aria-hidden="true">→</b><div><span>03</span><strong>Revision</strong><small>Room to reconsider</small></div></div></article>
    </div>
    <div className="continuity-note"><span aria-hidden="true">↳</span><p>Continuity comes from selective, useful memory. Every conversation adds context; every detail doesn’t need to stay forever.</p></div>
  </div></section>;
}

function Conversation() {
  return <section className="section conversation-section"><div className="shell conversation-grid">
    <div className="conversation-copy"><p className="eyebrow">02 / In good company</p><h2>Pick up a thought.<br /><em>Find a new one.</em></h2><p>A passing hello, a different opinion, or that conversation you never quite finished. There’s room for all of it.</p><Link className="text-link" to="/demo">Find your first conversation <span aria-hidden="true">↗</span></Link><div className="guest-note"><span aria-hidden="true">↳</span><p>The public demo is a temporary guest space, separate from Akane’s private owner memory.</p></div></div>
    <div className="conversation-showcase"><div className="showcase-art"><img src={`${assets}akane-conversation-600.webp`} width="600" height="450" loading="lazy" alt="Akane beside a sunlit window in a bright room" /></div><div className="sample-chat"><header><img src={`${assets}akane-logo-192.png`} width="36" height="36" alt="" loading="lazy" /><div><strong>Akane</strong><span>A continuing conversation</span></div><span className="sample-label">Illustrative example</span></header><div className="sample-messages"><p className="sample-you">Back to that rainy-day playlist?</p><p className="sample-akane">I’d keep the piano track. The slower one.</p><p className="sample-you">And the last song?</p><p className="sample-akane">Still a little too loud for rain.</p></div><footer>Written example · Not a live AI response</footer></div></div>
  </div></section>;
}

function LocalFirst() {
  return <section className="section local-section light-section"><div className="shell local-grid"><div><p className="eyebrow">03 / Close to home</p><h2>Personal by nature.<br /><em>Local by design.</em></h2><p>Akane’s core runs on personal hardware. The model, memory, and persistent state live in a system you control.</p><p>The website is a window into that system. The public demo connects to a separate HTTPS backend; connected services such as Discord use their own platforms.</p><Link className="text-link" to="/technology">Inside the architecture <span aria-hidden="true">→</span></Link></div><Architecture compact /></div></section>;
}

export function HomePage() {
  return <main id="main-content" tabIndex={-1}><Hero /><div className="principles-strip shell" aria-label="Akane’s foundations"><span>Made for a continuing connection</span><div><span>Lasting memory</span><span>Developing identity</span><span>Local-first AI</span></div></div><Continuity /><Conversation /><LocalFirst /><Closing /></main>;
}
