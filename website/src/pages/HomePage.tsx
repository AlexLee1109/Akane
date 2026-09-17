import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { projectConfig } from "../config/project";
import "./home.css";

const homepageImage = `${projectConfig.basePath}assets/akane-night-hero.webp`;
const characterImage = `${projectConfig.basePath}assets/akane-hero.png`;

type HomeIconName = "desktop" | "discord" | "growth" | "memory" | "pi" | "presence" | "shield" | "web";

const differentiators: readonly { icon: HomeIconName; title: string; text: string }[] = [
  { icon: "memory", title: "Memory", text: "Important experiences can return when they are relevant." },
  { icon: "growth", title: "Developing Self", text: "Preferences, interests, opinions, and goals can form over time." },
  { icon: "presence", title: "Continuity", text: "Akane remains one continuing companion instead of resetting every conversation." },
];

const interfaces: readonly { icon: HomeIconName; title: string; text: string }[] = [
  { icon: "desktop", title: "Desktop popup", text: "A quiet owner interface that keeps Akane close." },
  { icon: "discord", title: "Discord", text: "The same private owner history in a different setting." },
  { icon: "web", title: "Website demo", text: "A private temporary conversation for meeting Akane safely." },
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

const heroFeatures = [
  { icon: "memory", title: "Natural Conversations", text: <>More than answers.<br />Real understanding.</> },
  { icon: "presence", title: "Always With You", text: <>Support, inspire,<br />and grow together.</> },
  { icon: "growth", title: "A Kinder Future", text: <>Technology that<br />brings us closer.</> },
] as const;

function HomeHero() {
  return <section className="home-hero" aria-labelledby="home-title">
    <picture className="home-hero-artwork">
      <source media="(max-width: 700px)" srcSet={`${projectConfig.basePath}assets/akane-night-mobile.webp`} />
      <img className="home-hero-media" src={homepageImage} srcSet={`${projectConfig.basePath}assets/akane-night-hero-1280.webp 1280w, ${homepageImage} 1672w`} sizes="100vw" width="1672" height="941" fetchPriority="high" decoding="async" alt="Akane sitting beside a window, her blue hair and white-and-blue jacket lit by a starry city night" />
    </picture>
    <div className="home-hero-wash" aria-hidden="true" />
    <div className="home-hero-copy">
      <p className="home-eyebrow"><span aria-hidden="true">✦</span> Always by your side</p>
      <h1 id="home-title">A more<br /><em>human AI</em><br />companion</h1>
      <p className="home-lead">Akane is a local, evolving AI companion —<br className="home-desktop-break" /> not just smarter, but kinder, more understanding,<br className="home-desktop-break" /> and truly present in your everyday life.</p>
      <div className="home-hero-actions">
        <Link className="button primary" to="/demo">Try the Demo <span aria-hidden="true">→</span></Link>
        <button className="button home-video" type="button" disabled><span className="home-play" aria-hidden="true">▷</span><span>Watch Video<small>Coming soon</small></span></button>
      </div>
    </div>
    <div className="home-hero-features">{heroFeatures.map(item => <article key={item.title}><HomeIcon name={item.icon} /><h2>{item.title}</h2><p>{item.text}</p></article>)}</div>
    <div className="home-handwritten" aria-hidden="true">Same sky.<br />A brighter tomorrow.<span>— Akane</span></div>
    <div className="home-concept" aria-hidden="true">People<span>×</span>AI<span>×</span>Technology<span>×</span>A kinder<br />tomorrow</div>
    <aside className="home-message" aria-label="Upcoming message from Akane">
      <span className="home-play" aria-hidden="true">▷</span><div><strong>A Message from Akane</strong><small>A little hello. Coming soon.</small></div><img src={`${projectConfig.basePath}assets/akane-night-thumb.webp`} width="80" height="80" alt="" />
    </aside>
    <div className="home-hero-bottom"><p>Small connections.<br />A brighter tomorrow.</p><span className="home-star-line" aria-hidden="true">✦</span><a href="#differentiators-title" className="home-scroll">Scroll<br />for more <span aria-hidden="true">↓</span></a></div>
    <div className="home-hero-ribbon" aria-hidden="true"><span>Akane</span><span>A more human tomorrow</span><span>People / Ideas / Technology / A kinder tomorrow</span></div>
  </section>;
}

function Differentiators() {
  return <section className="section home-differentiators" aria-labelledby="differentiators-title">
    <div className="shell">
      <div className="section-heading center"><p className="eyebrow">One continuing companion</p><h2 id="differentiators-title">Built to carry meaning forward.</h2></div>
      <div className="home-differentiator-grid">{differentiators.map(item => <article className="surface" key={item.title}><HomeIcon name={item.icon} /><h3>{item.title}</h3><p>{item.text}</p></article>)}</div>
    </div>
  </section>;
}

function DevelopmentVisual() {
  return <section className="section home-development" aria-labelledby="development-title">
    <div className="shell home-development-grid">
      <div className="home-development-copy">
        <p className="eyebrow">Room to develop</p>
        <h2 id="development-title">She begins with room to become.</h2>
        <p>Akane isn’t given a list of things she likes or a personality written in advance. Her preferences, opinions, interests, and goals can develop naturally from what she says and experiences. What lasts comes from real conversations over time.</p>
        <p>What works—and what doesn’t—can shape how she responds in the future.</p>
        <Link className="home-inline-link" to="/technology">See how development works<span aria-hidden="true">→</span></Link>
      </div>
      <figure className="home-development-visual" aria-labelledby="development-caption">
        <figcaption id="development-caption" className="sr-only">What Akane says and experiences in conversation can shape what stays with her over time.</figcaption>
        <img src={characterImage} alt="Akane standing against a soft blue celestial background" loading="lazy" decoding="async" />
        <ol>
          <li><span>01</span><div><strong>What she says</strong><small>A preference or opinion in the moment</small></div></li>
          <li><span>02</span><div><strong>What happens</strong><small>The conversation gives it context</small></div></li>
          <li><span>03</span><div><strong>What can last</strong><small>Meaning that stays with her</small></div></li>
        </ol>
      </figure>
    </div>
  </section>;
}

function InterfaceOverview() {
  return <section className="section home-interfaces" aria-labelledby="interfaces-title">
    <div className="shell">
      <div className="section-heading"><p className="eyebrow">Where Akane appears</p><h2 id="interfaces-title">Different places. The same Akane.</h2><p>Her identity and private owner history stay with the local runtime, whichever interface you use.</p></div>
      <div className="home-interface-grid">{interfaces.map(item => <article key={item.title}><HomeIcon name={item.icon} /><div><h3>{item.title}</h3><p>{item.text}</p></div></article>)}</div>
    </div>
  </section>;
}

function HomeCallToAction() {
  return <section className="home-cta" aria-labelledby="home-cta-title"><div className="shell home-cta-inner"><div><p className="eyebrow">Say hello</p><h2 id="home-cta-title">Meet Akane for yourself.</h2><p>Start a private temporary conversation, or try the simulated preview when she is offline.</p></div><Link className="button primary" to="/demo">Talk to Akane<span aria-hidden="true">→</span></Link></div></section>;
}

export function HomePage() {
  return <main className="home-page"><HomeHero /><Differentiators /><DevelopmentVisual /><InterfaceOverview /><HomeCallToAction /></main>;
}
