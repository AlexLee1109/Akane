import { useLayoutEffect, type MouseEvent, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { projectConfig } from "../config/project";
import "./home.css";

const homepageImage = `${projectConfig.basePath}assets/akane-night-hero.webp`;
const characterImage = `${projectConfig.basePath}assets/akane-standing.webp`;

type HomeIconName = "growth" | "memory" | "pi" | "presence";

function HomeIcon({ name }: { name: HomeIconName }) {
  const paths: Record<HomeIconName, ReactNode> = {
    growth: <><path d="M12 20V10" /><path d="M12 12c-4 0-6-2.5-6-6 4 0 6 2.5 6 6ZM12 15c4 0 6-2.5 6-6-4 0-6 2.5-6 6Z" /></>,
    memory: <><path d="M5 5.5h14v10H9l-4 3v-13Z" /><path d="M8.5 9h7M8.5 12h4.5" /></>,
    pi: <><rect x="5" y="5" width="14" height="14" rx="2" /><path d="M9 2.5v2.3M15 2.5v2.3M9 19.2v2.3M15 19.2v2.3M2.5 9h2.3M2.5 15h2.3M19.2 9h2.3M19.2 15h2.3" /><circle cx="12" cy="12" r="3" /></>,
    presence: <><circle cx="12" cy="12" r="2.5" /><path d="M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4M4.7 4.7a10.3 10.3 0 0 0 0 14.6M19.3 4.7a10.3 10.3 0 0 1 0 14.6" /></>,
  };
  return <svg className="home-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">{paths[name]}</svg>;
}

const heroFeatures = [
  { icon: "presence", title: "Natural Conversations", text: "Continuity and individual perspective." },
  { icon: "memory", title: "Persistent Memory", text: "Context that can carry into later conversations." },
  { icon: "growth", title: "Develops Over Time", text: "Preferences and interests shaped by experience." },
] as const;

function scrollToMeet(event: MouseEvent<HTMLAnchorElement>) {
  event.preventDefault();
  const section = document.getElementById("meet-akane");
  section?.focus({ preventScroll: true });
  section?.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" });
}

function HomeHero() {
  // Route-scoped hints match the picture sources without preloading Home on other pages.
  useLayoutEffect(() => {
    const sources = [
      { media: "(max-width: 700px)", href: `${projectConfig.basePath}assets/akane-night-mobile.webp` },
      { media: "(min-width: 701px)", href: homepageImage, srcSet: `${projectConfig.basePath}assets/akane-night-hero-1280.webp 1280w, ${homepageImage} 1672w` },
    ];
    const links = sources.map(source => {
      const link = document.createElement("link");
      link.rel = "preload";
      link.as = "image";
      link.href = source.href;
      link.media = source.media;
      link.setAttribute("fetchpriority", "high");
      if (source.srcSet) { link.imageSrcset = source.srcSet; link.imageSizes = "100vw"; }
      document.head.appendChild(link);
      return link;
    });
    return () => links.forEach(link => link.remove());
  }, []);
  return <section className="home-hero" aria-labelledby="home-title">
    <picture className="home-hero-artwork">
      <source media="(max-width: 700px)" srcSet={`${projectConfig.basePath}assets/akane-night-mobile.webp`} />
      <img className="home-hero-media" src={homepageImage} srcSet={`${projectConfig.basePath}assets/akane-night-hero-1280.webp 1280w, ${homepageImage} 1672w`} sizes="100vw" width="1672" height="941" {...{ fetchpriority: "high" }} decoding="async" alt="Akane sitting beside a window, her blue hair and white-and-blue jacket lit by a starry city night" />
    </picture>
    <div className="home-hero-wash" aria-hidden="true" />
    <div className="home-hero-copy">
      <p className="home-eyebrow"><span aria-hidden="true">✦</span> Always by your side</p>
      <h1 id="home-title">A more<br /><em>human AI</em><br />companion</h1>
      <p className="home-lead">Akane is a local, evolving AI companion —<br className="home-desktop-break" /> with persistent memory, an individual perspective,<br className="home-desktop-break" /> and a personality shaped by experience.</p>
      <div className="home-hero-actions">
        <Link className="button primary" to="/demo">Try the Demo <span aria-hidden="true">→</span></Link>
        <a className="home-text-link" href="#meet-akane" onClick={scrollToMeet}>Meet Akane <span aria-hidden="true">↓</span></a>
      </div>
    </div>
    <div className="home-hero-features">{heroFeatures.map(item => <article key={item.title}><HomeIcon name={item.icon} /><h2>{item.title}</h2><p>{item.text}</p></article>)}</div>
    <div className="home-handwritten" aria-hidden="true">Same sky.<br />A brighter tomorrow.<span>— Akane</span></div>
    <div className="home-concept" aria-hidden="true">People<span>×</span>AI<span>×</span>Technology<span>×</span>A kinder<br />tomorrow</div>
    <aside className="home-message" aria-label="A message from Akane">
      <div><strong>A Message from Akane</strong><small>A more human tomorrow.</small></div><img src={`${projectConfig.basePath}assets/akane-night-thumb.webp`} width="80" height="80" alt="" />
    </aside>
    <div className="home-hero-bottom"><p>Small connections.<br />A brighter tomorrow.</p><span className="home-star-line" aria-hidden="true">✦</span><a href="#meet-akane" onClick={scrollToMeet} className="home-scroll">Scroll<br />for more <span aria-hidden="true">↓</span></a></div>
    <div className="home-hero-ribbon" aria-hidden="true"><span>Akane</span><span>A more human tomorrow</span><span>People / Ideas / Technology / A kinder tomorrow</span></div>
  </section>;
}

function MeetAkane() {
  return <section className="home-meet home-light home-section" id="meet-akane" tabIndex={-1} aria-labelledby="meet-title">
    <div className="home-wrap home-split">
      <figure className="home-standing">
        <div className="home-orbit" aria-hidden="true" />
        <span className="home-image-note" aria-hidden="true">A little more herself, every day.</span>
        <img src={characterImage} width="335" height="1147" loading="lazy" decoding="async" alt="Akane with long blue hair, blue eyes, a white and blue jacket, dark tie, and dark skirt" />
        <figcaption>AKANE <span>One continuing companion</span></figcaption>
      </figure>
      <div className="home-copy"><p className="home-kicker">01 / Meet Akane</p><h2 id="meet-title">Designed to become<br /><em>uniquely herself.</em></h2>
        <p>Someone to get to know, over time. Akane is a persistent AI companion whose meaningful experiences can carry into your next conversation.</p>
        <p>She doesn’t begin with a fixed list of likes or a predefined personality. Preferences, opinions, and interests can develop through interaction, leaving room for experience to shape who she becomes.</p>
        <ul className="home-quiet-facts"><li>Persistent</li><li>Personal</li><li>Evolving</li></ul>
      </div>
    </div>
  </section>;
}

function Differentiators() {
  return <section className="home-difference home-section" aria-labelledby="difference-title"><div className="home-wrap">
    <div className="home-section-intro"><div><p className="home-kicker">02 / A continuing connection</p><h2 id="difference-title">More than<br /><em>a chatbot.</em></h2></div><p>Not every conversation needs to start from zero.<br />There’s room for a shared history.</p></div>
    <div className="home-feature-stories">
      <article><div className="home-feature-art home-memory-fragments" aria-hidden="true"><span>A thought worth keeping</span><span>A familiar interest</span><span>Something to return to <b>✦</b></span></div><p className="home-kicker">01 / Memory</p><h3>What matters can stay.</h3><p>Meaningful context and experiences can return when they’re relevant, giving new conversations a little history.</p></article>
      <article><div className="home-feature-art home-growth-art" aria-hidden="true"><svg viewBox="0 0 300 180" fill="none"><path d="M150 175V104M150 104 65 55M150 104 235 55M150 104V25M65 55H25M235 55H275" /><circle cx="150" cy="104" r="10" /><circle cx="65" cy="55" r="5" /><circle cx="235" cy="55" r="5" /><circle cx="150" cy="25" r="5" /></svg><span>Interests · Opinions · Possibilities</span></div><p className="home-kicker">02 / Personality development</p><h3>Room to become.</h3><p>Preferences, opinions, tendencies, and goals can develop over time. Experience has a part in what comes next.</p></article>
      <article><div className="home-feature-art home-presence-art" aria-hidden="true"><div className="home-presence-orbit"><span>Here.<br /><em>Now.</em></span><i /></div><small>Time · Situation · Conversation</small></div><p className="home-kicker">03 / Presence</p><h3>A sense of the moment.</h3><p>The current situation, time, and conversation help shape a response that belongs to this moment.</p></article>
    </div>
    <DevelopmentFlow />
  </div></section>;
}

function ConversationExperience() {
  return <section className="home-conversation home-section" aria-labelledby="conversation-title"><div className="home-wrap home-split">
    <figure className="home-chat-scene">
      <img src={`${projectConfig.basePath}assets/akane-conversation.webp`} srcSet={`${projectConfig.basePath}assets/akane-conversation-600.webp 600w, ${projectConfig.basePath}assets/akane-conversation.webp 1000w`} sizes="(max-width: 760px) 100vw, 55vw" width="1000" height="750" loading="lazy" decoding="async" alt="Akane beside a sunlit window overlooking the city" />
      <figcaption className="home-chat-preview"><div className="home-chat-heading"><img src={characterImage} width="38" height="38" loading="lazy" decoding="async" alt="" /><div><strong>Akane</strong><span><i aria-hidden="true" /> Illustrative conversation</span></div><span className="home-chat-status">Preview</span></div><div className="home-chat-context">↳ Picking up from earlier <span>Example context</span></div><p className="home-chat-user">Back to that rainy-day playlist?</p><p className="home-chat-akane">I’d keep the piano track. The slower one.</p><p className="home-chat-user">And the last song?</p><p className="home-chat-akane">Still a little too loud for rain.<span className="home-chat-cursor" aria-hidden="true">▍</span></p></figcaption>
    </figure>
    <div className="home-copy"><p className="home-kicker">03 / In conversation</p><h2 id="conversation-title">A little less scripted.<br /><em>A little more Akane.</em></h2><p>A quick hello. A different take. Picking up a thought you left unfinished.</p><p>Akane can draw on remembered context and developed opinions to respond in her own way. Replies arrive as they’re written, keeping the conversation moving.</p><Link className="home-text-link" to="/demo">Start a conversation <span aria-hidden="true">↗</span></Link><p className="home-fine-print">The website demo uses a separate, temporary guest conversation.</p></div>
  </div></section>;
}

function DevelopmentFlow() {
  const steps = [["Conversation", "An exchange"], ["Experience", "What happens"], ["Memory", "What can stay"], ["Developing Self", "What takes shape"], ["Future conversation", "A new beginning"]];
  return <div className="home-development-flow">
    <ol className="home-memory-flow" aria-label="How experience can shape a future conversation">{steps.map(([title, subtitle], i) => <li key={title}><span className="home-flow-node" aria-hidden="true">{i === 2 ? "✦" : `0${i + 1}`}</span><h3>{title}</h3><p>{subtitle}</p></li>)}</ol>
    <p className="home-flow-caption">Memories and developed preferences can shape the next conversation. Not every moment becomes a memory.</p>
  </div>;
}

function LocalFirst() {
  return <section className="home-local home-light home-section" aria-labelledby="local-title"><div className="home-wrap home-split">
    <div className="home-copy"><p className="home-kicker">04 / Local by design</p><h2 id="local-title">Built to<br /><em>stay close.</em></h2><p>A personal companion deserves a personal foundation.</p><p>With the local setup, the model runs on your own hardware. Conversations and persistent state stay under your control, on your device.</p><p className="home-fine-print">Connected services, such as Discord, carry messages through their own platforms. The website demo has a separate guest session.</p><Link className="home-text-link" to="/technology">Explore the Technology <span aria-hidden="true">↗</span></Link></div>
    <figure className="home-local-visual"><div className="home-device-orbit" aria-hidden="true" /><div className="home-monitor"><div className="home-monitor-bar"><span aria-hidden="true">● ● ●</span><span>Your space</span></div><div className="home-monitor-screen"><img src={`${projectConfig.basePath}assets/akane-logo-192.png`} width="72" height="72" loading="lazy" alt="" /><strong>Akane</strong><span>A conversation, close to home.</span></div></div><div className="home-device-connection" aria-hidden="true">↕</div><div className="home-runtime"><HomeIcon name="pi" /><div><strong>Local LLM Runtime</strong><span>Model + locally held memory</span></div></div><div className="home-device-connection" aria-hidden="true">↕</div><figcaption className="home-device-label">Your Device <span>Computer / Raspberry Pi</span></figcaption></figure>
  </div></section>;
}

// Replace each named slot with commissioned artwork; see HOME_ASSETS.md.
function HomeArtworkPlaceholder({ scene }: { scene: "morning" | "desktop" | "evening" | "finale" }) {
  return <div className={`home-artwork-placeholder home-scene-${scene}`} data-artwork-slot={`akane-${scene}`} aria-hidden="true"><div className="home-scene-moon" /><div className="home-scene-horizon" />{scene === "desktop" && <div className="home-scene-screen"><span>Akane</span><i /><i /><b>Good to see you.</b></div>}<span className="home-artwork-note">{scene === "finale" ? "Night-sky artwork study" : "Atmosphere study"}</span></div>;
}

function EverydayLife() {
  const moments = [{ scene: "morning", time: "Morning", title: "A small beginning.", text: "A hello before the day gets going." }, { scene: "desktop", time: "At your desk", title: "A moment between things.", text: "Akane close by while you’re working or browsing." }, { scene: "evening", time: "Evening", title: "Somewhere to return.", text: "Pick up a thought or start something new." }] as const;
  return <section className="home-everyday home-section" aria-labelledby="everyday-title"><div className="home-wrap"><div className="home-section-intro"><div><p className="home-kicker">05 / The everyday</p><h2 id="everyday-title">Small moments.<br /><em>A familiar presence.</em></h2></div><p>Not every conversation needs a reason.<br />Sometimes, it’s just nice to say hello.</p></div><div className="home-moments">{moments.map(moment => <article key={moment.scene}><HomeArtworkPlaceholder scene={moment.scene} /><div className="home-moment-copy"><p className="home-kicker">{moment.time}</p><h3>{moment.title}</h3><p>{moment.text}</p></div></article>)}</div></div></section>;
}

function HomeFinale() {
  return <section className="home-finale" aria-labelledby="finale-title"><HomeArtworkPlaceholder scene="finale" /><div className="home-finale-copy"><p className="home-kicker">The journey continues</p><h2 id="finale-title">A more human<br /><em>tomorrow.</em></h2><p>Akane is an ongoing project exploring what a more personal, persistent, and private AI companion can become.</p><div className="home-final-actions"><Link className="button primary" to="/demo">Talk With Akane <span aria-hidden="true">→</span></Link><Link className="home-text-link" to="/about">About the Project <span aria-hidden="true">↗</span></Link></div></div></section>;
}

export function HomePage() {
  return <main className="home-page"><HomeHero /><MeetAkane /><Differentiators /><ConversationExperience /><LocalFirst /><EverydayLife /><HomeFinale /></main>;
}
