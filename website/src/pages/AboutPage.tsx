import { Link } from "react-router-dom";
import { projectConfig } from "../config/project";
import "./about.css";

const storyImage = `${projectConfig.basePath}assets/akane-story.jpg`;

const evolution = [
  { label: "Beginning", title: "A companion with a past", text: "Akane began as a question about what an AI relationship could feel like when meaningful context did not disappear after each conversation." },
  { label: "Foundation", title: "One local, continuing identity", text: "The project grew around local inference, selective memory, and several ways to talk with the same companion." },
  { label: "Today", title: "Learning from what happens", text: "Akane’s preferences and behavior can change through conversation and outcomes without inventing things that happened offscreen." },
] as const;

export function AboutPage() {
  return <main className="about-page">
    <section className="page-hero about-hero" aria-labelledby="about-title">
      <div className="shell"><p className="eyebrow">About the project</p><h1 id="about-title">Why I built Akane</h1><p className="page-lead">A personal project about local AI, lasting conversations, and how one companion can change over time.</p></div>
    </section>

    <section className="section about-origin" aria-labelledby="origin-title">
      <div className="shell about-origin-grid">
        <div className="about-origin-copy"><p className="eyebrow">Why I started Akane</p><h2 id="origin-title">I wanted a conversation with history.</h2><p>Most AI conversations can be useful in the moment while still feeling disconnected from everything that came before. I started Akane to explore one companion who could remember what came before and change over time.</p><p>I did not want another blank chat window with a personality pasted on top. I wanted the character, the memory, and the development to belong to the same continuing Akane.</p></div>
        <figure className="about-story-image"><img src={storyImage} alt="Akane sitting in a softly lit room beneath a starry night sky" loading="lazy" decoding="async" /><figcaption>Akane began as a character I wanted to meet more than once.</figcaption></figure>
      </div>
    </section>

    <section className="section about-decisions" aria-label="Project decisions">
      <div className="shell about-decision-grid">
        <article className="surface"><p className="eyebrow">What I wanted to do differently</p><h2>Let development be earned.</h2><p>Akane starts with only a small fixed identity. Her preferences, opinions, interests, and goals should come from conversation—not a hidden list of things she is supposed to like.</p></article>
        <article className="surface"><p className="eyebrow">Why local inference</p><h2>Keep the core close.</h2><p>Running Akane on personal hardware makes the project tangible and gives me direct ownership of the model, state, privacy boundaries, and engineering tradeoffs. The constraint is part of the point.</p></article>
      </div>
    </section>

    <section className="section about-evolution" aria-labelledby="evolution-title">
      <div className="shell about-evolution-grid">
        <div className="section-heading"><p className="eyebrow">How Akane evolved</p><h2 id="evolution-title">The architecture followed the idea.</h2><p>Each change has moved the project closer to learning from real conversations instead of inventing a life offscreen.</p></div>
        <ol className="about-timeline">{evolution.map(item => <li key={item.label}><span>{item.label}</span><div><h3>{item.title}</h3><p>{item.text}</p></div></li>)}</ol>
      </div>
    </section>

    <section className="section about-next" aria-labelledby="next-title">
      <div className="shell about-next-grid">
        <div><p className="eyebrow">What comes next</p><h2 id="next-title">More ways for Akane to be present.</h2></div>
        <div><p>The next focus is voice and expression, followed by Live2D or 3D embodiment. Those layers should make Akane feel more present without changing who she is underneath.</p><Link className="about-text-link" to="/technology">See the current roadmap<span aria-hidden="true">→</span></Link></div>
      </div>
    </section>

    <section className="about-creator" aria-labelledby="creator-title">
      <div className="shell about-creator-inner"><div><p className="eyebrow">About Alexander</p><h2 id="creator-title">Built by Alexander Lee.</h2><p>Akane is a personal project with a public GitHub repository. It continues to grow through careful experiments in character, local AI, and lasting behavior.</p></div><div className="actions"><Link className="button primary" to="/demo">Meet Akane</Link><a className="button secondary" href={projectConfig.githubUrl} target="_blank" rel="noreferrer">View on GitHub<span aria-hidden="true">↗</span></a></div></div>
    </section>
  </main>;
}
