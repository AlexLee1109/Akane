import { Link } from "react-router-dom";
import { projectConfig } from "../config/project";
import "./about.css";

const milestones = [
  { label: "Beginning", title: "A companion with a past", text: "Akane began with a simple question: what changes when an AI conversation can carry meaningful context forward instead of beginning from zero?" },
  { label: "Today", title: "One working local system", text: "Conversation, persistent Self and Memory, background Reflection, a desktop popup, Discord, and an isolated web demo now share one coordinated runtime." },
  { label: "Next", title: "A more expressive presence", text: "Voice, expression, and Live2D are the next presentation layers. The aim is to make Akane feel more present without changing who she is underneath." },
] as const;

export function AboutPage() {
  return <main className="about-page">
    <section className="about-hero" aria-labelledby="about-title">
      <div className="shell about-hero-inner">
        <p className="eyebrow">Project story</p>
        <h1 id="about-title">Building a companion<br />that can continue.</h1>
        <p className="about-lead">Akane is a personal AI companion project built around local inference, durable continuity, and the feeling that a conversation can matter tomorrow.</p>
      </div>
    </section>

    <section className="about-story section-pad" aria-labelledby="why-akane-title">
      <div className="shell about-story-grid">
        <div>
          <p className="eyebrow">Why Akane exists</p>
          <h2 id="why-akane-title">More than a blank chat window</h2>
        </div>
        <div className="about-prose">
          <p>Most AI conversations are useful in the moment, but feel disconnected from everything that came before. Akane explores a more personal direction: one companion with a consistent identity, selective memory, and a small life between conversations.</p>
          <p>Running the core model locally on a Raspberry Pi makes that experiment tangible. The constraints are real—limited compute, a finite context window, and one shared runtime—but they encourage careful decisions about what deserves to persist and what the model actually needs now.</p>
        </div>
      </div>
    </section>

    <section className="about-principles section-pad" aria-labelledby="about-principles-title">
      <div className="shell">
        <div className="section-heading">
          <p className="eyebrow">Guiding ideas</p>
          <h2 id="about-principles-title">Personal by design</h2>
        </div>
        <div className="about-principle-grid">
          <article><span>01</span><h3>Local at the core</h3><p>Core inference runs on personal hardware rather than depending on a cloud-first model service.</p></article>
          <article><span>02</span><h3>Continuity with restraint</h3><p>Akane retains selected meaning—not every line—so memory can help without overwhelming the present.</p></article>
          <article><span>03</span><h3>One companion</h3><p>Desktop and Discord share owner continuity; temporary web guests stay completely isolated.</p></article>
        </div>
      </div>
    </section>

    <section className="about-evolution section-pad" aria-labelledby="evolution-title">
      <div className="shell about-evolution-grid">
        <div className="about-evolution-heading">
          <p className="eyebrow">Evolution</p>
          <h2 id="evolution-title">A system growing in layers</h2>
          <p>Each stage keeps the companion model and continuity underneath the interface.</p>
        </div>
        <ol className="about-timeline">
          {milestones.map(item => <li key={item.label}><span>{item.label}</span><div><h3>{item.title}</h3><p>{item.text}</p></div></li>)}
        </ol>
      </div>
    </section>

    <section className="about-creator section-pad" aria-labelledby="creator-title">
      <div className="shell about-creator-inner">
        <div><p className="eyebrow">Created by Alexander Lee</p><h2 id="creator-title">An open engineering project with a personal center.</h2></div>
        <div><p>Akane brings together local model serving, state design, interface work, and character-driven product thinking. The repository documents what is working today and the tradeoffs behind it.</p><div className="actions"><Link className="button primary" to="/demo">Meet Akane</Link><a className="button secondary" href={projectConfig.githubUrl} target="_blank" rel="noreferrer">View on GitHub<span aria-hidden="true">↗</span></a></div></div>
      </div>
    </section>
  </main>;
}
