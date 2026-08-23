import { useState } from "react";
import { Link } from "react-router-dom";
import { projectConfig } from "../config/project";
import { useInView } from "../lib/useInView";
import "./about.css";

const storyImage = `${projectConfig.basePath}assets/akane-story.jpg`;

const principles = [
  { number: "01", title: "Local at the core", text: "Core inference runs on personal hardware rather than depending on a cloud-first model service.", visual: "local" },
  { number: "02", title: "Continuity with restraint", text: "Akane retains selected meaning—not every line—so memory can help without overwhelming the present.", visual: "continuity" },
  { number: "03", title: "Room to develop", text: "Preferences, memories, interests, and opinions can take shape over time instead of resetting with every conversation.", visual: "growth" },
] as const;

const milestones = [
  { label: "Beginning", title: "A companion with a past", text: "Akane began with a simple question: what changes when an AI conversation can carry meaningful context forward instead of beginning from zero?" },
  { label: "Today", title: "A companion taking shape", text: "Akane can now carry memories, interests, and shared context across conversations while staying grounded in one continuing identity." },
  { label: "Next", title: "A more expressive presence", text: "Voice, expression, and Live2D are the next creative layers—new ways for Akane to feel present without changing who she is underneath." },
] as const;

export function AboutPage() {
  const [activePrinciple, setActivePrinciple] = useState(0);
  const { ref: timelineRef, visible: timelineVisible } = useInView<HTMLOListElement>(0.25);
  const selectedPrinciple = principles[activePrinciple];

  return <main className="about-page">
    <section className="about-hero" aria-labelledby="about-title">
      <div className="shell about-hero-inner">
        <p className="eyebrow">About the project</p>
        <h1 id="about-title">Why I built Akane</h1>
        <p className="about-lead">Akane is an experiment in what happens when an AI companion can remember, develop, and continue instead of starting over every time.</p>
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
          <p>Choosing to run Akane locally made the idea feel more personal: one companion on familiar hardware, shaped gradually through real conversations rather than a collection of disconnected sessions.</p>
        </div>
      </div>
      <figure className="shell about-story-image">
        <img src={storyImage} alt="Akane sitting in a softly lit room beneath a starry night sky" loading="lazy" decoding="async" />
        <figcaption>A companion imagined as part of a place—not only a window waiting for a prompt.</figcaption>
      </figure>
    </section>

    <section className="about-principles section-pad" aria-labelledby="about-principles-title">
      <div className="shell">
        <div className="section-heading">
          <p className="eyebrow">Guiding ideas</p>
          <h2 id="about-principles-title">Personal by design</h2>
        </div>
        <div className="about-principle-experience">
          <div className="about-principle-tabs" role="tablist" aria-label="Guiding ideas">
            {principles.map((principle, index) => <button key={principle.number} type="button" role="tab" aria-selected={activePrinciple === index} aria-controls="about-principle-panel" onClick={() => setActivePrinciple(index)}><span>{principle.number}</span>{principle.title}</button>)}
          </div>
          <article id="about-principle-panel" className="about-principle-panel" role="tabpanel" key={selectedPrinciple.number}>
            <div><span>{selectedPrinciple.number}</span><h3>{selectedPrinciple.title}</h3><p>{selectedPrinciple.text}</p></div>
            <div className={`about-principle-visual ${selectedPrinciple.visual}`} aria-hidden="true">
              <i /><i /><i />
            </div>
          </article>
        </div>
      </div>
    </section>

    <section className="about-evolution section-pad" aria-labelledby="evolution-title">
      <div className="shell about-evolution-grid">
        <div className="about-evolution-heading">
          <p className="eyebrow">Evolution</p>
          <h2 id="evolution-title">How Akane has grown</h2>
          <p>Each stage has brought the original idea closer to a companion with a lasting sense of continuity.</p>
        </div>
        <ol ref={timelineRef} className={`about-timeline ${timelineVisible ? "is-visible" : ""}`}>
          {milestones.map(item => <li key={item.label}><span>{item.label}</span><div><h3>{item.title}</h3><p>{item.text}</p></div></li>)}
        </ol>
      </div>
    </section>

    <section className="about-creator section-pad" aria-labelledby="creator-title">
      <div className="shell about-creator-inner">
        <div><p className="eyebrow">Created by Alexander Lee</p><h2 id="creator-title">A personal project, still growing.</h2></div>
        <div><p>I built Akane to explore the space between a useful AI tool and a character with a lasting presence. The project continues to grow through conversation, experimentation, and the small details that make her feel consistent.</p><div className="actions"><Link className="button primary" to="/demo">Meet Akane</Link><a className="button secondary" href={projectConfig.githubUrl} target="_blank" rel="noreferrer">View on GitHub<span aria-hidden="true">↗</span></a></div></div>
      </div>
    </section>
  </main>;
}
