import { Link } from "react-router-dom";
import { projectConfig } from "../config/project";
import "./technology.css";

const responseSteps = [
  { title: "Interface", text: "Desktop, Discord, or an isolated website guest sends a message." },
  { title: "Relevant context", text: "Only useful recent and persistent state is selected for this turn." },
  { title: "One Qwen generation", text: "One local model call produces the complete dialogue result." },
  { title: "Reply + evidence", text: "The stream separates Akane’s visible reply from hidden semantic evidence." },
  { title: "Validated persistence", text: "Grounded changes and the completed turn are written in one atomic commit." },
] as const;

const runtimeFacts = [
  ["Hardware", "Raspberry Pi 5"],
  ["Model", projectConfig.modelName],
  ["Inference", "llama.cpp / llama-cpp-python"],
  ["Server", "FastAPI"],
  ["State", "Atomic JSON persistence"],
  ["Delivery", "Streamed responses"],
] as const;

const interfaces = [
  { title: "Desktop popup", text: "A thin private owner interface for keeping Akane nearby." },
  { title: "Discord", text: "The same owner profile and continuing companion in Discord." },
  { title: "Website guest", text: "A temporary isolated profile that never accesses private owner memory." },
] as const;

const roadmap = [
  { label: "Now", title: "Developmental Self + consequence learning", text: "Grounded experience can shape durable judgments and future behavior." },
  { label: "Next", title: "Voice + expression", text: "A more expressive presentation for the same continuing Akane." },
  { label: "Then", title: "Live2D / 3D embodiment", text: "A richer visual presence without moving identity into the renderer." },
  { label: "Later", title: "Deeper world modeling", text: "More capable learned adaptation, developed carefully from grounded evidence." },
] as const;

function SectionHeading({ eyebrow, title, description, id }: { eyebrow: string; title: string; description?: string; id: string }) {
  return <div className="section-heading"><p className="eyebrow">{eyebrow}</p><h2 id={id}>{title}</h2>{description && <p>{description}</p>}</div>;
}

function TechnologyHero() {
  return <section className="page-hero tech-hero" aria-labelledby="technology-title">
    <div className="shell"><p className="eyebrow">Technology</p><h1 id="technology-title">How Akane works</h1><p className="page-lead">One local generation carries the dialogue path. Grounded evidence from that same turn can shape a persistent, developing Self.</p></div>
  </section>;
}

function ResponseFlow() {
  return <section className="section tech-response" aria-labelledby="response-title">
    <div className="shell">
      <SectionHeading eyebrow="How Akane responds" title="One generation, one completed turn." description="The critical path stays small enough for local hardware and clear enough to reason about." id="response-title" />
      <figure className="tech-response-flow" aria-labelledby="response-caption">
        <figcaption id="response-caption" className="sr-only">An interface selects relevant context for one Qwen generation, which produces a reply and semantic evidence before validated persistence.</figcaption>
        <ol>{responseSteps.map((step, index) => <li key={step.title}><span>0{index + 1}</span><h3>{step.title}</h3><p>{step.text}</p></li>)}</ol>
      </figure>
    </div>
  </section>;
}

function DevelopmentFlow() {
  return <section className="section tech-development" aria-labelledby="development-title">
    <div className="shell tech-development-grid">
      <div>
        <SectionHeading eyebrow="How Akane develops" title="A Self built from evidence, not seed interests." description="Akane begins with a minimal fixed identity. Preferences and opinions are not filled in ahead of time; they can form from judgments she actually makes and experiences grounded in conversation." id="development-title" />
        <p className="tech-development-note">Outcomes can resolve predictions and shape behavioral tendencies or strategies. Unresolved questions and recurring evidence can also support curiosity and developmental goals.</p>
      </div>
      <figure className="tech-development-map" aria-labelledby="development-map-caption">
        <figcaption id="development-map-caption" className="sr-only">Conversation enters Qwen, which produces a spoken response and hidden semantic evidence. Evidence can form an Experience, Developed Self, Outcomes, and learned behavior.</figcaption>
        <div className="tech-map-node input"><small>Present moment</small><strong>Conversation</strong></div>
        <span className="tech-map-arrow" aria-hidden="true">↓</span>
        <div className="tech-map-node model"><small>One local generation</small><strong>Qwen</strong></div>
        <span className="tech-map-arrow" aria-hidden="true">↓</span>
        <div className="tech-map-branches">
          <article><small>Visible</small><strong>Spoken response</strong><p>What Akane says to you.</p></article>
          <article className="evidence"><small>Hidden</small><strong>Semantic evidence</strong><p>Structured meaning from the same generation.</p><span aria-hidden="true">↓</span><b>Grounded Experience</b><span aria-hidden="true">↓</span><div><b>Developed Self</b><b>Outcomes</b></div></article>
        </div>
        <div className="tech-learning-row"><span>Predictions</span><span>Behavioral tendencies</span><span>Strategies</span><span>Curiosity</span><span>Developmental goals</span></div>
      </figure>
    </div>
  </section>;
}

function Runtime() {
  return <section className="section tech-runtime" aria-labelledby="runtime-title">
    <div className="shell">
      <SectionHeading eyebrow="How Akane runs" title="A compact local stack." description="The website is a static interface. Akane’s model, state, and developmental path stay with the local backend." id="runtime-title" />
      <dl className="tech-runtime-grid">{runtimeFacts.map(([term, detail]) => <div className="surface" key={term}><dt>{term}</dt><dd>{detail}</dd></div>)}</dl>
    </div>
  </section>;
}

function Interfaces() {
  return <section className="section tech-interfaces" aria-labelledby="tech-interfaces-title">
    <div className="shell tech-interfaces-grid">
      <SectionHeading eyebrow="Where Akane appears" title="Thin interfaces around one companion." description="Presentation changes by surface; identity and state ownership do not." id="tech-interfaces-title" />
      <div>{interfaces.map(item => <article key={item.title}><h3>{item.title}</h3><p>{item.text}</p></article>)}</div>
    </div>
  </section>;
}

function Roadmap() {
  return <section className="section tech-roadmap" aria-labelledby="roadmap-title">
    <div className="shell">
      <div className="tech-roadmap-heading"><SectionHeading eyebrow="Roadmap" title="Build outward from the same Akane." id="roadmap-title" /><Link className="button secondary" to="/demo">Talk to Akane<span aria-hidden="true">→</span></Link></div>
      <ol>{roadmap.map(item => <li key={item.label}><span>{item.label}</span><div><h3>{item.title}</h3><p>{item.text}</p></div></li>)}</ol>
    </div>
  </section>;
}

export function TechnologyPage() {
  return <main className="technology-page"><TechnologyHero /><ResponseFlow /><DevelopmentFlow /><Runtime /><Interfaces /><Roadmap /></main>;
}
