import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { projectConfig } from "../config/project";
import "./technology.css";

type IconName = "arrow" | "brain" | "browser" | "model" | "profile" | "shield" | "stream";
type RoadmapStatus = "In development" | "Planned";

interface ArchitectureStage {
  icon: IconName;
  title: string;
  description: string;
}

interface StackGroup {
  category: string;
  title: string;
  items: readonly string[];
}

interface Tradeoff {
  title: string;
  constraint: string;
  response: string;
  importance: string;
}

const architectureStages: readonly ArchitectureStage[] = [
  {
    icon: "browser",
    title: "Receive and resolve",
    description: "The active interface sends the request and the backend resolves the owner or isolated temporary guest profile.",
  },
  {
    icon: "profile",
    title: "Select context",
    description: "Recent conversation and only the relevant Self, Memory, time, InnerLife, and interface context are selected.",
  },
  {
    icon: "brain",
    title: "Compile the prompt",
    description: "Identity, behavioral rules, current input, and selected context are budgeted into one model-ready prompt.",
  },
  {
    icon: "model",
    title: "Run local inference",
    description: `The shared coordinator reserves the ${projectConfig.modelName} runtime through llama.cpp on the Raspberry Pi.`,
  },
  {
    icon: "stream",
    title: "Deliver and commit",
    description: "Each interface receives either a live stream or one completed reply. State is committed only after successful generation.",
  },
];

const implementationGroups = [
  {
    title: "Context and identity",
    items: [
      ["Character", "Owns Akane’s compact identity, temperament, appearance, and seed interests."],
      ["ContextBuilder", "Selects one bounded snapshot of relevant Self, Memory, InnerLife, time, and connected interface evidence."],
      ["PromptPlan", "Compiles identity, current input, raw recent dialogue, and selected context for the small local model."],
    ],
  },
  {
    title: "Continuity and state",
    items: [
      ["Store", "Owns one schema-versioned JSON document and every canonical state transaction."],
      ["StateChangeProposal", "Carries validated conversation, Self, Memory, InnerLife, and Reflection-range changes."],
      ["ReflectionEngine", "Separately extracts small post-turn changes; it never writes state directly."],
      ["InnerLife", "Maintains lightweight current activity, previous activity, and optional focus between conversations."],
    ],
  },
  {
    title: "Runtime coordination",
    items: [
      ["GenerationScheduler", "Bounds foreground work and prevents one profile from racing itself."],
      ["InferenceRuntime", "Owns the singleton llama.cpp runtime, priority reservations, token accounting, and token streaming."],
      ["AutonomyCoordinator", "Runs reflection before low-duty-cycle InnerLife without competing with visible conversation."],
    ],
  },
] as const;

const stackGroups: readonly StackGroup[] = [
  {
    category: "Model",
    title: projectConfig.modelName,
    items: ["Instruction-tuned local generation", "llama.cpp via llama-cpp-python", "Q4_K_M GGUF", "4,096-token configured default"],
  },
  {
    category: "Backend",
    title: "Shared Python runtime",
    items: ["Python", "FastAPI", "Streamed HTTP transport"],
  },
  {
    category: "Interfaces",
    title: "One companion, three surfaces",
    items: ["Desktop popup", "Discord", "React and TypeScript website"],
  },
  {
    category: "Persistence",
    title: "Validated local state",
    items: ["One schema-versioned JSON file", "Atomic state replacement", "Conversation and reflection jobs"],
  },
  {
    category: "Developer integration",
    title: "Bounded editor context",
    items: ["VS Code extension", "Explicitly requested context", "Read-only workspace boundary"],
  },
  {
    category: "Deployment",
    title: "Pi runtime, static website",
    items: ["Raspberry Pi 5", "ngrok HTTPS endpoint", "GitHub Pages"],
  },
];

const tradeoffs: readonly Tradeoff[] = [
  {
    title: "One shared runtime",
    constraint: "Foreground and background work compete for one local model.",
    response: "A shared scheduler and reservation coordinate work by priority.",
    importance: "Conversation stays responsive without a duplicate Pi runtime.",
  },
  {
    title: "Context must earn its tokens",
    constraint: "The 4,096-token window cannot hold every stored detail.",
    response: "The prompt protects identity and current input, then selects relevant context.",
    importance: "Continuity helps without crowding out the current conversation.",
  },
  {
    title: "State must survive interruption",
    constraint: "A streamed reply can stop or disconnect before completion.",
    response: "State commits only after successful generation and validation.",
    importance: "Partial replies do not become permanent history or memory.",
  },
  {
    title: "Interfaces must remain one companion",
    constraint: "Each interface needs different behavior and privacy boundaries.",
    response: "Thin adapters share backend ownership while profile resolution isolates guests.",
    importance: "Akane stays consistent without exposing private continuity.",
  },
];

const roadmapItems: readonly { title: string; status: RoadmapStatus }[] = [
  { title: "Expression rendering", status: "In development" },
  { title: "Voice and lip synchronization", status: "Planned" },
  { title: "Live2D presentation", status: "Planned" },
];

function TechIcon({ name }: { name: IconName }) {
  const paths: Record<IconName, ReactNode> = {
    arrow: <><path d="M5 12h13" /><path d="m14 8 4 4-4 4" /></>,
    brain: <><path d="M9.5 4.5A3.5 3.5 0 0 0 6 8v.3A3.2 3.2 0 0 0 4 11.2 3.1 3.1 0 0 0 6.2 14 3.5 3.5 0 0 0 9.5 19.5" /><path d="M14.5 4.5A3.5 3.5 0 0 1 18 8v.3a3.2 3.2 0 0 1 2 2.9 3.1 3.1 0 0 1-2.2 2.8 3.5 3.5 0 0 1-3.3 5.5M12 4v16M8 9.5h4M12 14.5h4" /></>,
    browser: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 8h18M7 6h.01M10 6h.01" /></>,
    model: <><rect x="5" y="5" width="14" height="14" rx="2" /><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" /><path d="m10 14 2-5 2 5" /></>,
    profile: <><circle cx="9" cy="8" r="3" /><path d="M3.5 19a5.5 5.5 0 0 1 11 0M16 9h5M18.5 6.5v5" /></>,
    shield: <><path d="M12 3 19 6v5c0 4.8-3 8.2-7 10-4-1.8-7-5.2-7-10V6l7-3Z" /><path d="m9 12 2 2 4-5" /></>,
    stream: <><path d="M4 7h10M4 12h16M4 17h10" /><path d="m16 5 4 2-4 2M16 15l4 2-4 2" /></>,
  };

  return <svg className="tech-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">{paths[name]}</svg>;
}

function SectionHeading({ eyebrow, title, description, id }: { eyebrow: string; title: string; description?: string; id: string }) {
  return <div className="tech-section-heading">
    <p className="tech-eyebrow">{eyebrow}</p>
    <h2 id={id}>{title}</h2>
    {description && <p>{description}</p>}
  </div>;
}

function TechnologyHero() {
  return <section className="tech-hero" aria-labelledby="technology-title">
    <div className="shell tech-hero-inner">
      <p className="tech-eyebrow">Local runtime <span>•</span> Shared continuity <span>•</span> Multiple interfaces</p>
      <h1 id="technology-title">How Akane works</h1>
      <p className="tech-hero-accent">One coordinated path from message to memory.</p>
      <p className="tech-hero-lead">Akane combines local language-model inference with persistent Self and Memory, lightweight offscreen life, and multiple interfaces through one shared backend running on a Raspberry Pi 5.</p>
    </div>
  </section>;
}

function ArchitectureAndLifecycle() {
  return <section id="architecture" className="tech-section tech-architecture-section" aria-labelledby="architecture-title">
    <div className="shell">
      <SectionHeading
        eyebrow="System architecture"
        title="From message to continuity"
        description="One request follows the same coordinated path from every interface."
        id="architecture-title"
      />
      <figure className="tech-architecture" aria-describedby="architecture-summary">
        <figcaption id="architecture-summary" className="sr-only">An interface request resolves an owner or guest profile, selects relevant context, compiles one prompt, uses the configured local model runtime, then delivers visible text and commits validated state.</figcaption>
        <ol>
          {architectureStages.map((stage, index) => <li key={stage.title}>
            <article>
              <span className="tech-stage-index">0{index + 1}</span>
              <TechIcon name={stage.icon} />
              <div><h3>{stage.title}</h3><p>{stage.description}</p></div>
            </article>
            {index < architectureStages.length - 1 && <span className="tech-stage-arrow" aria-hidden="true">→</span>}
          </li>)}
        </ol>
      </figure>
      <div className="tech-background-lane"><span>After a completed turn</span><i aria-hidden="true">→</i><strong>Background Reflection</strong><i aria-hidden="true">→</i><span>Validated Self &amp; Memory changes</span></div>
      <details className="tech-implementation-details">
        <summary>Developer details <span>Classes and state ownership</span></summary>
        <div className="tech-detail-groups">
          {implementationGroups.map(group => <section key={group.title}>
            <h3>{group.title}</h3>
            <dl>{group.items.map(([name, description]) => <div key={name}><dt>{name}</dt><dd>{description}</dd></div>)}</dl>
          </section>)}
        </div>
      </details>
    </div>
  </section>;
}

function ProfileIsolation() {
  const shared = ["Shared request pipeline", "Shared inference coordinator", "Same local model"];
  const isolated = ["Conversation history", "Memory", "Self", "Private profile state"];

  return <section className="tech-section tech-isolation-section" aria-labelledby="isolation-title">
    <div className="shell tech-isolation-inner">
      <div className="tech-isolation-copy">
        <p className="tech-eyebrow">Profile isolation</p>
        <h2 id="isolation-title">One runtime without mixed identities</h2>
        <p>Desktop and Discord share the owner’s private continuity. Every website visitor receives a separate temporary guest profile.</p>
      </div>
      <figure className="tech-profile-diagram" aria-describedby="profile-summary">
        <figcaption id="profile-summary" className="sr-only">Desktop popup and Discord share owner continuity. The website demo uses temporary guest continuity. Both use the same request pipeline, inference coordinator, and local model without sharing history, Memory, Self, or private profile state.</figcaption>
        <div className="tech-profile-branches">
          <article><TechIcon name="profile" /><span>Owner continuity</span><h3>Desktop popup</h3><h3>Discord</h3></article>
          <article><TechIcon name="shield" /><span>Temporary guest continuity</span><h3>Website demo</h3></article>
        </div>
        <div className="tech-boundary-lists">
          <div><strong>Both use</strong><ul>{shared.map(item => <li key={item}>{item}</li>)}</ul></div>
          <div><strong>They do not share</strong><ul>{isolated.map(item => <li key={item}>{item}</li>)}</ul></div>
        </div>
      </figure>
    </div>
  </section>;
}

function TechnologyStack() {
  return <section className="tech-section tech-stack-section" aria-labelledby="stack-title">
    <div className="shell">
      <SectionHeading eyebrow="Confirmed technology stack" title="Six deliberate groups" description="The runtime stays local while the static website and HTTPS bridge remain separate deployment concerns." id="stack-title" />
      <div className="tech-stack-grid">
        {stackGroups.map(group => <article key={group.category}>
          <span>{group.category}</span><h3>{group.title}</h3>
          <ul>{group.items.map(item => <li key={item}>{item}</li>)}</ul>
        </article>)}
      </div>
    </div>
  </section>;
}

function EngineeringTradeoffs() {
  return <section className="tech-section tech-tradeoffs-section" aria-labelledby="tradeoffs-title">
    <div className="shell">
      <SectionHeading eyebrow="Engineering tradeoffs" title="The hardware shapes the architecture" description="Akane’s constraints lead to explicit scheduling, budgeting, commit, and isolation decisions." id="tradeoffs-title" />
      <div className="tech-tradeoffs-grid">
        {tradeoffs.map(item => <article key={item.title}>
          <h3>{item.title}</h3>
          <dl>
            <div><dt>Constraint</dt><dd>{item.constraint}</dd></div>
            <div><dt>Engineering response</dt><dd>{item.response}</dd></div>
            <div><dt>Why it matters</dt><dd>{item.importance}</dd></div>
          </dl>
        </article>)}
      </div>
    </div>
  </section>;
}

function PlannedPresentation() {
  return <section className="tech-section tech-roadmap-section" aria-labelledby="roadmap-title">
    <div className="shell">
      <SectionHeading eyebrow="Planned presentation" title="A richer presence, not a second personality" id="roadmap-title" />
      <p className="tech-roadmap-principle">Personality, memory, and conversation state remain backend-owned. The presentation layer only renders voice, movement, and expression.</p>
      <div className="tech-roadmap-list">
        {roadmapItems.map(item => <div key={item.title}><strong>{item.title}</strong><span className={`tech-status ${item.status === "Planned" ? "planned" : "in-development"}`}>{item.status}</span></div>)}
      </div>
    </div>
  </section>;
}

function TechnologyCTA() {
  return <section className="tech-cta" aria-labelledby="technology-cta-title">
    <div className="shell tech-cta-inner">
      <div><p className="tech-eyebrow">Explore Akane</p><h2 id="technology-cta-title">See the real pipeline in action.</h2><p>Try the isolated guest demo to experience the shared runtime.</p></div>
      <Link className="tech-button" to="/demo">Try the Demo<TechIcon name="arrow" /></Link>
    </div>
  </section>;
}

export function TechnologyPage() {
  return <main className="technology-page">
    <TechnologyHero />
    <ArchitectureAndLifecycle />
    <ProfileIsolation />
    <TechnologyStack />
    <EngineeringTradeoffs />
    <PlannedPresentation />
    <TechnologyCTA />
  </main>;
}
