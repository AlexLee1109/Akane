import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { projectConfig } from "../config/project";
import "./technology.css";

type IconName =
  | "arrow"
  | "brain"
  | "browser"
  | "clock"
  | "code"
  | "coordinator"
  | "desktop"
  | "discord"
  | "emotion"
  | "identity"
  | "memory"
  | "model"
  | "persistence"
  | "profile"
  | "shield"
  | "stream";

type RoadmapStatus = "Available" | "In development" | "Planned";

interface ArchitectureStage {
  icon: IconName;
  title: string;
  description: string;
}

interface LifecycleStage {
  title: string;
  description: string;
}

interface CoreSystem {
  icon: IconName;
  title: string;
  description: string;
  details: readonly string[];
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

interface RoadmapItem {
  icon: IconName;
  title: string;
  status: RoadmapStatus;
  description: string;
}

const architectureStages: readonly ArchitectureStage[] = [
  {
    icon: "browser",
    title: "Interfaces",
    description: "Desktop popup, Discord, and website guest demo.",
  },
  {
    icon: "profile",
    title: "Profile and context",
    description: "Resolve the owner or guest profile and select relevant conversation, memory, relationship, emotion, and time context.",
  },
  {
    icon: "brain",
    title: "Prompt compilation",
    description: "Combine identity, behavioral rules, current input, and selected context within the available token budget.",
  },
  {
    icon: "model",
    title: "Local inference",
    description: `Coordinate one ${projectConfig.modelName} runtime through llama.cpp on the Raspberry Pi.`,
  },
  {
    icon: "stream",
    title: "Streaming and state commit",
    description: "Stream visible text, then validate and persist completed state changes.",
  },
];

const implementationDetails = [
  ["StateStore", "Owns schema-versioned profile, conversation, memory, relationship, emotion, and presence state."],
  ["PromptPlan", "Budgets character policy, current input, complete recent pairs, and selected context before final tokenization."],
  ["GenerationScheduler", "Bounds visible work and coordinates foreground requests around one constrained model."],
  ["ModelManager", "Owns the singleton llama.cpp runtime, reservations, tokenization, and streamed generation."],
  ["Atomic JSON commit", "Validates state, writes a temporary document, flushes it, and atomically replaces the authoritative file."],
  ["NDJSON streaming", "Carries start, text delta, presentation, completion, and safe error events over streamed HTTP."],
  ["OffscreenPresenceWorker", "Coordinates event-driven background presence through the same model and state authorities."],
] as const;

const lifecycleStages: readonly LifecycleStage[] = [
  { title: "Receive", description: "The active interface validates the message and resolves its server-authorized profile." },
  { title: "Select context", description: "A stable snapshot supplies complete recent pairs and only the durable context relevant to this turn." },
  { title: "Compile", description: "Identity, rules, the current message, and selected context are assembled within the prompt budget." },
  { title: "Coordinate inference", description: "Visible work reserves the shared local runtime, with owner requests ahead of guests and background activity." },
  { title: "Stream", description: "Akane’s visible text reaches the active interface incrementally while internal state proposals remain hidden." },
  { title: "Validate and commit", description: "Only a completed turn and independently accepted state changes become durable; partial or cancelled generations do not." },
];

const coreSystems: readonly CoreSystem[] = [
  {
    icon: "identity",
    title: "Identity and behavior",
    description: "One character definition and one set of behavioral boundaries shape every interface.",
    details: ["Validated identity and soul files", "Hard behavioral rules", "Interface-neutral response pipeline"],
  },
  {
    icon: "memory",
    title: "Memory and continuity",
    description: "Recent dialogue and selected durable memories inform a turn without placing every stored detail into the prompt.",
    details: ["Complete recent conversation pairs", "Selective durable recall", "Meaningful preferences and experiences"],
  },
  {
    icon: "emotion",
    title: "Relationship and emotion",
    description: "Validated profile state preserves relationship, grounded emotion, mood, interests, preferences, and opinions.",
    details: ["Profile-scoped relationship state", "Grounded emotional continuity", "Validated state proposals"],
  },
  {
    icon: "clock",
    title: "Presence and initiative",
    description: "Time and offscreen-life state can support continuity without creating a second personality or model runtime.",
    details: ["Local time context", "Recorded presence state", "Coordinated background work"],
  },
];

const stackGroups: readonly StackGroup[] = [
  {
    category: "Model",
    title: projectConfig.modelName,
    items: ["Instruction-tuned Gemma generation", "llama.cpp via llama-cpp-python", "Q4_K_M GGUF", "4,096-token configured context"],
  },
  {
    category: "Backend",
    title: "Shared Python runtime",
    items: ["Python", "FastAPI", "Uvicorn", "Streaming HTTP transport"],
  },
  {
    category: "Interfaces",
    title: "One companion, three surfaces",
    items: ["Desktop popup", "Discord", "React and TypeScript website"],
  },
  {
    category: "Persistence",
    title: "Validated local state",
    items: ["Schema-versioned atomic JSON", "Conversation records", "Memory and companion state"],
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
    constraint: "Foreground and background work compete for one constrained local model.",
    response: "A shared scheduler and model reservation coordinate work by priority.",
    importance: "Conversation stays responsive without loading duplicate runtimes on the Pi.",
  },
  {
    title: "Context must earn its tokens",
    constraint: "A 4,096-token window cannot hold every conversation and every piece of companion state.",
    response: "The prompt compiler protects identity and current input, keeps complete recent pairs, and selects optional context by relevance.",
    importance: "Continuity remains useful without crowding out the current conversation.",
  },
  {
    title: "State must survive interruption",
    constraint: "Streaming can be stopped, time out, or lose its connection before a response finishes.",
    response: "The system previews state for inference and commits only after successful completion and validation.",
    importance: "Partial replies do not silently become permanent history or memory.",
  },
  {
    title: "Interfaces must remain one companion",
    constraint: "Popup, Discord, and the website need different adapter behavior and privacy boundaries.",
    response: "Thin interfaces share personality, memory, and inference ownership while profile resolution isolates guests.",
    importance: "Akane stays consistent without leaking private owner continuity to the public demo.",
  },
];

const roadmapItems: readonly RoadmapItem[] = [
  {
    icon: "stream",
    title: "Speech output",
    status: "Planned",
    description: "Akane currently communicates through text; no website voice or TTS controls are implemented.",
  },
  {
    icon: "identity",
    title: "Live2D presentation",
    status: "Planned",
    description: "The current popup and website use approved static character artwork; no Live2D renderer ships today.",
  },
  {
    icon: "emotion",
    title: "Expression synchronization",
    status: "In development",
    description: "Typed presentation-state plumbing is available, but visual expression rendering is not yet implemented.",
  },
  {
    icon: "coordinator",
    title: "Lip synchronization",
    status: "Planned",
    description: "Mouth-state boundaries exist in presentation state, but there is no audio-driven lip-sync pipeline.",
  },
  {
    icon: "desktop",
    title: "Deeper desktop integration",
    status: "Planned",
    description: "Future desktop work can build on the existing popup without moving personality or continuity into the interface.",
  },
];

function TechIcon({ name }: { name: IconName }) {
  const paths: Record<IconName, ReactNode> = {
    arrow: <><path d="M5 12h13" /><path d="m14 8 4 4-4 4" /></>,
    brain: <><path d="M9.5 4.5A3.5 3.5 0 0 0 6 8v.3A3.2 3.2 0 0 0 4 11.2 3.1 3.1 0 0 0 6.2 14 3.5 3.5 0 0 0 9.5 19.5" /><path d="M14.5 4.5A3.5 3.5 0 0 1 18 8v.3a3.2 3.2 0 0 1 2 2.9 3.1 3.1 0 0 1-2.2 2.8 3.5 3.5 0 0 1-3.3 5.5M12 4v16M8 9.5h4M12 14.5h4" /></>,
    browser: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 8h18M7 6h.01M10 6h.01" /></>,
    clock: <><circle cx="12" cy="12" r="8.5" /><path d="M12 7v5l3.5 2" /></>,
    code: <><path d="m8.5 8-4 4 4 4M15.5 8l4 4-4 4M14 5l-4 14" /></>,
    coordinator: <><circle cx="12" cy="12" r="2.2" /><circle cx="5" cy="6" r="1.7" /><circle cx="19" cy="6" r="1.7" /><circle cx="5" cy="18" r="1.7" /><circle cx="19" cy="18" r="1.7" /><path d="m6.5 7.3 3.8 3.2M17.5 7.3l-3.8 3.2M6.5 16.7l3.8-3.2M17.5 16.7l-3.8-3.2" /></>,
    desktop: <><rect x="3" y="4" width="18" height="13" rx="2" /><path d="M8 21h8M12 17v4" /></>,
    discord: <><path d="M6 7c4-2 8-2 12 0l1.5 9a11 11 0 0 1-3.8 2l-1.2-1.7M18 7l-1.5 9a11 11 0 0 1-3.5 1.8" /><circle cx="9" cy="12" r="1" /><circle cx="15" cy="12" r="1" /></>,
    emotion: <><path d="M20.5 9.5c0 5-8.5 10-8.5 10s-8.5-5-8.5-10A4.5 4.5 0 0 1 12 7.4a4.5 4.5 0 0 1 8.5 2.1Z" /></>,
    identity: <><circle cx="12" cy="8" r="3.5" /><path d="M5 20a7 7 0 0 1 14 0" /><path d="M18.5 4.5 20 3m-1.5 1.5L20 6m-1.5-1.5H21.5" /></>,
    memory: <><path d="M5 5.5h14v11H9l-4 3v-14Z" /><path d="M8.5 9h7M8.5 12h5" /></>,
    model: <><rect x="5" y="5" width="14" height="14" rx="2" /><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" /><path d="m10 14 2-5 2 5" /></>,
    persistence: <><ellipse cx="12" cy="5.5" rx="7" ry="3" /><path d="M5 5.5v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6M5 11.5v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" /></>,
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
      <p className="tech-hero-accent">One companion. One coordinated runtime.</p>
      <p className="tech-hero-lead">Akane combines local language-model inference with persistent memory, relationship continuity, grounded emotion, and multiple interfaces through one shared backend running on a Raspberry Pi 5.</p>
    </div>
  </section>;
}

function ArchitectureOverview() {
  return <section id="architecture" className="tech-section tech-architecture-section" aria-labelledby="architecture-title">
    <div className="shell">
      <SectionHeading
        eyebrow="System architecture"
        title="Five stages, one coordinated path"
        description="Each interface enters the same understandable request flow before implementation details come into view."
        id="architecture-title"
      />
      <figure className="tech-architecture" aria-describedby="architecture-summary">
        <figcaption id="architecture-summary" className="sr-only">Desktop popup, Discord, and the website resolve a profile and relevant context, compile one prompt, use the local Gemma runtime, then stream visible text and commit validated state.</figcaption>
        <ol>
          {architectureStages.map((stage, index) => <li key={stage.title}>
            <article>
              <span className="tech-stage-index">0{index + 1}</span>
              <TechIcon name={stage.icon} />
              <div><h3>{stage.title}</h3><p>{stage.description}</p></div>
            </article>
            {index < architectureStages.length - 1 && <span className="tech-stage-arrow" aria-hidden="true">↓</span>}
          </li>)}
        </ol>
      </figure>
      <details className="tech-implementation-details">
        <summary>View implementation details</summary>
        <dl>
          {implementationDetails.map(([name, description]) => <div key={name}><dt>{name}</dt><dd>{description}</dd></div>)}
        </dl>
      </details>
    </div>
  </section>;
}

function ResponseLifecycle() {
  return <section className="tech-section tech-lifecycle-section" aria-labelledby="lifecycle-title">
    <div className="shell">
      <SectionHeading
        eyebrow="Response lifecycle"
        title="From message to continuity"
        description="One request becomes visible text, then one explicit decision about what becomes durable."
        id="lifecycle-title"
      />
      <ol className="tech-lifecycle">
        {lifecycleStages.map((stage, index) => <li key={stage.title}>
          <span>{String(index + 1).padStart(2, "0")}</span>
          <div><h3>{stage.title}</h3><p>{stage.description}</p></div>
        </li>)}
      </ol>
    </div>
  </section>;
}

function ProfileIsolation() {
  const shared = ["Shared request pipeline", "Shared inference coordinator", "Same local model runtime"];
  const isolated = ["Conversation history", "Memory", "Relationship", "Private profile state"];

  return <section className="tech-section tech-isolation-section" aria-labelledby="isolation-title">
    <div className="shell tech-isolation-inner">
      <div className="tech-isolation-copy">
        <p className="tech-eyebrow">Profile isolation</p>
        <h2 id="isolation-title">One runtime without mixed identities</h2>
        <p>Desktop and Discord share the owner’s private continuity. Every website visitor receives a separate temporary guest profile.</p>
      </div>
      <figure className="tech-profile-diagram" aria-describedby="profile-summary">
        <figcaption id="profile-summary" className="sr-only">Desktop popup and Discord share owner continuity. The website demo uses temporary guest continuity. Both use the same request pipeline, inference coordinator, and local model without sharing history, memory, relationship, or private profile state.</figcaption>
        <div className="tech-profile-branches">
          <article>
            <TechIcon name="profile" />
            <span>Owner continuity</span>
            <h3>Desktop popup</h3>
            <h3>Discord</h3>
          </article>
          <article>
            <TechIcon name="shield" />
            <span>Temporary guest continuity</span>
            <h3>Website demo</h3>
          </article>
        </div>
        <div className="tech-boundary-lists">
          <div><strong>Both use</strong><ul>{shared.map(item => <li key={item}>{item}</li>)}</ul></div>
          <div><strong>They do not share</strong><ul>{isolated.map(item => <li key={item}>{item}</li>)}</ul></div>
        </div>
      </figure>
    </div>
  </section>;
}

function CoreSystems() {
  return <section className="tech-section tech-core-section" aria-labelledby="core-title">
    <div className="shell">
      <SectionHeading
        eyebrow="Core systems"
        title="What keeps Akane consistent"
        description="Each system has a narrow responsibility, while the backend remains the authority for personality, state, and inference."
        id="core-title"
      />
      <div className="tech-core-grid">
        {coreSystems.map(system => <article key={system.title}>
          <TechIcon name={system.icon} />
          <h3>{system.title}</h3>
          <p>{system.description}</p>
          <ul>{system.details.map(detail => <li key={detail}>{detail}</li>)}</ul>
        </article>)}
      </div>
    </div>
  </section>;
}

function TechnologyStack() {
  return <section className="tech-section tech-stack-section" aria-labelledby="stack-title">
    <div className="shell">
      <SectionHeading
        eyebrow="Confirmed technology stack"
        title="Six deliberate groups"
        description="The runtime stays local while the static website and HTTPS bridge remain separate deployment concerns."
        id="stack-title"
      />
      <div className="tech-stack-grid">
        {stackGroups.map(group => <article key={group.category}>
          <span>{group.category}</span>
          <h3>{group.title}</h3>
          <ul>{group.items.map(item => <li key={item}>{item}</li>)}</ul>
        </article>)}
      </div>
    </div>
  </section>;
}

function EngineeringTradeoffs() {
  return <section className="tech-section tech-tradeoffs-section" aria-labelledby="tradeoffs-title">
    <div className="shell">
      <SectionHeading
        eyebrow="Engineering tradeoffs"
        title="The hardware shapes the architecture"
        description="Akane’s constraints lead to explicit scheduling, budgeting, commit, and isolation decisions."
        id="tradeoffs-title"
      />
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
      <SectionHeading
        eyebrow="Planned presentation layer"
        title="A richer presence, not a second personality"
        description="The backend owns personality, memory, emotion, and conversation state. The future presentation layer will render voice, expression, movement, and lip synchronization."
        id="roadmap-title"
      />
      <div className="tech-roadmap-principle">
        <TechIcon name="shield" />
        <div><span className="tech-status available">Available</span><p><strong>Presentation-state plumbing is implemented.</strong> Visual expression rendering, voice, and lip synchronization are not.</p></div>
      </div>
      <div className="tech-roadmap-grid">
        {roadmapItems.map(item => <article key={item.title}>
          <TechIcon name={item.icon} />
          <span className={`tech-status ${item.status.toLowerCase().replace(" ", "-")}`}>{item.status}</span>
          <h3>{item.title}</h3>
          <p>{item.description}</p>
        </article>)}
      </div>
    </div>
  </section>;
}

function TechnologyCTA() {
  return <section className="tech-cta" aria-labelledby="technology-cta-title">
    <div className="shell tech-cta-inner">
      <div><p className="tech-eyebrow">Explore Akane</p><h2 id="technology-cta-title">See the real pipeline in action.</h2><p>Try the isolated guest demo to experience the shared runtime.</p></div>
      <div className="tech-actions"><Link className="tech-button tech-button-light" to="/demo">Try the Demo<TechIcon name="arrow" /></Link></div>
    </div>
  </section>;
}

export function TechnologyPage() {
  return <main className="technology-page">
    <TechnologyHero />
    <ArchitectureOverview />
    <ResponseLifecycle />
    <ProfileIsolation />
    <CoreSystems />
    <TechnologyStack />
    <EngineeringTradeoffs />
    <PlannedPresentation />
    <TechnologyCTA />
  </main>;
}
