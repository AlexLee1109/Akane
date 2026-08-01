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

type RoadmapStatus = "In development" | "Planned" | "Exploratory";

interface ArchitectureStage {
  icon: IconName;
  label: string;
  detail: string;
  emphasis?: boolean;
}

interface CoreSystem {
  icon: IconName;
  title: string;
  description: string;
  systems: readonly string[];
}

interface StackItem {
  category: string;
  technology: string;
  purpose: string;
}

interface Tradeoff {
  number: string;
  title: string;
  constraint: string;
  consequence: string;
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
    label: "Interfaces",
    detail: "Desktop popup · Discord · website guest demo",
  },
  {
    icon: "shield",
    label: "Adapters and public API",
    detail: "Authenticated owner routes and a narrow guest-session API",
  },
  {
    icon: "profile",
    label: "Profile and request resolution",
    detail: "Canonical owner identity or server-resolved temporary guest",
  },
  {
    icon: "brain",
    label: "Context and prompt compilation",
    detail: "StateStore snapshot · relevance plan · PromptPlan",
  },
  {
    icon: "coordinator",
    label: "Shared inference coordinator",
    detail: "GenerationScheduler · priority-aware ModelManager reservation",
    emphasis: true,
  },
  {
    icon: "model",
    label: "Gemma 3n E4B through llama.cpp",
    detail: "Exact embedded-template tokenization · Q4_K_M GGUF",
    emphasis: true,
  },
  {
    icon: "stream",
    label: "Streamed response",
    detail: "Visible text is released while structured state stays hidden",
  },
  {
    icon: "persistence",
    label: "Validated state commit",
    detail: "Accepted changes and the completed turn are written atomically",
  },
];

const contextInputs = [
  "Identity, soul, and hard rules",
  "Complete recent conversation pairs",
  "Selected durable memories",
  "Relationship continuity",
  "Grounded emotion and mood",
  "Time and recorded presence",
] as const;

const persistedState = [
  "Conversation records",
  "Durable memories",
  "Canonical profile state",
  "Relationship, emotion, and mood",
  "Presence and initiative metadata",
] as const;

const responseLifecycle = [
  {
    title: "Receive",
    text: "The active adapter normalizes the message. Private routes resolve to the canonical owner; the public API resolves a bearer session to one temporary guest profile.",
  },
  {
    title: "Select context",
    text: "StateStore publishes an immutable snapshot. Recent complete pairs are retained, while durable memory and companion state are selected only when relevant.",
  },
  {
    title: "Compile",
    text: "Identity, behavioral constraints, the current message, and optional context are budgeted into one PromptPlan, then tokenized with the GGUF’s embedded chat template.",
  },
  {
    title: "Coordinate inference",
    text: "GenerationScheduler bounds visible work and ModelManager reserves the singleton runtime. Owner requests rank ahead of guests, which rank ahead of background work.",
  },
  {
    title: "Stream",
    text: "One Gemma completion releases visible text incrementally. Potential structured-state prefixes are retained so internal metadata never becomes dialogue.",
  },
  {
    title: "Validate and commit",
    text: "After successful generation, proposed fields are validated independently and the completed turn is committed atomically. Failed, cancelled, or partial generations are not saved.",
  },
] as const;

const coreSystems: readonly CoreSystem[] = [
  {
    icon: "identity",
    title: "Context and identity",
    description: "Controls who Akane is, how she speaks, and which information earns space in the current prompt.",
    systems: [
      "CharacterProfile",
      "PromptPlan compiler",
      "Complete recent-pair window",
      "Relevant-context planner",
      "TimeContext and requested editor context",
    ],
  },
  {
    icon: "memory",
    title: "Continuity",
    description: "Preserves meaningful personal and companion state without inserting every stored detail into every response.",
    systems: [
      "StateStore durable memory",
      "Canonical profile state",
      "Relationship continuity",
      "Grounded emotion and mood",
      "Offscreen presence and initiative",
    ],
  },
  {
    icon: "coordinator",
    title: "Runtime coordination",
    description: "Keeps foreground conversation, background activity, and state changes safe around one constrained model runtime.",
    systems: [
      "GenerationScheduler",
      "ModelManager reservations",
      "Visible-response stream filter",
      "Independent proposal validation",
      "AutonomousLifeWorker",
    ],
  },
];

const stackItems: readonly StackItem[] = [
  {
    category: "Local model",
    technology: "Gemma 3n E4B IT",
    purpose: "The configured instruction-tuned model, loaded from a Q4_K_M GGUF on the Raspberry Pi.",
  },
  {
    category: "Inference runtime",
    technology: "llama.cpp · llama-cpp-python",
    purpose: "Loads the GGUF, applies its embedded chat template, tokenizes exactly, and streams generation.",
  },
  {
    category: "Backend",
    technology: "Python · FastAPI · Uvicorn",
    purpose: "Owns the shared state, request orchestration, local API, and public guest-session boundary.",
  },
  {
    category: "Streaming transport",
    technology: "NDJSON over streamed HTTP",
    purpose: "Carries start, text-delta, completion, and error events; the private popup stream also exposes cancellation.",
  },
  {
    category: "Website",
    technology: "React · TypeScript · Vite",
    purpose: "Builds the static HashRouter site and offline preview for deployment under a GitHub Pages base path.",
  },
  {
    category: "Desktop interface",
    technology: "pywebview · HTML · CSS · JavaScript",
    purpose: "Hosts the transparent companion popup and bridges its streamed HTTP conversation to the window.",
  },
  {
    category: "Discord interface",
    technology: "discord.py · aiohttp",
    purpose: "Normalizes Discord events and calls the authenticated backend without loading another model or state store.",
  },
  {
    category: "Persistence",
    technology: "Schema-versioned atomic JSON",
    purpose: "Stores validated profiles and conversations through one temporary-file, fsync, and atomic-replace path.",
  },
  {
    category: "Developer context",
    technology: "VS Code extension · Node.js",
    purpose: "Offers bounded, read-only editor context when explicitly requested; it is not a separate companion runtime.",
  },
  {
    category: "Deployment",
    technology: "Raspberry Pi 5 · ngrok · GitHub Pages",
    purpose: "Runs inference and the API on the Pi while the static site is hosted separately and reaches it through HTTPS.",
  },
];

const tradeoffs: readonly Tradeoff[] = [
  {
    number: "01",
    title: "One shared model runtime",
    constraint: "Foreground conversation and background work compete for one memory-constrained local model.",
    consequence: "A priority-aware reservation serializes tokenization and inference; owner work can cooperatively displace lower-priority background work.",
  },
  {
    number: "02",
    title: "Context must earn its tokens",
    constraint: "A 4,096-token window cannot hold every conversation, memory, emotion, relationship entry, and presence detail.",
    consequence: "Protected identity and the current input stay fixed while recent complete pairs and optional context are selected and budgeted.",
  },
  {
    number: "03",
    title: "State must survive interruption",
    constraint: "Streaming can begin before a local completion finishes, and a client may disconnect partway through.",
    consequence: "Visible deltas are provisional. Only a successful completion reaches validated proposal handling and the atomic commit path.",
  },
  {
    number: "04",
    title: "Interfaces must not become separate companions",
    constraint: "Popup, Discord, the public website, and editor context need different adapter behavior without duplicating personality or memory logic.",
    consequence: "Thin adapters share the same orchestration core, while profile resolution keeps private owner continuity separate from every guest.",
  },
];

const roadmapItems: readonly RoadmapItem[] = [
  {
    icon: "stream",
    title: "Streaming speech output",
    status: "Planned",
    description: "Akane currently communicates through text. Speech and audio output remain outside the implemented conversation path.",
  },
  {
    icon: "identity",
    title: "Live2D presentation",
    status: "Planned",
    description: "The current website and popup render static character artwork; no Live2D model or Cubism runtime is shipped.",
  },
  {
    icon: "emotion",
    title: "Expression synchronization",
    status: "In development",
    description: "A typed presentation state machine and renderer boundary exist, but static artwork does not yet render expression changes.",
  },
  {
    icon: "clock",
    title: "Lip synchronization",
    status: "Exploratory",
    description: "The presentation boundary anticipates mouth state, but there is no implemented audio-driven lip-sync pipeline today.",
  },
];

function TechIcon({ name, className = "" }: { name: IconName; className?: string }) {
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

  return <svg className={`tech-icon ${className}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">{paths[name]}</svg>;
}

function SectionHeading({ eyebrow, title, description, id }: { eyebrow: string; title: string; description?: string; id: string }) {
  return <div className="tech-section-heading">
    <p className="tech-eyebrow"><span aria-hidden="true" />{eyebrow}</p>
    <h2 id={id}>{title}</h2>
    {description && <p>{description}</p>}
  </div>;
}

function GithubAction({ className = "tech-button tech-button-secondary", children = "View on GitHub" }: { className?: string; children?: ReactNode }) {
  return projectConfig.githubUrl
    ? <a className={className} href={projectConfig.githubUrl} target="_blank" rel="noreferrer">{children}<span aria-hidden="true">↗</span></a>
    : <span className={`${className} disabled`} aria-disabled="true">{children}</span>;
}

function FlowArrow() {
  return <span className="tech-flow-arrow" aria-hidden="true"><TechIcon name="arrow" /></span>;
}

function TechnologyHero() {
  const character = `${projectConfig.basePath}assets/akane-hero.png`;
  const facts = ["Raspberry Pi 5", "Gemma 3n E4B IT", "llama.cpp", "Python"] as const;

  return <section className="tech-hero" aria-labelledby="technology-title">
    <div className="tech-hero-grid" aria-hidden="true" />
    <div className="tech-hero-orbit tech-hero-orbit-one" aria-hidden="true" />
    <div className="tech-hero-orbit tech-hero-orbit-two" aria-hidden="true" />
    <div className="shell tech-hero-inner">
      <div className="tech-hero-copy">
        <p className="tech-eyebrow"><span aria-hidden="true" />Local runtime <b>•</b> Shared continuity <b>•</b> Multiple interfaces</p>
        <h1 id="technology-title">How Akane works</h1>
        <p className="tech-hero-accent">One companion. One coordinated runtime.</p>
        <p className="tech-hero-lead">Akane combines local language-model inference with persistent memory, relationship continuity, grounded emotion, and multiple interfaces through one shared backend running on a Raspberry Pi 5.</p>
        <p className="tech-boundary-note">The model and private owner state run on the Pi. This static website is hosted separately and reaches a narrow public API through a configured HTTPS endpoint.</p>
        <div className="tech-actions">
          <button className="tech-button tech-button-primary" type="button" onClick={() => document.getElementById("architecture")?.scrollIntoView()}>Explore the architecture<TechIcon name="arrow" /></button>
          <GithubAction />
        </div>
        <ul className="tech-hero-facts" aria-label="Configured runtime facts">
          {facts.map((fact, index) => <li key={fact}><span>{String(index + 1).padStart(2, "0")}</span>{fact}</li>)}
        </ul>
      </div>
      <div className="tech-artwork" aria-hidden="true">
        <div className="tech-local-badge"><TechIcon name="model" /><span>Configured runtime<strong>Local inference on Pi</strong></span></div>
        <span className="tech-star tech-star-one">✦</span>
        <span className="tech-star tech-star-two">✦</span>
        <span className="tech-star tech-star-three">✦</span>
        <img src={character} width="333" height="1146" alt="" fetchPriority="high" decoding="async" />
      </div>
    </div>
  </section>;
}

function ArchitectureOverview() {
  return <section id="architecture" className="tech-section tech-architecture-section" aria-labelledby="architecture-title">
    <div className="shell">
      <SectionHeading
        eyebrow="System architecture"
        title="One coordinated companion"
        description="Every interface enters the same conversation path. The backend resolves the active profile, gathers relevant context, coordinates local inference, streams the response, and commits validated state."
        id="architecture-title"
      />
      <figure className="tech-architecture" aria-describedby="architecture-summary">
        <figcaption id="architecture-summary" className="sr-only">Desktop popup, Discord, and the public website enter one shared request pipeline. Selected context feeds prompt compilation. The Gemma runtime streams visible text before validated conversation and companion state is committed to atomic JSON persistence.</figcaption>
        <div className="tech-architecture-legend" aria-label="Diagram legend">
          <span><i className="request" aria-hidden="true" />Request flow</span>
          <span><i className="context" aria-hidden="true">C</i>Context input</span>
          <span><i className="persistent" aria-hidden="true">P</i>Persistent state</span>
        </div>
        <div className="tech-architecture-layout">
          <div className="tech-request-lane">
            <p className="tech-lane-label">Request path</p>
            <ol>
              {architectureStages.map((stage, index) => <li key={stage.label}>
                <article className={stage.emphasis ? "emphasis" : ""}>
                  <TechIcon name={stage.icon} />
                  <div><h3>{stage.label}</h3><p>{stage.detail}</p></div>
                </article>
                {index < architectureStages.length - 1 && <span className="tech-down-connector" aria-hidden="true"><i /><TechIcon name="arrow" /></span>}
              </li>)}
            </ol>
          </div>
          <aside className="tech-context-rail" aria-labelledby="context-rail-title">
            <div className="tech-rail-heading"><TechIcon name="brain" /><div><span>Context input</span><h3 id="context-rail-title">Selected for this turn</h3></div></div>
            <p>These inputs feed PromptPlan compilation. They are not independent model services.</p>
            <ul>{contextInputs.map(item => <li key={item}><span aria-hidden="true">C</span>{item}</li>)}</ul>
            <div className="tech-rail-feed" aria-hidden="true"><span>feeds prompt</span><TechIcon name="arrow" /></div>
          </aside>
          <aside className="tech-persistence-rail" aria-labelledby="persistence-rail-title">
            <div className="tech-rail-heading"><TechIcon name="persistence" /><div><span>Persistent state</span><h3 id="persistence-rail-title">StateStore authority</h3></div></div>
            <p>A schema-versioned JSON document holds validated state. Runtime-only values are not presented as permanent.</p>
            <ul>{persistedState.map(item => <li key={item}><span aria-hidden="true">P</span>{item}</li>)}</ul>
            <div className="tech-rail-feed reverse" aria-hidden="true"><TechIcon name="arrow" /><span>snapshot / commit</span></div>
          </aside>
        </div>
      </figure>
    </div>
  </section>;
}

function RequestLifecycle() {
  return <section className="tech-section tech-lifecycle-section" aria-labelledby="lifecycle-title">
    <div className="shell">
      <SectionHeading
        eyebrow="Response lifecycle"
        title="From message to continuity"
        description="A normal request performs one coordinated inference, then makes one explicit decision about what becomes durable."
        id="lifecycle-title"
      />
      <ol className="tech-lifecycle">
        {responseLifecycle.map((stage, index) => <li key={stage.title}>
          <span className="tech-step-number">{String(index + 1).padStart(2, "0")}</span>
          <div><h3>{stage.title}</h3><p>{stage.text}</p></div>
        </li>)}
      </ol>
    </div>
  </section>;
}

function CoreSystems() {
  return <section className="tech-section tech-core-section" aria-labelledby="core-title">
    <div className="shell">
      <SectionHeading
        eyebrow="Companion core"
        title="Systems that preserve Akane across conversations"
        description="Stable character policy, selected context, evolving continuity, and runtime control each have a distinct responsibility."
        id="core-title"
      />
      <div className="tech-core-grid">
        {coreSystems.map((group, index) => <article key={group.title}>
          <header><span className="tech-core-index">0{index + 1}</span><TechIcon name={group.icon} /></header>
          <h3>{group.title}</h3>
          <p>{group.description}</p>
          <ul>{group.systems.map(system => <li key={system}>{system}</li>)}</ul>
        </article>)}
      </div>
    </div>
  </section>;
}

function ProfileIsolation() {
  return <section className="tech-section tech-isolation-section" aria-labelledby="isolation-title">
    <div className="shell tech-isolation-shell">
      <div className="tech-isolation-copy">
        <p className="tech-eyebrow"><span aria-hidden="true" />Multiple interfaces</p>
        <h2 id="isolation-title">One backend without mixing identities</h2>
        <p>Desktop and Discord share the owner’s private continuity. Each website visitor receives an isolated temporary guest profile that cannot access the owner’s conversation history, memories, relationship, emotion, mood, presence, or private state.</p>
        <p className="tech-vscode-note"><TechIcon name="code" /><span><strong>The VS Code bridge is context, not another companion.</strong> It offers bounded read-only editor context only when a message requests it and a bridge is connected.</span></p>
      </div>
      <figure className="tech-profile-diagram" aria-describedby="profile-summary">
        <figcaption id="profile-summary" className="sr-only">The desktop popup and Discord use one private owner profile. The website demo uses a separate temporary guest profile. Both enter the shared conversation pipeline and local runtime without sharing personal continuity.</figcaption>
        <div className="tech-profile-branches">
          <article className="tech-profile-card owner">
            <header><TechIcon name="profile" /><div><span>Private continuity</span><h3>Owner profile</h3></div></header>
            <ul><li><TechIcon name="desktop" />Desktop popup</li><li><TechIcon name="discord" />Discord</li></ul>
          </article>
          <article className="tech-profile-card guest">
            <header><TechIcon name="shield" /><div><span>Isolated continuity</span><h3>Temporary guest</h3></div></header>
            <ul><li><TechIcon name="browser" />Website demo</li></ul>
          </article>
        </div>
        <div className="tech-profile-join" aria-hidden="true"><span /><span /></div>
        <div className="tech-shared-pipeline"><TechIcon name="coordinator" /><div><span>Shared infrastructure</span><strong>Conversation pipeline · ModelManager · local Gemma runtime</strong></div></div>
      </figure>
      <dl className="tech-session-facts">
        <div><dt>Idle expiration</dt><dd>30 minutes</dd></div>
        <div><dt>Maximum lifetime</dt><dd>2 hours</dd></div>
        <div><dt>Public capacity</dt><dd>1 active · 2 queued</dd></div>
        <div><dt>Guest response cap</dt><dd>256 tokens</dd></div>
        <div><dt>Scheduling</dt><dd>Owner before guest</dd></div>
      </dl>
      <p className="tech-config-note">Configured defaults shown. Session tokens and internal profile identifiers are intentionally omitted.</p>
    </div>
  </section>;
}

function RuntimeAndPersistence() {
  const runtimeFacts = [
    ["Hardware", "Raspberry Pi 5"],
    ["Model", "Gemma 3n E4B Instruct"],
    ["Runtime", "llama.cpp via llama-cpp-python"],
    ["Format", "Q4_K_M quantized GGUF"],
    ["Context", "4,096 tokens"],
    ["Generation", "One coordinated local inference path"],
  ] as const;
  const reliability = [
    ["Atomic replacement", "Writes go to a temporary file, flush to disk, then replace the authoritative JSON document."],
    ["Validated state", "Current-schema headers, profiles, conversations, and proposed state fields are checked before publication."],
    ["Preview before commit", "A snapshot feeds inference; successful completion and accepted state changes are committed afterward."],
    ["Explicit recovery", "Corrupt or unsupported authoritative state raises a recovery error instead of silently resetting continuity."],
    ["Schema migration", "Validated legacy sources can be merged and rewritten once into the current canonical document."],
  ] as const;

  return <section className="tech-section tech-runtime-section" aria-labelledby="runtime-title">
    <div className="shell">
      <SectionHeading
        eyebrow="Local runtime"
        title="Designed around constrained hardware"
        description="Akane favors a small number of explicit authorities and controlled state transitions over distributed services the hardware does not need."
        id="runtime-title"
      />
      <div className="tech-runtime-grid">
        <article className="tech-runtime-card">
          <header><TechIcon name="model" /><div><span>Configured deployment</span><h3>Inference runtime</h3></div></header>
          <dl>{runtimeFacts.map(([term, value]) => <div key={term}><dt>{term}</dt><dd>{value}</dd></div>)}</dl>
        </article>
        <article className="tech-reliability-card">
          <header><TechIcon name="persistence" /><div><span>StateStore</span><h3>Persistence and reliability</h3></div></header>
          <p>State moves through controlled commit paths so an interrupted generation does not silently become permanent companion history.</p>
          <ul>{reliability.map(([title, text]) => <li key={title}><strong>{title}</strong><span>{text}</span></li>)}</ul>
          <small>Persistence is JSON-based. No transactional database guarantee is implied.</small>
        </article>
      </div>
    </div>
  </section>;
}

function TechnologyStack() {
  return <section className="tech-section tech-stack-section" aria-labelledby="stack-title">
    <div className="shell">
      <SectionHeading
        eyebrow="Technology stack"
        title="Small, deliberate building blocks"
        description="Every item below is present in current source, manifests, configuration, or the documented deployment path."
        id="stack-title"
      />
      <div className="tech-stack-grid">
        {stackItems.map(item => <article key={item.category}>
          <span className="tech-stack-category">{item.category}</span>
          <h3>{item.technology}</h3>
          <p>{item.purpose}</p>
          <span className="tech-implemented"><span aria-hidden="true">✓</span>Implemented</span>
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
        description="These are deliberate constraints, not claims of a perfectly solved system. Each one changes how Akane selects context, schedules work, and protects continuity."
        id="tradeoffs-title"
      />
      <div className="tech-tradeoffs-grid">
        {tradeoffs.map(item => <article key={item.number}>
          <span>{item.number}</span>
          <h3>{item.title}</h3>
          <p>{item.constraint}</p>
          <div><strong>Why it matters</strong><p>{item.consequence}</p></div>
        </article>)}
      </div>
    </div>
  </section>;
}

function TechnologyRoadmap() {
  return <section className="tech-section tech-roadmap-section" aria-labelledby="roadmap-title">
    <div className="shell">
      <SectionHeading
        eyebrow="What comes next"
        title="Giving the same intelligence a richer presence"
        description="Presentation work stays visibly outside the implemented architecture until it can render the existing backend state without becoming a second source of personality."
        id="roadmap-title"
      />
      <div className="tech-roadmap-principle">
        <TechIcon name="shield" />
        <div><span>Architectural principle</span><p>The backend owns personality, memory, emotion, and conversation state. The future presentation layer will render voice, expression, motion, and lip synchronization without redefining Akane.</p></div>
      </div>
      <div className="tech-roadmap-grid">
        {roadmapItems.map(item => <article key={item.title}>
          <TechIcon name={item.icon} />
          <span className={`tech-roadmap-status ${item.status.toLowerCase().replace(" ", "-")}`}>{item.status}</span>
          <h3>{item.title}</h3>
          <p>{item.description}</p>
        </article>)}
      </div>
    </div>
  </section>;
}

function TechnologyCTA() {
  return <section className="tech-cta shell" aria-labelledby="technology-cta-title">
    <div className="tech-cta-star" aria-hidden="true">✦</div>
    <div><p className="tech-eyebrow"><span aria-hidden="true" />Explore Akane</p><h2 id="technology-cta-title">See the architecture in motion.</h2><p>Try the isolated live guest experience, inspect the implementation, or follow the presentation layer as it develops.</p></div>
    <div className="tech-actions"><Link className="tech-button tech-button-light" to="/demo">Try the Demo<TechIcon name="arrow" /></Link><GithubAction className="tech-button tech-button-outline" /></div>
  </section>;
}

export function TechnologyPage() {
  return <main className="technology-page">
    <TechnologyHero />
    <ArchitectureOverview />
    <RequestLifecycle />
    <CoreSystems />
    <ProfileIsolation />
    <RuntimeAndPersistence />
    <TechnologyStack />
    <EngineeringTradeoffs />
    <TechnologyRoadmap />
    <TechnologyCTA />
  </main>;
}
