import { type AkanePresentationState, presentationStateMachine } from "../presentation";

interface AkaneStageProps {
  imageSrc: string;
  connection: "connecting" | "live" | "offline" | "preview";
  generating: boolean;
  hasResponseText: boolean;
  backendPresentation?: AkanePresentationState;
}

function CharacterRenderer({ imageSrc, state }: { imageSrc: string; state: AkanePresentationState }) {
  return <img
    className="akane-image akane-stage-image"
    src={imageSrc}
    alt="Akane, a blue-haired anime-style companion in a white jacket"
    decoding="async"
    width="335"
    height="1147"
    data-activity={state.activity}
    data-expression={state.expression}
    data-mouth-level={state.mouthLevel}
  />;
}

export function AkaneStage(props: AkaneStageProps) {
  const state = presentationStateMachine(
    props.connection,
    props.generating,
    props.hasResponseText,
    props.backendPresentation,
  );
  const activity = props.connection === "preview" ? "Simulated preview"
    : props.connection === "offline" ? "Waiting for a connection"
      : props.connection === "connecting" ? "Connecting to Akane…"
        : props.generating ? (props.hasResponseText ? "Writing a reply…" : "Thinking…") : "Ready when you are";
  return <section className="akane-stage demo-panel" aria-label="Akane stage">
    <div className="akane-stage-heading"><span>YOUR COMPANION</span><span lang="ja">茜</span></div>
    <CharacterRenderer imageSrc={props.imageSrc} state={state} />
    <div className="akane-stage-caption"><p>Akane<span className="stage-caption-line">A familiar presence.</span></p><span role="status">{activity}</span></div>
  </section>;
}
