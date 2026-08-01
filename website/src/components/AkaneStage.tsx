import { type AkanePresentationState, presentationStateMachine } from "../presentation";

interface AkaneStageProps {
  imageSrc: string;
  responseText?: string;
  connection: "connecting" | "live" | "showcase";
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
    data-activity={state.activity}
    data-expression={state.expression}
    data-mouth-level={state.mouthLevel}
  />;
}

function SpeechBubble({ text }: { text: string }) {
  return <div className="akane-stage-bubble">{text}</div>;
}

function ConversationView({ text }: { text: string }) {
  return <SpeechBubble text={text} />;
}

export function AkaneStage(props: AkaneStageProps) {
  const state = presentationStateMachine(
    props.connection,
    props.generating,
    props.hasResponseText,
    props.backendPresentation,
  );
  return <section className="akane-stage demo-panel" aria-label="Akane stage">
    {props.connection !== "connecting" && <span className={`akane-stage-mode ${props.connection}`}>{props.connection === "live" ? "Live" : "Preview"}</span>}
    {props.responseText?.trim() && <ConversationView text={props.responseText} />}
    <CharacterRenderer imageSrc={props.imageSrc} state={state} />
  </section>;
}
