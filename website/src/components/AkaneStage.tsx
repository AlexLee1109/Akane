import { type AkanePresentationState, presentationStateMachine } from "../presentation";

interface AkaneStageProps {
  imageSrc: string;
  responseText?: string;
  connection: "connecting" | "live" | "showcase";
  generating: boolean;
  hasResponseText: boolean;
  isThinking: boolean;
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

function SpeechBubble({ text, thinking }: { text?: string; thinking: boolean }) {
  return <div className={`akane-stage-bubble ${thinking ? "thinking" : ""}`}>
    {thinking
      ? <><span className="akane-thinking-dots" aria-hidden="true"><i /><i /><i /></span><span className="sr-only">Akane is thinking</span></>
      : text}
  </div>;
}

function ConversationView({ text, thinking }: { text?: string; thinking: boolean }) {
  return <SpeechBubble text={text} thinking={thinking} />;
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
    {(props.responseText?.trim() || props.isThinking) && <ConversationView text={props.responseText} thinking={props.isThinking} />}
    <CharacterRenderer imageSrc={props.imageSrc} state={state} />
  </section>;
}
