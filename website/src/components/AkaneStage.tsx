import { type AkanePresentationState, presentationStateMachine } from "../presentation";

interface AkaneStageProps {
  imageSrc: string;
  responseText?: string;
  connection: "connecting" | "live" | "showcase";
  connectionLabel: string;
  generating: boolean;
  hasResponseText: boolean;
  modelName: string;
  sessionLabel: string;
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

function ExpressionController(props: { imageSrc: string; state: AkanePresentationState }) {
  return <CharacterRenderer {...props} />;
}

function LipSyncController(props: { imageSrc: string; state: AkanePresentationState }) {
  return <ExpressionController {...props} />;
}

function AudioPlaybackController(props: { imageSrc: string; state: AkanePresentationState }) {
  return <LipSyncController {...props} />;
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
    <span className="akane-stage-orbit akane-stage-orbit-one" aria-hidden="true" />
    <span className="akane-stage-orbit akane-stage-orbit-two" aria-hidden="true" />
    <span className="akane-stage-star akane-stage-star-one" aria-hidden="true">✦</span>
    <span className="akane-stage-star akane-stage-star-two" aria-hidden="true">✦</span>
    <div className="akane-stage-badges"><span>{props.connectionLabel}</span><span>{props.modelName}</span></div>
    {props.responseText?.trim() && <ConversationView text={props.responseText} />}
    <AudioPlaybackController imageSrc={props.imageSrc} state={state} />
    <div className="akane-stage-footer"><span>✦ {state.activity}</span><span>{props.sessionLabel}</span></div>
  </section>;
}
