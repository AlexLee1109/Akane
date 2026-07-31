import { type AkanePresentationState, presentationStateMachine } from "../presentation";

interface AkaneStageProps {
  imageSrc: string;
  bubbleText: string;
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
    className="akane-image"
    src={imageSrc}
    alt="Akane, a blue-haired anime-style companion in a white jacket"
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
  return <div className="stage-bubble">{text}</div>;
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
  return <section className="stage panel" aria-label="Akane stage">
    <div className="stage-badges"><span>{props.connectionLabel}</span><span>{props.modelName}</span></div>
    <ConversationView text={props.bubbleText} />
    <AudioPlaybackController imageSrc={props.imageSrc} state={state} />
    <div className="stage-footer"><span>✦ {state.activity}</span><span>{props.sessionLabel}</span></div>
  </section>;
}
