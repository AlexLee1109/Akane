export interface AkanePresentationState {
  activity:
    | "idle"
    | "connecting"
    | "listening"
    | "thinking"
    | "speaking"
    | "interrupted"
    | "offline";
  expression:
    | "neutral"
    | "calm"
    | "curious"
    | "amused"
    | "happy"
    | "concerned"
    | "sad"
    | "irritated";
  mouthLevel: number;
  lookTarget?: { x: number; y: number };
}

export function presentationStateMachine(
  connection: "connecting" | "live" | "offline" | "preview",
  generating: boolean,
  hasResponseText: boolean,
  backendState?: AkanePresentationState,
): AkanePresentationState {
  if (connection === "connecting") {
    return { activity: "connecting", expression: "neutral", mouthLevel: 0 };
  }
  if (connection === "offline") {
    return { activity: "offline", expression: "neutral", mouthLevel: 0 };
  }
  if (connection === "preview") {
    return { activity: "idle", expression: "neutral", mouthLevel: 0 };
  }
  if (backendState) return backendState;
  if (generating) {
    return {
      activity: hasResponseText ? "speaking" : "thinking",
      expression: "neutral",
      mouthLevel: hasResponseText ? 0.5 : 0,
    };
  }
  return { activity: "idle", expression: "neutral", mouthLevel: 0 };
}
