type DemoConnection = "connecting" | "live" | "offline" | "preview";

interface DemoControlsProps {
  connection: DemoConnection;
  guestEnabled: boolean;
  activeSession: boolean;
  generating: boolean;
  actionPending: boolean;
  retryExhausted: boolean;
  canReconnect: boolean;
  onOpenPreview: () => void;
  onReconnect: () => void;
  onReset: () => void;
  onEndSession: () => void;
  onClearPreview: () => void;
}

const statusLabels: Record<DemoConnection, string> = {
  connecting: "Connecting",
  live: "Live",
  offline: "Offline",
  preview: "Preview",
};

export function DemoControls(props: DemoControlsProps) {
  const busy = props.generating || props.actionPending;
  const live = props.connection === "live";
  const preview = props.connection === "preview";
  const privacyText = live
    ? "Private demo: Your conversation uses a temporary profile and never accesses Akane’s private owner memory."
    : preview
      ? "Preview: No messages are sent to Akane."
      : props.connection === "offline"
        ? "Akane is offline. No messages can be sent."
        : "Checking the live demo connection.";

  return <div className="demo-controls demo-panel">
    <div className="status-pill" aria-label={`Demo status: ${statusLabels[props.connection]}`} aria-live="polite"><i className={`demo-status-dot ${props.connection}`} aria-hidden="true" />{statusLabels[props.connection]}</div>
    {props.connection === "offline" && <div className="demo-mode-note"><span>Akane is offline right now.</span><button type="button" onClick={props.onOpenPreview}>Try Preview</button></div>}
    {preview && <span className="demo-mode-note">Preview responses are simulated and are not sent to Akane.</span>}
    {live && !props.guestEnabled && <span className="demo-mode-note">Guest messages are unavailable right now.</span>}
    <details className="demo-options">
      <summary aria-label="Open conversation options"><span aria-hidden="true">•••</span> Options</summary>
      <div className="demo-options-body">
        <p>{privacyText}</p>
        <div className="demo-control-actions">
          {live && props.activeSession && <button className="demo-control-button" type="button" disabled={busy} onClick={props.onReset}>New conversation</button>}
          {live && props.activeSession && <button className="demo-control-button danger" type="button" disabled={busy} onClick={props.onEndSession}>End guest session</button>}
          {live && <button className="demo-control-button" type="button" disabled={busy} onClick={props.onOpenPreview}>Use Preview</button>}
          {preview && <button className="demo-control-button" type="button" disabled={busy} onClick={props.onClearPreview}>Clear Preview</button>}
          {props.canReconnect && <button className="demo-control-button" type="button" disabled={busy} onClick={props.onReconnect}>Reconnect</button>}
        </div>
        {props.retryExhausted && props.canReconnect && <p className="demo-retry-note">Automatic retries have finished. Reconnect remains available.</p>}
      </div>
    </details>
  </div>;
}
