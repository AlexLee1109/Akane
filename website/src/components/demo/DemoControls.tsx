type DemoConnection = "connecting" | "live" | "showcase";

interface DemoControlsProps {
  connection: DemoConnection;
  guestEnabled: boolean;
  activeSession: boolean;
  generating: boolean;
  actionPending: boolean;
  retryExhausted: boolean;
  canReconnect: boolean;
  onStartGuest: () => void;
  onOpenPreview: () => void;
  onReconnect: () => void;
  onReset: () => void;
  onEndSession: () => void;
  onClearPreview: () => void;
}

export function DemoControls(props: DemoControlsProps) {
  const busy = props.generating || props.actionPending;
  const live = props.connection === "live";
  const preview = props.connection === "showcase";

  return <details className="demo-controls demo-panel">
    <summary><span><i className={`demo-status-dot ${live ? "live" : props.connection}`} aria-hidden="true" /><strong>Session & privacy</strong></span><small>{live ? (props.activeSession ? "Temporary guest active" : "Live connection") : preview ? "Prerecorded preview" : "Checking availability"}</small></summary>
    <div className="demo-control-body">
      <section className="demo-control-section" aria-labelledby="demo-controls-title"><h2 id="demo-controls-title" className="sr-only">Session controls</h2>
        <ul className="demo-connection-list">
          <li><i className={`demo-status-dot ${live ? "live" : props.connection}`} aria-hidden="true" />{live ? "Connected" : preview ? "Preview mode" : "Connecting"}</li>
          <li>Running on Raspberry Pi</li>
          <li>{live ? "Live stream active" : preview ? "Messages stay on this page" : "Checking live stream"}</li>
        </ul>
      </section>
      <section className="demo-control-section"><h3>Guest Session</h3><p>{preview ? "Preview messages are never sent to Akane." : "Isolated temporary memory. The owner profile is never modified and this session is automatically deleted."}</p></section>
      <section className="demo-control-section"><h3>Session Actions</h3><div className="demo-control-actions">
        {live && !props.activeSession && <button className="demo-control-button primary" type="button" disabled={busy || !props.guestEnabled} onClick={props.onStartGuest}>Start Guest Session</button>}
        {live && !props.activeSession && <button className="demo-control-button" type="button" disabled={busy} onClick={props.onOpenPreview}>Use Preview</button>}
        {preview && <button className="demo-control-button" type="button" disabled={busy} onClick={props.onClearPreview}>Clear Preview</button>}
        {props.canReconnect && <button className="demo-control-button" type="button" disabled={busy} onClick={props.onReconnect}>Reconnect</button>}
        {live && <button className="demo-control-button" type="button" disabled={busy || !props.activeSession} onClick={props.onReset}>Reset Conversation</button>}
        {live && <button className="demo-control-button danger" type="button" disabled={busy || !props.activeSession} onClick={props.onEndSession}>End Guest Session</button>}
      </div>{props.retryExhausted && props.canReconnect && <p className="demo-retry-note">Automatic retries have finished. Reconnect remains available.</p>}</section>
    </div>
  </details>;
}
