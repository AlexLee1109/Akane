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

  return <aside className="demo-controls demo-panel" aria-labelledby="demo-controls-title">
    {props.connection === "connecting" && <>
      <header className="demo-control-header"><h2 id="demo-controls-title">Preparing the demo</h2></header>
      <div className="demo-control-body"><p>Checking the configured Raspberry Pi connection.</p></div>
    </>}

    {props.connection === "live" && props.activeSession && <>
      <header className="demo-control-header"><h2 id="demo-controls-title">Temporary guest session active</h2></header>
      <div className="demo-control-body">
        <p>Your conversation is temporary and isolated from the owner profile.</p>
        <div className="demo-control-actions">
          <button className="demo-control-button" type="button" disabled={busy} onClick={props.onReset}>Reset</button>
          <button className="demo-control-button danger" type="button" disabled={busy} onClick={props.onEndSession}>End session</button>
        </div>
      </div>
    </>}

    {props.connection === "live" && !props.activeSession && <>
      <header className="demo-control-header"><h2 id="demo-controls-title">Start a guest session</h2></header>
      <div className="demo-control-body">
        <p>Guest sessions are temporary and isolated from the owner profile.</p>
        <div className="demo-control-actions">
          <button className="demo-control-button primary" type="button" disabled={busy || !props.guestEnabled} onClick={props.onStartGuest}>Start guest session</button>
          <button className="demo-control-button" type="button" disabled={busy} onClick={props.onOpenPreview}>Use prerecorded Preview</button>
        </div>
      </div>
    </>}

    {props.connection === "showcase" && <>
      <header className="demo-control-header"><h2 id="demo-controls-title">Prerecorded Preview</h2></header>
      <div className="demo-control-body">
        <p>Messages are not sent to Akane or saved.</p>
        <div className="demo-control-actions">
          <button className="demo-control-button" type="button" disabled={busy} onClick={props.onClearPreview}>Clear preview</button>
          {props.canReconnect && <button className="demo-control-button primary" type="button" disabled={busy} onClick={props.onReconnect}>Reconnect</button>}
        </div>
        {props.retryExhausted && props.canReconnect && <p className="demo-retry-note">Automatic retries have finished. Manual reconnect remains available.</p>}
      </div>
    </>}
  </aside>;
}
