import type { ReactNode } from "react";

type DemoConnection = "connecting" | "live" | "showcase";

interface DemoControlsProps {
  connection: DemoConnection;
  connectionLabel: string;
  guestEnabled: boolean;
  activeSession: boolean;
  generating: boolean;
  actionPending: boolean;
  retryExhausted: boolean;
  canReconnect: boolean;
  onStartGuest: () => void;
  onOpenPreview: () => void;
  onClearPreview: () => void;
  onReconnect: () => void;
  onReset: () => void;
  onEndSession: () => void;
}

function ControlIcon({ children }: { children: ReactNode }) {
  return <svg className="demo-control-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{children}</svg>;
}

export function DemoControls(props: DemoControlsProps) {
  const busy = props.generating || props.actionPending;

  return <aside className="demo-controls demo-panel" aria-labelledby="demo-controls-title">
    <header className="demo-panel-header">
      <div><span>Live controls</span><h2 id="demo-controls-title">Session controls</h2></div>
    </header>

    <section className="demo-control-group" aria-labelledby="connection-controls-title">
      <div className="demo-control-heading">
        <ControlIcon><path d="M4 12a8 8 0 0 1 13.7-5.6M20 12a8 8 0 0 1-13.7 5.6" /><path d="m17 3 .7 3.4-3.4.7M7 21l-.7-3.4 3.4-.7" /></ControlIcon>
        <div><span>Connection</span><h3 id="connection-controls-title">{props.connectionLabel}</h3></div>
      </div>
      {props.connection === "connecting" && <p>Checking the configured Raspberry Pi connection.</p>}
      {props.connection === "live" && <p>Connected to the real Akane runtime on the Raspberry Pi.</p>}
      {props.connection === "showcase" && <p><strong>Preview Mode uses prerecorded responses.</strong> Messages are not sent to Akane and are not saved.</p>}
      {props.connection === "showcase" && <button className="demo-control-button" type="button" onClick={props.onClearPreview} aria-label="Clear the prerecorded preview conversation">Clear preview</button>}
      {props.canReconnect && <button className="demo-control-button" type="button" onClick={props.onReconnect} aria-label="Reconnect to live Akane">Reconnect</button>}
      {props.retryExhausted && <p className="demo-retry-note">Automatic retries have finished. Manual reconnect remains available.</p>}
    </section>

    <section className="demo-control-group" aria-labelledby="session-controls-title">
      <div className="demo-control-heading">
        <ControlIcon><circle cx="9" cy="8" r="3" /><path d="M3.5 19a5.5 5.5 0 0 1 11 0M16 9h5M18.5 6.5v5" /></ControlIcon>
        <div><span>Session</span><h3 id="session-controls-title">{props.activeSession ? "Temporary guest session active" : "No guest session"}</h3></div>
      </div>

      {props.activeSession
        ? <div className="demo-control-actions">
            <p>{props.connection === "live" ? "Your temporary continuity is active and isolated from the owner profile." : "Your live guest session is paused while Preview Mode is open."}</p>
            <button className="demo-control-button" type="button" disabled={busy || props.connection !== "live"} onClick={props.onReset} aria-label="Reset the current guest conversation">Reset conversation</button>
            <button className="demo-control-button danger" type="button" disabled={busy} onClick={props.onEndSession} aria-label="End the temporary guest session">End guest session</button>
          </div>
        : props.connection === "live"
          ? <div className="demo-control-actions">
              <p>Temporary guest sessions use real memory and relationship continuity while active, but remain isolated from the owner profile.</p>
              <button className="demo-control-button primary" type="button" disabled={busy || !props.guestEnabled} onClick={props.onStartGuest} aria-label="Start a temporary guest session">Start guest session</button>
              <button className="demo-control-button" type="button" disabled={busy} onClick={props.onOpenPreview} aria-label="Open prerecorded Preview Mode">Offline Preview</button>
            </div>
          : props.connection === "showcase"
            ? <p>Preview messages remain only in this browser view.</p>
            : <p>Session controls will be available after the connection check.</p>}
    </section>
  </aside>;
}
