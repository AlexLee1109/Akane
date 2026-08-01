import type { FormEvent, KeyboardEvent } from "react";

interface DemoComposerProps {
  value: string;
  placeholder: string;
  disabled: boolean;
  generating: boolean;
  error: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
}

export function DemoComposer(props: DemoComposerProps) {
  function submit(event: FormEvent) {
    event.preventDefault();
    props.onSend();
  }

  function keyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      props.onSend();
    }
  }

  return <form className="demo-composer demo-panel" onSubmit={submit}>
    <label className="sr-only" htmlFor="demo-message">Message Akane</label>
    <textarea
      id="demo-message"
      value={props.value}
      onChange={event => props.onChange(event.target.value)}
      onKeyDown={keyDown}
      maxLength={750}
      disabled={props.disabled}
      placeholder={props.placeholder}
      rows={2}
      aria-describedby={props.error ? "demo-composer-error" : undefined}
    />
    {props.generating
      ? <button className="demo-send-button stop" type="button" onClick={props.onStop} aria-label="Stop Akane's response"><span aria-hidden="true">■</span>Stop</button>
      : <button className="demo-send-button" type="submit" disabled={props.disabled || !props.value.trim()} aria-label="Send message to Akane">Send<span aria-hidden="true">→</span></button>}
    {props.error && <p id="demo-composer-error" className="demo-form-error" role="alert">{props.error}</p>}
  </form>;
}
