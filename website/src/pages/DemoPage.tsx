import { useEffect, useRef, useState } from "react";
import { AkaneStage } from "../components/AkaneStage";
import { DemoComposer } from "../components/demo/DemoComposer";
import { DemoControls } from "../components/demo/DemoControls";
import { DemoConversation, type DemoMessage } from "../components/demo/DemoConversation";
import { projectConfig } from "../config/project";
import { akaneClient, PublicApiError, type PublicHealth, type PublicSession } from "../lib/akaneClient";
import { clearGuestToken, getGuestToken, storeGuestToken } from "../lib/session";
import type { AkanePresentationState } from "../presentation";
import "./demo.css";

type ConnectionState = "connecting" | "live" | "showcase";

const characterAsset = `${projectConfig.basePath}assets/akane-hero.png`;
const previewReplies = [
  "You made it. I was hoping we’d get a little time to talk—what’s been on your mind today?",
  "That sounds like something worth holding onto. Tell me the part you don’t want to forget.",
  "We can take it one step at a time. What would make today feel a little lighter?",
];
const retryDelays = [5_000, 15_000, 30_000, 60_000];

function currentTime() {
  return new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

export function DemoPage() {
  const initialConnection: ConnectionState =
    projectConfig.demoMode === "showcase" || !projectConfig.apiUrl ? "showcase" : "connecting";
  const [connection, setConnection] = useState<ConnectionState>(initialConnection);
  const [health, setHealth] = useState<PublicHealth | null>(null);
  const [messages, setMessages] = useState<DemoMessage[]>([]);
  const [input, setInput] = useState("");
  const [activeSession, setActiveSession] = useState<PublicSession | null>(null);
  const activeSessionRef = useRef<PublicSession | null>(null);
  const [backendPresentation, setBackendPresentation] = useState<AkanePresentationState>();
  const [generating, setGenerating] = useState(false);
  const [actionPending, setActionPending] = useState(false);
  const [error, setError] = useState("");
  const [retryExhausted, setRetryExhausted] = useState(false);
  const [reconnectKey, setReconnectKey] = useState(0);
  const aborter = useRef<AbortController | null>(null);

  function activateSession(session: PublicSession | null) {
    activeSessionRef.current = session;
    setActiveSession(session);
  }

  useEffect(() => {
    if (projectConfig.demoMode === "showcase" || !projectConfig.apiUrl) {
      setConnection("showcase");
      setHealth(null);
      return;
    }

    const controller = new AbortController();
    let stopped = false;
    let timer: number | undefined;
    setConnection("connecting");
    setRetryExhausted(false);

    async function probe(attempt: number) {
      try {
        const result = await akaneClient.health(controller.signal);
        if (stopped) return;
        if (result.status === "offline" || !result.streaming) {
          throw new PublicApiError("model_unavailable", "Live Akane is still starting up.");
        }

        const currentToken = activeSessionRef.current?.sessionToken || getGuestToken();
        if (currentToken) {
          try {
            const resumed = await akaneClient.revalidateSession(currentToken);
            activateSession(resumed);
            setMessages([]);
          } catch (cause) {
            if (cause instanceof PublicApiError && ["session_expired", "unauthorized"].includes(cause.code)) {
              clearGuestToken();
              activateSession(null);
              setMessages([]);
              setError("That temporary guest session expired. Start a new guest session to continue.");
            } else {
              throw cause;
            }
          }
        }

        setMessages([]);
        setBackendPresentation(undefined);
        setHealth(result);
        setConnection("live");
        setRetryExhausted(false);
      } catch (cause) {
        if (stopped || (cause as Error).name === "AbortError") return;
        setHealth(null);
        setConnection("showcase");
        if (attempt < retryDelays.length) {
          timer = window.setTimeout(() => { void probe(attempt + 1); }, retryDelays[attempt]);
        } else {
          setRetryExhausted(true);
        }
      }
    }

    void probe(0);
    return () => {
      stopped = true;
      controller.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [reconnectKey]);

  function reconnect() {
    if (projectConfig.demoMode !== "live" || !projectConfig.apiUrl) return;
    setError("");
    setReconnectKey(current => current + 1);
  }

  function availabilityFailed(cause: unknown) {
    return cause instanceof PublicApiError
      && !["busy", "queue_full", "rate_limited", "generation_timeout"].includes(cause.code)
      && (["offline", "model_unavailable", "invalid_response"].includes(cause.code) || cause.status >= 500);
  }

  function openPreview() {
    setConnection("showcase");
    setMessages([]);
    setBackendPresentation(undefined);
    setError("");
  }

  function clearPreview() {
    setMessages([]);
    setBackendPresentation(undefined);
    setError("");
  }

  async function createOrRevalidateGuest() {
    const storedToken = getGuestToken();
    let session: PublicSession;
    try {
      session = storedToken
        ? await akaneClient.revalidateSession(storedToken)
        : await akaneClient.createSession();
    } catch (cause) {
      if (!(cause instanceof PublicApiError) || !["session_expired", "unauthorized"].includes(cause.code)) {
        throw cause;
      }
      clearGuestToken();
      session = await akaneClient.createSession();
    }
    storeGuestToken(session.sessionToken);
    activateSession(session);
    return session;
  }

  async function startGuest() {
    if (connection !== "live" || generating || actionPending) return;
    setActionPending(true);
    setError("");
    try {
      await createOrRevalidateGuest();
      setMessages([]);
    } catch (cause) {
      setError((cause as Error).message);
      if (availabilityFailed(cause)) reconnect();
    } finally {
      setActionPending(false);
    }
  }

  async function resetConversation() {
    if (generating) return;
    aborter.current?.abort();
    setError("");
    if (!activeSession) {
      setMessages([]);
      setBackendPresentation(undefined);
      return;
    }
    if (connection !== "live") {
      setError("Reconnect to live Akane before resetting this conversation.");
      return;
    }
    setActionPending(true);
    try {
      await akaneClient.resetConversation(activeSession.sessionToken);
      setMessages([]);
      setBackendPresentation(undefined);
    } catch (cause) {
      setError((cause as Error).message);
      if (availabilityFailed(cause)) reconnect();
    } finally {
      setActionPending(false);
    }
  }

  async function endGuestSession() {
    if (!activeSession || actionPending) return;
    setActionPending(true);
    setError("");
    try {
      if (connection === "live") await akaneClient.deleteSession(activeSession.sessionToken);
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      clearGuestToken();
      activateSession(null);
      setMessages([]);
      setBackendPresentation(undefined);
      setActionPending(false);
    }
  }

  function sendPreview(message: string) {
    const reply = previewReplies[Math.abs([...message].reduce((sum, character) => sum + character.charCodeAt(0), 0)) % previewReplies.length];
    const now = currentTime();
    setMessages(current => [
      ...current,
      { id: crypto.randomUUID(), role: "you", text: message, time: now },
      { id: crypto.randomUUID(), role: "akane", text: reply, time: now, preview: true },
    ]);
  }

  async function renewExpiredGuest() {
    clearGuestToken();
    const session = await akaneClient.createSession();
    storeGuestToken(session.sessionToken);
    activateSession(session);
    setMessages([]);
    setBackendPresentation(undefined);
    setError("The temporary guest session expired, so a fresh guest session is ready. Please send that message again.");
  }

  async function send() {
    const message = input.trim();
    if (!message || generating || actionPending) return;
    if (message.length > 750) {
      setError("Keep messages at or under 750 characters.");
      return;
    }
    if (connection === "connecting") {
      setError("Akane is still connecting. You can send once live mode or Preview Mode is ready.");
      return;
    }
    if (connection === "showcase") {
      setInput("");
      setError("");
      sendPreview(message);
      return;
    }
    let session = activeSession;
    if (!session) {
      if (health?.guestEnabled !== true) {
        setError("Temporary guest sessions are not available right now.");
        return;
      }
      setActionPending(true);
      setError("");
      try {
        session = await createOrRevalidateGuest();
      } catch (cause) {
        setError((cause as Error).message);
        if (availabilityFailed(cause)) reconnect();
        return;
      } finally {
        setActionPending(false);
      }
    }

    setInput("");
    setError("");
    setGenerating(true);
    const now = currentTime();
    const replyId = crypto.randomUUID();
    setMessages(current => [
      ...current,
      { id: crypto.randomUUID(), role: "you", text: message, time: now },
      { id: replyId, role: "akane", text: "", time: now },
    ]);
    const controller = new AbortController();
    aborter.current = controller;
    let completed = "";
    try {
      await akaneClient.streamChat(session.sessionToken, message, {
        onDelta: delta => {
          completed += delta;
          setMessages(current => current.map(item => item.id === replyId ? { ...item, text: completed } : item));
        },
        onPresentation: setBackendPresentation,
      }, controller.signal);
    } catch (cause) {
      setMessages(current => current.filter(item => item.id !== replyId));
      if ((cause as Error).name !== "AbortError") {
        if (cause instanceof PublicApiError && cause.code === "session_expired") {
          try { await renewExpiredGuest(); }
          catch (renewalError) {
            clearGuestToken();
            activateSession(null);
            setMessages([]);
            setError((renewalError as Error).message);
          }
        } else {
          setError((cause as Error).message);
          if (availabilityFailed(cause)) reconnect();
        }
      }
    } finally {
      aborter.current = null;
      setGenerating(false);
      setBackendPresentation(undefined);
    }
  }

  const previewMode = connection === "showcase";
  const needsSession = connection === "live" && !activeSession;
  const inputDisabled = generating || actionPending || connection === "connecting"
    || (needsSession && health?.guestEnabled !== true);
  const connectionLabel = connection === "connecting"
    ? "Connecting"
    : previewMode
      ? "Preview Mode · Prerecorded"
      : health?.status === "busy"
        ? "Live · Busy"
        : activeSession
          ? "Live · Temporary guest session"
          : "Live · No guest session";
  const lastMessage = messages.at(-1);
  const responseText = lastMessage?.role === "akane" && lastMessage.text.trim() ? lastMessage.text : undefined;
  const isThinking = generating && lastMessage?.role === "akane" && !lastMessage.text.trim();
  const composerPlaceholder = connection === "connecting"
    ? "Connecting to Akane…"
    : needsSession
      ? "Message Akane to begin a guest session…"
      : previewMode
        ? "Message the prerecorded preview…"
        : "Message Akane…";

  return <main className="demo-page">
    <section className="demo-intro shell" aria-labelledby="demo-title">
      <div><p className="demo-eyebrow">Meet Akane</p><h1 id="demo-title">Take a moment. Say hello.</h1><p>Live conversations run through the local companion on a Raspberry Pi. When it is unavailable, you can explore an honest prerecorded preview.</p></div>
      <div className="demo-header-status" aria-label="Demo status" aria-live="polite"><span className={`demo-status-dot ${connection}`} aria-hidden="true" /><div><small>Current mode</small><strong>{connectionLabel}</strong></div></div>
    </section>

    <section className="demo-workspace shell" aria-label="Akane demo workspace">
      <AkaneStage
        imageSrc={characterAsset}
        responseText={responseText}
        connection={connection}
        generating={generating}
        hasResponseText={Boolean(responseText)}
        isThinking={Boolean(isThinking)}
        backendPresentation={backendPresentation}
      />
      <DemoConversation messages={messages} generating={generating} />
      <DemoComposer
        value={input}
        placeholder={composerPlaceholder}
        disabled={inputDisabled}
        generating={generating}
        error={error}
        onChange={setInput}
        onSend={() => { void send(); }}
        onStop={() => { aborter.current?.abort(); }}
      />
      <DemoControls
        connection={connection}
        guestEnabled={health?.guestEnabled === true}
        activeSession={Boolean(activeSession)}
        generating={generating}
        actionPending={actionPending}
        retryExhausted={retryExhausted}
        canReconnect={previewMode && projectConfig.demoMode === "live" && Boolean(projectConfig.apiUrl)}
        onStartGuest={() => { void startGuest(); }}
        onOpenPreview={openPreview}
        onReconnect={reconnect}
        onReset={() => { void resetConversation(); }}
        onEndSession={() => { void endGuestSession(); }}
        onClearPreview={clearPreview}
      />
    </section>

    <p className="demo-privacy-note shell"><strong>Private guest boundary:</strong> Live messages use a temporary guest profile isolated from the owner profile. Preview Mode is prerecorded and sends nothing to Akane.</p>
  </main>;
}
