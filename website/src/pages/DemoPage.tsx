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

type ConnectionState = "connecting" | "live" | "offline" | "preview";

const characterAsset = `${projectConfig.basePath}assets/akane-standing.webp`;
const previewReplies = [
  "Hey. Nice to meet you.",
  "Hmm, I think I’d pick ramen.",
  "That sounds kind of fun, actually.",
  "I’m not sure yet, but I have a feeling I’d like it.",
  "Okay, yeah. I can see the appeal.",
];
const retryDelays = [5_000, 15_000, 30_000, 60_000];

function currentTime() {
  return new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

export function DemoPage() {
  const initialConnection: ConnectionState =
    projectConfig.demoMode === "showcase" || !projectConfig.apiUrl ? "preview" : "connecting";
  const [connection, setConnection] = useState<ConnectionState>(initialConnection);
  const [previewSelected, setPreviewSelected] = useState(initialConnection === "preview");
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

  useEffect(() => () => aborter.current?.abort(), []);

  function activateSession(session: PublicSession | null) {
    activeSessionRef.current = session;
    setActiveSession(session);
  }

  useEffect(() => {
    if (previewSelected || projectConfig.demoMode === "showcase" || !projectConfig.apiUrl) {
      setConnection("preview");
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
            if (stopped) return;
            activateSession(resumed);
          } catch (cause) {
            if (stopped) return;
            if (cause instanceof PublicApiError && ["session_expired", "unauthorized"].includes(cause.code)) {
              clearGuestToken();
              activateSession(null);
              setMessages([]);
              setError("That temporary conversation expired. Send a message to begin a new one.");
            } else {
              throw cause;
            }
          }
        }

        setBackendPresentation(undefined);
        setHealth(result);
        setConnection("live");
        setRetryExhausted(false);
      } catch (cause) {
        if (stopped || (cause as Error).name === "AbortError") return;
        setHealth(null);
        setConnection("offline");
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
  }, [reconnectKey, previewSelected]);

  function reconnect(clearError = true) {
    if (projectConfig.demoMode !== "live" || !projectConfig.apiUrl) return;
    if (clearError) setError("");
    if (previewSelected) setMessages([]);
    setPreviewSelected(false);
    setReconnectKey(current => current + 1);
  }

  function availabilityFailed(cause: unknown) {
    return cause instanceof PublicApiError
      && !["busy", "queue_full", "rate_limited", "generation_timeout"].includes(cause.code)
      && (["offline", "model_unavailable", "invalid_response"].includes(cause.code) || cause.status >= 500);
  }

  function openPreview() {
    setPreviewSelected(true);
    setConnection("preview");
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
      if (availabilityFailed(cause)) reconnect(false);
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
      setError("Akane is still connecting. You can send once Live or Preview is ready.");
      return;
    }
    if (connection === "offline") {
      setError("Akane is offline right now. Try Preview instead.");
      return;
    }
    if (connection === "preview") {
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
        if (availabilityFailed(cause)) reconnect(false);
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
      if ((cause as Error).name === "AbortError") {
        setMessages(current => completed.trim()
          ? current.map(item => item.id === replyId ? { ...item, interrupted: true } : item)
          : current.filter(item => item.id !== replyId));
      } else {
        setMessages(current => current.filter(item => item.id !== replyId));
        setInput(message);
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
          if (availabilityFailed(cause)) reconnect(false);
        }
      }
    } finally {
      aborter.current = null;
      setGenerating(false);
      setBackendPresentation(undefined);
    }
  }

  const previewMode = connection === "preview";
  const needsSession = connection === "live" && !activeSession;
  const inputDisabled = generating || actionPending || connection === "connecting" || connection === "offline"
    || (needsSession && health?.guestEnabled !== true);
  const lastMessage = messages.at(-1);
  const responseText = lastMessage?.role === "akane" && lastMessage.text.trim() ? lastMessage.text : undefined;
  const composerPlaceholder = connection === "connecting"
    ? "Connecting to Akane…"
    : connection === "offline"
      ? "Akane is offline right now."
      : previewMode
        ? "Try a simulated message…"
        : needsSession
          ? "Message Akane to begin…"
          : "Message Akane…";

  return <main id="main-content" tabIndex={-1} className="demo-page">
    <section className="demo-intro shell" aria-labelledby="demo-title">
      <p className="eyebrow">A little space to connect</p><h1 id="demo-title">Say hello to Akane.</h1><p>{previewMode ? "Explore the interface with clearly labeled simulated replies." : "Your first message begins a private, temporary guest conversation."}</p>
    </section>

    <section className="demo-workspace shell" aria-label="Akane demo workspace">
      <AkaneStage
        imageSrc={characterAsset}
        connection={connection}
        generating={generating}
        hasResponseText={Boolean(responseText)}
        backendPresentation={backendPresentation}
      />
      <div className="demo-chat-panel">
      <DemoConversation messages={messages} generating={generating} connection={connection} onSuggestion={setInput} />
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
        canReconnect={connection !== "live" && projectConfig.demoMode === "live" && Boolean(projectConfig.apiUrl)}
        onOpenPreview={openPreview}
        onReconnect={() => reconnect()}
        onReset={() => { void resetConversation(); }}
        onEndSession={() => { void endGuestSession(); }}
        onClearPreview={clearPreview}
      />
      </div>
    </section>

  </main>;
}
