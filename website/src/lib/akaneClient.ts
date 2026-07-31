import { projectConfig } from "../config/project";
import type { AkanePresentationState } from "../presentation";

export type PublicHealthStatus = "available" | "busy" | "offline";

export interface PublicHealth {
  status: PublicHealthStatus;
  streaming: boolean;
  guestEnabled: boolean;
}

export interface PublicSession {
  profileType: "guest";
  sessionToken: string;
  expiresAt: number;
}

export interface AkaneStreamCallbacks {
  onStart?: () => void;
  onDelta: (text: string) => void;
  onDone?: () => void;
  onPresentation?: (state: AkanePresentationState) => void;
}

export class PublicApiError extends Error {
  constructor(public readonly code: string, message: string, public readonly status = 0) {
    super(message);
    this.name = "PublicApiError";
  }
}

function apiUrl(path: string) {
  if (!projectConfig.apiUrl) {
    throw new PublicApiError("offline", "Live Akane is not configured right now.");
  }
  return `${projectConfig.apiUrl}${path}`;
}

function requestHeaders(token?: string, streaming = false) {
  return {
    Accept: streaming ? "application/x-ndjson" : "application/json",
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "1",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function defaultErrorCode(status: number) {
  if (status === 401 || status === 403) return "unauthorized";
  if (status === 408 || status === 504) return "generation_timeout";
  if (status === 429) return "rate_limited";
  if (status === 503) return "model_unavailable";
  return status >= 500 ? "internal_error" : "invalid_request";
}

async function responseError(response: Response) {
  let code = defaultErrorCode(response.status);
  let message = response.status >= 500
    ? "Live Akane is temporarily unavailable."
    : "Akane could not accept that request.";
  try {
    const payload: unknown = await response.json();
    const error = isRecord(payload) && isRecord(payload.error) ? payload.error : null;
    if (error && typeof error.code === "string") code = error.code;
    if (error && typeof error.message === "string" && error.message.trim()) message = error.message;
  } catch {
    // Keep the safe status-derived message when the response is not JSON.
  }
  return new PublicApiError(code, message, response.status);
}

async function apiFetch(path: string, init: RequestInit) {
  try {
    return await fetch(apiUrl(path), init);
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new PublicApiError("offline", "Live Akane is temporarily offline.");
  }
}

async function requestJson(
  path: string,
  method: "GET" | "POST" | "DELETE",
  options: { token?: string; body?: unknown; signal?: AbortSignal } = {},
) {
  const response = await apiFetch(path, {
    method,
    headers: requestHeaders(options.token),
    ...(options.body === undefined ? {} : { body: JSON.stringify(options.body) }),
    signal: options.signal,
  });
  if (!response.ok) throw await responseError(response);
  try {
    const payload: unknown = await response.json();
    if (!isRecord(payload)) throw new Error("Invalid JSON object.");
    return payload;
  } catch {
    throw new PublicApiError("invalid_response", "Akane sent an invalid response.", response.status);
  }
}

function parseSession(payload: Record<string, unknown>, fallbackToken = ""): PublicSession {
  const sessionToken = typeof payload.session_token === "string"
    ? payload.session_token.trim()
    : fallbackToken;
  if (payload.profile_type !== "guest" || !sessionToken || typeof payload.expires_at !== "number") {
    throw new PublicApiError("invalid_response", "Akane sent invalid session details.");
  }
  return { profileType: "guest", sessionToken, expiresAt: payload.expires_at };
}

function streamError(event: Record<string, unknown>) {
  const error = isRecord(event.error) ? event.error : null;
  const code = error && typeof error.code === "string" ? error.code : "internal_error";
  const message = error && typeof error.message === "string" && error.message.trim()
    ? error.message
    : "Akane could not finish that reply.";
  return new PublicApiError(code, message);
}

function presentationState(event: Record<string, unknown>): AkanePresentationState {
  const activities = new Set(["idle", "connecting", "listening", "thinking", "speaking", "interrupted", "offline"]);
  const expressions = new Set(["neutral", "calm", "curious", "amused", "happy", "concerned", "sad", "irritated"]);
  const mouthLevel = event.mouthLevel === undefined ? 0 : event.mouthLevel;
  const lookTarget = isRecord(event.lookTarget) ? event.lookTarget : undefined;
  if (!activities.has(String(event.activity)) || !expressions.has(String(event.expression))
      || typeof mouthLevel !== "number" || mouthLevel < 0 || mouthLevel > 1
      || (lookTarget && (typeof lookTarget.x !== "number" || typeof lookTarget.y !== "number"))) {
    throw new PublicApiError("invalid_response", "Akane sent an invalid presentation event.");
  }
  return {
    activity: event.activity as AkanePresentationState["activity"],
    expression: event.expression as AkanePresentationState["expression"],
    mouthLevel,
    ...(lookTarget ? { lookTarget: { x: lookTarget.x as number, y: lookTarget.y as number } } : {}),
  };
}

export const akaneClient = {
  async health(signal?: AbortSignal): Promise<PublicHealth> {
    const payload = await requestJson("/api/public/health", "GET", { signal });
    if (payload.status !== "available" && payload.status !== "busy" && payload.status !== "offline") {
      throw new PublicApiError("invalid_response", "Akane sent an invalid health response.");
    }
    return {
      status: payload.status,
      streaming: payload.streaming === true,
      guestEnabled: payload.guest_enabled === true,
    };
  },

  async createSession(): Promise<PublicSession> {
    return parseSession(await requestJson("/api/public/session", "POST"));
  },

  async revalidateSession(token: string): Promise<PublicSession> {
    return parseSession(await requestJson("/api/public/session", "POST", { token }), token);
  },

  async streamChat(
    token: string,
    message: string,
    callbacks: AkaneStreamCallbacks,
    signal?: AbortSignal,
  ) {
    const response = await apiFetch("/api/public/chat", {
      method: "POST",
      headers: requestHeaders(token, true),
      body: JSON.stringify({ message }),
      signal,
    });
    if (!response.ok) throw await responseError(response);
    if (!response.body) {
      throw new PublicApiError("invalid_response", "Akane did not provide a response stream.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let started = false;
    let completed = false;

    const handleLine = (line: string) => {
      if (!line.trim()) return;
      let event: unknown;
      try { event = JSON.parse(line); }
      catch { throw new PublicApiError("invalid_response", "Akane sent an invalid stream event."); }
      if (!isRecord(event) || typeof event.type !== "string") {
        throw new PublicApiError("invalid_response", "Akane sent an invalid stream event.");
      }
      if (event.type === "error") throw streamError(event);
      if (event.type === "start") {
        if (started) throw new PublicApiError("invalid_response", "Akane started the response twice.");
        started = true;
        callbacks.onStart?.();
        return;
      }
      if (event.type === "delta") {
        if (!started || completed || typeof event.text !== "string") {
          throw new PublicApiError("invalid_response", "Akane sent an invalid response delta.");
        }
        callbacks.onDelta(event.text);
        return;
      }
      if (event.type === "done") {
        if (!started || completed) {
          throw new PublicApiError("invalid_response", "Akane sent an invalid completion event.");
        }
        completed = true;
        callbacks.onDone?.();
        return;
      }
      if (event.type === "presentation") {
        if (!started || completed) {
          throw new PublicApiError("invalid_response", "Akane sent an invalid presentation event.");
        }
        callbacks.onPresentation?.(presentationState(event));
        return;
      }
      throw new PublicApiError("invalid_response", "Akane sent an unknown stream event.");
    };

    try {
      while (!completed) {
        const result = await reader.read();
        buffer += decoder.decode(result.value || new Uint8Array(), { stream: !result.done });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) handleLine(line);
        if (result.done) {
          if (buffer.trim()) handleLine(buffer);
          break;
        }
      }
      if (!completed) throw new PublicApiError("offline", "Akane's response was interrupted.");
    } finally {
      if (!completed) await reader.cancel().catch(() => undefined);
      reader.releaseLock();
    }
  },

  async resetConversation(token: string) {
    await requestJson("/api/public/session/reset", "POST", { token });
  },

  async deleteSession(token: string) {
    await requestJson("/api/public/session", "DELETE", { token });
  },
};
