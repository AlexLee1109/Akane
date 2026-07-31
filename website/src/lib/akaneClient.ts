import { projectConfig } from "../config/project";

export interface AkaneStreamCallbacks {
  onStart?: () => void;
  onToken: (token: string) => void;
  onComplete?: () => void;
  onError?: (error: Error) => void;
}

export interface AkaneClient {
  sendMessage(sessionId: string, message: string, callbacks: AkaneStreamCallbacks, signal?: AbortSignal): Promise<void>;
  cancel(sessionId: string): Promise<void>;
}

const mockReplies = [
  "I'm Akane, a local-first companion. I keep my character and ongoing context in the backend that hosts me, rather than inside this static demo.",
  "Right now, I'm here for a conversation. Outside a turn, Akane can maintain an offscreen activity and decide whether there is a grounded reason to reach out.",
  "One opinion I genuinely have is that small, well-kept rituals make technology feel more humane. A quiet check-in can matter more than a very clever notification.",
];

function mockStream(message: string, callbacks: AkaneStreamCallbacks, signal?: AbortSignal) {
  const reply = mockReplies[Math.abs([...message].reduce((sum, character) => sum + character.charCodeAt(0), 0)) % mockReplies.length];
  callbacks.onStart?.();
  return new Promise<void>((resolve, reject) => {
    let index = 0;
    const timer = window.setInterval(() => {
      if (signal?.aborted) {
        window.clearInterval(timer);
        reject(new DOMException("Request cancelled.", "AbortError"));
        return;
      }
      callbacks.onToken(reply.slice(index, index + 4));
      index += 4;
      if (index >= reply.length) {
        window.clearInterval(timer);
        callbacks.onComplete?.();
        resolve();
      }
    }, 28);
  });
}

export const akaneClient: AkaneClient = {
  async sendMessage(sessionId, message, callbacks, signal) {
    if (!projectConfig.apiUrl) return mockStream(message, callbacks, signal);
    const response = await fetch(`${projectConfig.apiUrl.replace(/\/$/, "")}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId, conversation_id: sessionId, profile_id: sessionId, source: "web" }),
      signal,
    });
    if (!response.ok || !response.body) throw new Error(response.status === 503 ? "Akane is busy. Please try again in a moment." : "The Akane backend is unavailable.");
    callbacks.onStart?.();
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        let event: { type?: string; content?: string; error?: string };
        try { event = JSON.parse(line); } catch { throw new Error("The backend sent an invalid response."); }
        if (event.type === "delta") callbacks.onToken(event.content || "");
        if (event.type === "error") throw new Error(event.error || "Akane could not finish that reply.");
      }
      if (done) break;
    }
    callbacks.onComplete?.();
  },
  async cancel(sessionId) {
    if (!projectConfig.apiUrl) return;
    await fetch(`${projectConfig.apiUrl.replace(/\/$/, "")}/api/chat/cancel`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: sessionId, profile_id: sessionId }),
    });
  },
};
