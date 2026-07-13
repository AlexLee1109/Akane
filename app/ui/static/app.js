const SEARCH_PARAMS = new URLSearchParams(window.location.search);
const POPUP_ROLE = SEARCH_PARAMS.get("popup_role") || "full";
const SESSION_ID = SEARCH_PARAMS.get("session_id") || "popup";

const elements = {
  messages:
    document.getElementById("messages") ||
    document.getElementById("speechBody"),
  speechBubble: document.getElementById("speechBubble"),
  characterWrap: document.getElementById("characterWrap"),
  characterArt: document.getElementById("characterArt"),
  composer:
    document.getElementById("composer") ||
    document.getElementById("composerWrapper"),
  input: document.getElementById("messageInput"),
  send: document.getElementById("sendButton"),
  notice: document.getElementById("notice"),
  statusDot:
    document.getElementById("statusDot") ||
    document.querySelector(".status-dot"),
  statusText: document.getElementById("statusText"),
  minimizeButton: document.getElementById("minimizeButton"),
  closeButton: document.getElementById("closeButton"),
  messageActionButton: document.getElementById("messageActionButton"),
  backendForm: document.getElementById("backendForm"),
  backendSelect: document.getElementById("backendSelect"),
  modelInput: document.getElementById("modelInput"),
  applyBackendButton: document.getElementById("applyBackendButton"),
  emptyState: document.getElementById("emptyState"),
  memoryActionButton: document.getElementById("memoryActionButton"),
  memoryCloseBtn: document.getElementById("memoryCloseBtn"),
  memoryPanel: document.getElementById("memoryPanel"),
  memoryContent: document.getElementById("memoryContent"),
};

let currentMessages = [];
let isSending = false;
let statusPoll = null;
let preservePreview = false;
let transientStatus = "";
let lastRenderedMessagesKey = "";
let lastKnownStateVersion = -1;

let bubbleVisible = false;
let bubbleHideTimer = null;
let bubbleStreaming = false;
let streamingAssistantText = "";
let bubbleResizeObserver = null;

let backendValues = {
  llama_cpp: "",
  openrouter: "",
};

const messageTimestampCache = new Map();
const API_BASE =
  SEARCH_PARAMS.get("api_base")?.replace(/\/$/, "") || "";

function apiUrl(path) {
  return API_BASE ? `${API_BASE}${path}` : path;
}

async function callWindowApi(methodName) {
  if (!window.pywebview?.api?.[methodName]) {
    return;
  }

  try {
    await window.pywebview.api[methodName]();
  } catch {
    showNotice("Window action failed.", "error");
  }
}

async function callWindowApiWithArg(methodName, arg) {
  if (!window.pywebview?.api?.[methodName]) {
    return;
  }

  try {
    await window.pywebview.api[methodName](arg);
  } catch {
    // Ignore optional window synchronization failures.
  }
}

function requestComposerFocus() {
  return;
}

function setComposerOpenState(open) {
  document.body.dataset.composerOpen = open ? "true" : "false";

  if (elements.composer) {
    elements.composer.hidden = !open;
  }
}

function focusComposerInput() {
  if (!elements.input) {
    return;
  }

  elements.input.disabled = false;

  window.setTimeout(() => {
    elements.input?.focus();

    try {
      const length = elements.input?.value?.length ?? 0;
      elements.input?.setSelectionRange?.(length, length);
    } catch {
      // Ignore selection errors from embedded webviews.
    }
  }, 0);
}

window.__akaneComposerShown = () => {
  setComposerOpenState(true);
  focusComposerInput();
};

window.__akaneComposerHidden = () => {
  setComposerOpenState(false);
};

function formatMemoryForDisplay(memory) {
  const sections = [];

  const user = memory?.user || {};
  const preferences = memory?.preferences || [];
  const userSection = [];

  if (user.name) {
    userSection.push(
      `<div class="memory-item"><strong>Name:</strong> ${escapeHtml(
        user.name,
      )}</div>`,
    );
  }

  for (const [key, value] of Object.entries(user)) {
    if (key === "name") {
      continue;
    }

    userSection.push(
      `<div class="memory-item"><strong>${escapeHtml(
        key,
      )}:</strong> ${escapeHtml(String(value))}</div>`,
    );
  }

  if (preferences.length > 0) {
    preferences.slice(0, 8).forEach((preference) => {
      userSection.push(
        `<div class="memory-item">${escapeHtml(
          preference.content || "",
        )}</div>`,
      );
    });
  }

  if (userSection.length > 0) {
    sections.push(`
      <div class="memory-section">
        <div class="memory-section-title">User Preferences</div>
        ${userSection.join("")}
      </div>
    `);
  }

  const activities = memory?.activities || {};
  const activeActivities = Object.entries(activities).filter(
    ([, activity]) => activity.status === "active",
  );

  if (activeActivities.length > 0) {
    const activityItems = activeActivities
      .slice(0, 5)
      .map(([name, activity]) => {
        const details = (activity.details || [])
          .slice(0, 2)
          .join("; ");

        return `
          <div class="memory-item">
            <strong>${escapeHtml(name)}</strong>
            ${details ? `: ${escapeHtml(details)}` : ""}
          </div>
        `;
      })
      .join("");

    sections.push(`
      <div class="memory-section">
        <div class="memory-section-title">
          Recent Files or Projects
        </div>
        ${activityItems}
      </div>
    `);
  }

  const facts = memory?.facts || [];

  if (facts.length > 0) {
    const factItems = facts
      .slice(0, 6)
      .map(
        (fact) =>
          `<div class="memory-item">${escapeHtml(
            fact.content || "",
          )}</div>`,
      )
      .join("");

    sections.push(`
      <div class="memory-section">
        <div class="memory-section-title">Persistent Facts</div>
        ${factItems}
      </div>
    `);
  }

  const extraPreferences = preferences.slice(8);

  if (extraPreferences.length > 0) {
    const preferenceItems = extraPreferences
      .slice(0, 4)
      .map(
        (preference) =>
          `<div class="memory-item">${escapeHtml(
            preference.content || "",
          )}</div>`,
      )
      .join("");

    sections.push(`
      <div class="memory-section">
        <div class="memory-section-title">
          Additional Preferences
        </div>
        ${preferenceItems}
      </div>
    `);
  }

  return (
    sections.join("") ||
    '<div class="memory-empty">No memories stored yet.</div>'
  );
}

function syncBubbleWindowSize() {
  if (POPUP_ROLE !== "bubble" || !elements.speechBubble) {
    return;
  }

  if (!bubbleVisible) {
    void callWindowApiWithArg("sync_bubble_size", [8, 8]);
    return;
  }

  const rect =
    elements.speechBubble.getBoundingClientRect();

  const header =
    elements.speechBubble.querySelector(
      ".speech-bubble-header",
    );

  const body =
    elements.speechBubble.querySelector(
      ".speech-bubble-body",
    );

  const computed =
    window.getComputedStyle(elements.speechBubble);

  const documentElement = document.documentElement;
  const documentBody = document.body;

  const headerRect = header?.getBoundingClientRect();
  const bodyRect = body?.getBoundingClientRect();

  const paddingTop =
    parseFloat(computed.paddingTop || "0") || 0;

  const paddingRight =
    parseFloat(computed.paddingRight || "0") || 0;

  const paddingBottom =
    parseFloat(computed.paddingBottom || "0") || 0;

  const paddingLeft =
    parseFloat(computed.paddingLeft || "0") || 0;

  const bodyBottom = bodyRect
    ? bodyRect.bottom - rect.top
    : 0;

  const headerBottom = headerRect
    ? headerRect.bottom - rect.top
    : 0;

  const measuredWidth = Math.max(
    rect.width,
    elements.speechBubble.scrollWidth,
    documentElement?.scrollWidth || 0,
    documentBody?.scrollWidth || 0,
    body?.scrollWidth || 0,
    header?.scrollWidth || 0,
    (bodyRect?.width || 0) +
      paddingLeft +
      paddingRight +
      28,
  );

  const measuredHeight = Math.max(
    rect.height,
    elements.speechBubble.scrollHeight,
    documentElement?.scrollHeight || 0,
    documentBody?.scrollHeight || 0,
    bodyBottom + paddingBottom + 84,
    headerBottom + paddingBottom + 84,
    paddingTop +
      paddingBottom +
      (headerRect?.height || 0) +
      (bodyRect?.height || 0) +
      120,
    paddingTop +
      paddingBottom +
      (header?.scrollHeight || 0) +
      (body?.scrollHeight || 0) +
      120,
  );

  const nextWidth = Math.ceil(measuredWidth);
  const nextHeight = Math.ceil(measuredHeight);

  if (
    !Number.isFinite(nextWidth) ||
    !Number.isFinite(nextHeight) ||
    nextWidth <= 0 ||
    nextHeight <= 0
  ) {
    return;
  }

  void callWindowApiWithArg(
    "sync_bubble_size",
    [nextWidth, nextHeight],
  );
}

function scheduleBubbleSync() {
  if (POPUP_ROLE !== "bubble") {
    return;
  }

  window.requestAnimationFrame(syncBubbleWindowSize);
  window.setTimeout(syncBubbleWindowSize, 40);
  window.setTimeout(syncBubbleWindowSize, 120);
  window.setTimeout(syncBubbleWindowSize, 260);
  window.setTimeout(syncBubbleWindowSize, 420);
}

function ensureBubbleResizeObserver() {
  if (
    POPUP_ROLE !== "bubble" ||
    bubbleResizeObserver ||
    !elements.speechBubble
  ) {
    return;
  }

  if (typeof ResizeObserver !== "function") {
    return;
  }

  bubbleResizeObserver = new ResizeObserver(() => {
    if (bubbleVisible) {
      syncBubbleWindowSize();
    }
  });

  bubbleResizeObserver.observe(elements.speechBubble);

  if (elements.messages) {
    bubbleResizeObserver.observe(elements.messages);
  }
}

function clearBubbleHideTimer() {
  if (!bubbleHideTimer) {
    return;
  }

  window.clearTimeout(bubbleHideTimer);
  bubbleHideTimer = null;
}

function setBubbleVisible(visible) {
  bubbleVisible = Boolean(visible);

  if (elements.speechBubble) {
    elements.speechBubble.hidden = !bubbleVisible;
  }

  scheduleBubbleSync();
}

function showBubbleNow() {
  clearBubbleHideTimer();
  bubbleStreaming = true;
  setBubbleVisible(true);
}

function hideBubbleSoon(delayMs = 0) {
  clearBubbleHideTimer();

  bubbleHideTimer = window.setTimeout(() => {
    bubbleStreaming = false;

    if (elements.messages) {
      elements.messages.innerHTML = "";
    }

    setBubbleVisible(false);
  }, delayMs);
}

function setBubbleText(text) {
  if (!elements.messages) {
    return;
  }

  const nextText = String(text || "");

  if (!nextText.trim()) {
    elements.messages.innerHTML = "";
    setBubbleVisible(false);
    return;
  }

  setBubbleVisible(true);
  elements.messages.innerHTML =
    renderMessageBody(nextText);

  scheduleBubbleSync();
}

window.__akaneSetBubbleText = setBubbleText;

window.__akanePendingMessage = "";

window.__akaneStreamEvent = (jsonLine) => {
  let event;

  try {
    event = JSON.parse(jsonLine);
  } catch {
    return;
  }

  if (
    event.type === "done" ||
    event.type === "error"
  ) {
    setSendingState(false);
  }

  handleStreamEvent(
    event,
    window.__akanePendingMessage,
  );
};

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderInlineMarkdown(text) {
  const escaped = escapeHtml(text);

  return escaped.replace(
    /\*\*(.+?)\*\*/g,
    "<strong>$1</strong>",
  );
}

function renderMessageBody(text) {
  return renderInlineMarkdown(text).replace(
    /\n/g,
    "<br>",
  );
}

function scrollMessages() {
  if (!elements.messages) {
    return;
  }

  elements.messages.scrollTop =
    elements.messages.scrollHeight;
}

function showNotice(text, tone = "neutral") {
  if (!elements.notice) {
    return;
  }

  if (!text) {
    elements.notice.hidden = true;
    elements.notice.textContent = "";
    elements.notice.dataset.tone = "neutral";
    return;
  }

  elements.notice.hidden = false;
  elements.notice.dataset.tone = tone;
  elements.notice.textContent = text;

  window.clearTimeout(showNotice.timeoutId);

  showNotice.timeoutId = window.setTimeout(
    () => showNotice(""),
    3200,
  );
}

function setStatus(model) {
  let tone = "ready";

  if (model?.error) {
    tone = "error";
  } else if (isSending) {
    tone = "busy";
  }

  if (elements.statusText) {
    elements.statusText.textContent = "";
  }

  if (elements.statusDot) {
    elements.statusDot.dataset.tone = tone;
  }
}

function setTransientStatus(label = "") {
  transientStatus = label;
}

function timestampForMessage(message, index) {
  const key =
    `${index}:${message.role}:${message.content || ""}`;

  if (messageTimestampCache.has(key)) {
    return messageTimestampCache.get(key);
  }

  const now = new Date();

  const seeded = new Date(
    now.getTime() -
      Math.max(
        currentMessages.length - index,
        0,
      ) *
        60000,
  );

  const label = seeded.toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });

  messageTimestampCache.set(key, label);

  return label;
}

function makeBubbleRow(message, index) {
  const row = document.createElement("div");

  row.className =
    `message-row ${
      message.role === "user"
        ? "message-row-user"
        : "message-row-assistant"
    }`;

  if (message.role !== "user") {
    const orb = document.createElement("div");
    orb.className = "assistant-orb";
    orb.setAttribute("aria-hidden", "true");

    const orbCore =
      document.createElement("span");

    orbCore.className = "assistant-orb-core";

    orb.append(orbCore);
    row.append(orb);
  }

  const article =
    document.createElement("article");

  article.className =
    `bubble ${
      message.role === "user"
        ? "bubble-user"
        : "bubble-assistant"
    }`;

  const body = document.createElement("div");
  body.className = "bubble-body";
  body.innerHTML = renderMessageBody(
    message.content || "",
  );

  const meta = document.createElement("div");
  meta.className = "bubble-meta";
  meta.textContent = timestampForMessage(
    message,
    index,
  );

  if (message.role === "user") {
    const check =
      document.createElement("span");

    check.className = "bubble-check";
    check.textContent = "✓";

    meta.append(check);
  }

  article.append(body, meta);
  row.append(article);

  return row;
}

function renderMessages(messages) {
  currentMessages = Array.isArray(messages)
    ? messages
    : [];

  lastRenderedMessagesKey =
    JSON.stringify(currentMessages);

  if (!elements.messages) {
    return;
  }

  if (POPUP_ROLE === "bubble") {
    const latestAssistant = [
      ...currentMessages,
    ]
      .reverse()
      .find(
        (message) =>
          message.role === "assistant",
      );

    const lastMessage =
      currentMessages[
        currentMessages.length - 1
      ];

    if (
      latestAssistant?.content &&
      (
        bubbleStreaming ||
        lastMessage?.role === "assistant"
      )
    ) {
      setBubbleText(latestAssistant.content);
    } else if (
      bubbleVisible &&
      bubbleStreaming
    ) {
      window.requestAnimationFrame(
        syncBubbleWindowSize,
      );
    } else if (!bubbleStreaming) {
      setBubbleVisible(false);
    }

    return;
  }

  if (POPUP_ROLE === "companion") {
    const latestAssistant = [
      ...currentMessages,
    ]
      .reverse()
      .find(
        (message) =>
          message.role === "assistant",
      );

    const lastMessage =
      currentMessages[
        currentMessages.length - 1
      ];

    if (
      latestAssistant?.content &&
      (
        bubbleStreaming ||
        lastMessage?.role === "assistant"
      )
    ) {
      setBubbleText(latestAssistant.content);
    }

    return;
  }

  elements.messages.innerHTML = "";

  if (!currentMessages.length) {
    if (elements.emptyState?.content) {
      elements.messages.append(
        elements.emptyState.content.cloneNode(
          true,
        ),
      );
    } else {
      const fallback =
        document.createElement("div");

      fallback.className = "empty-state";

      elements.messages.append(fallback);
    }

    return;
  }

  const fragment =
    document.createDocumentFragment();

  currentMessages.forEach(
    (message, index) => {
      fragment.append(
        makeBubbleRow(message, index),
      );
    },
  );

  elements.messages.append(fragment);
  scrollMessages();
}

function upsertStreamingMessages(
  userText,
  assistantText = "",
) {
  const next = [
    ...currentMessages,
    {
      role: "user",
      content: userText,
    },
    {
      role: "assistant",
      content: assistantText,
    },
  ];

  renderMessages(next);

  if (POPUP_ROLE === "bubble") {
    void callWindowApiWithArg(
      "push_bubble_text",
      assistantText || "",
    );
  }
}

function autosizeInput() {
  if (!elements.input) {
    return;
  }

  elements.input.style.height = "0px";

  elements.input.style.height =
    `${Math.min(
      elements.input.scrollHeight,
      120,
    )}px`;
}

function setSendingState(sending) {
  isSending = sending;

  if (!sending) {
    transientStatus = "";
  }

  if (elements.send) {
    elements.send.disabled = sending;
  }

  if (elements.applyBackendButton) {
    elements.applyBackendButton.disabled =
      sending;
  }

  if (elements.backendSelect) {
    elements.backendSelect.disabled = sending;
  }

  if (elements.modelInput) {
    elements.modelInput.disabled = sending;
  }
}

function backendPlaceholder(backend) {
  return backend === "openrouter"
    ? "OpenRouter model slug"
    : "Local GGUF path";
}

function syncBackendControls(model) {
  if (
    !elements.backendSelect ||
    !elements.modelInput
  ) {
    return;
  }

  if (model?.local_model_path) {
    backendValues.llama_cpp =
      model.local_model_path;
  }

  if (model?.openrouter_model) {
    backendValues.openrouter =
      model.openrouter_model;
  }

  const backend =
    model?.backend ||
    elements.backendSelect.value ||
    "llama_cpp";

  elements.backendSelect.value = backend;

  elements.modelInput.placeholder =
    backendPlaceholder(backend);

  if (
    document.activeElement !==
    elements.modelInput
  ) {
    elements.modelInput.value =
      backend === "openrouter"
        ? backendValues.openrouter
        : backendValues.llama_cpp;
  }
}

async function fetchState() {
  const payload = await loadStatePayload();

  renderMessages(payload.messages || []);

  lastKnownStateVersion = Number(
    payload.version || 0,
  );

  setStatus(payload.model || {});
  syncBackendControls(payload.model || {});
}

async function loadStatePayload(
  includeMessages = true,
) {
  const response = await fetch(
    apiUrl(
      `/api/state?session_id=${encodeURIComponent(
        SESSION_ID,
      )}&include_messages=${
        includeMessages ? "1" : "0"
      }`,
    ),
    {
      cache: "no-store",
    },
  );

  if (!response.ok) {
    throw new Error(
      "Could not load chat state.",
    );
  }

  return response.json();
}

async function sendMessage(message) {
  preservePreview = false;

  if (message.startsWith("/")) {
    return sendCommand(message);
  }

  setSendingState(true);

  setStatus({
    loading: false,
    loaded: true,
  });

  streamingAssistantText = "";

  upsertStreamingMessages(message, "");

  if (
    window.pywebview?.api
      ?.send_message_stream
  ) {
    window.__akanePendingMessage = message;

    try {
      await window.pywebview.api
        .send_message_stream(message);
    } catch (error) {
      showNotice(
        error?.message ||
          "Could not start streaming.",
        "error",
      );

      setSendingState(false);
    }

    return;
  }

  try {
    const response = await fetch(
      apiUrl("/api/chat/stream"),
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message,
          session_id: SESSION_ID,
          source: "popup",
        }),
      },
    );

    if (!response.ok) {
      let payload = {};

      try {
        payload = await response.json();
      } catch {
        payload = {};
      }

      throw new Error(
        payload.error ||
          payload.detail ||
          "Something went wrong.",
      );
    }

    const reader =
      response.body?.getReader();

    if (!reader) {
      throw new Error(
        "Streaming is unavailable.",
      );
    }

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const {
        value,
        done,
      } = await reader.read();

      if (done) {
        break;
      }

      buffer += decoder.decode(value, {
        stream: true,
      });

      let newlineIndex =
        buffer.indexOf("\n");

      while (newlineIndex !== -1) {
        const line = buffer
          .slice(0, newlineIndex)
          .trim();

        buffer = buffer.slice(
          newlineIndex + 1,
        );

        if (line) {
          handleStreamEvent(
            JSON.parse(line),
            message,
          );
        }

        newlineIndex =
          buffer.indexOf("\n");
      }
    }

    const finalLine = buffer.trim();

    if (finalLine) {
      handleStreamEvent(
        JSON.parse(finalLine),
        message,
      );
    }
  } finally {
    setSendingState(false);
  }
}

async function sendCommand(message) {
  setSendingState(true);

  setStatus({
    loading: false,
    loaded: true,
  });

  try {
    const response = await fetch(
      apiUrl("/api/chat"),
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message,
          session_id: SESSION_ID,
          source: "popup",
        }),
      },
    );

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(
        payload.error ||
          payload.detail ||
          "Something went wrong.",
      );
    }

    if (Array.isArray(payload.messages)) {
      renderMessages(payload.messages);
    }

    if (payload.notice) {
      showNotice(payload.notice);
    }

    if (!payload.ephemeral) {
      preservePreview = false;
      await refreshStatus();
    } else {
      preservePreview = true;
    }

    return payload;
  } finally {
    setSendingState(false);
  }
}

async function switchBackend() {
  const backend =
    elements.backendSelect.value;

  const value =
    elements.modelInput.value.trim();

  const payload = {
    backend,
  };

  if (backend === "openrouter") {
    payload.model_name =
      value || backendValues.openrouter;
  } else {
    payload.local_model_path =
      value || backendValues.llama_cpp;
  }

  setSendingState(true);

  try {
    const response = await fetch(
      apiUrl("/api/backend"),
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      },
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.error ||
          data.detail ||
          "Could not switch backend.",
      );
    }

    syncBackendControls(data.model || {});
    setStatus(data.model || {});

    const target =
      backend === "openrouter"
        ? data.model?.openrouter_model ||
          "OpenRouter"
        : data.model?.local_model_path ||
          "local model";

    showNotice(
      `Switched to ${
        backend === "openrouter"
          ? "OpenRouter"
          : "local Llama"
      }: ${target}`,
    );
  } finally {
    setSendingState(false);

    if (elements.input) {
      elements.input.disabled = false;
      elements.input.focus();
    }

    if (elements.send) {
      elements.send.disabled = false;
    }
  }
}

function handleStreamEvent(
  event,
  userMessage,
) {
  if (event.type === "start") {
    bubbleStreaming = true;
    streamingAssistantText = "";

    if (Array.isArray(event.messages)) {
      renderMessages([
        ...event.messages,
        {
          role: "assistant",
          content: "",
        },
      ]);
    }

    return;
  }

  if (event.type === "delta") {
    const incoming = String(
      event.content || "",
    );

    if (!incoming) {
      return;
    }

    showBubbleNow();

    /*
     * Support both possible server formats:
     *
     * True deltas:
     * "Hel", "lo", " there"
     *
     * Accumulated output:
     * "Hel", "Hello", "Hello there"
     */
    if (
      incoming.startsWith(
        streamingAssistantText,
      )
    ) {
      streamingAssistantText = incoming;
    } else {
      streamingAssistantText += incoming;
    }

    const next = [...currentMessages];

    const lastMessage =
      next[next.length - 1];

    if (
      lastMessage?.role === "assistant"
    ) {
      next[next.length - 1] = {
        ...lastMessage,
        content: streamingAssistantText,
      };
    } else {
      if (
        !next.length &&
        userMessage
      ) {
        next.push({
          role: "user",
          content: userMessage,
        });
      }

      next.push({
        role: "assistant",
        content: streamingAssistantText,
      });
    }

    renderMessages(next);

    /*
     * Python's setBubbleText replaces the displayed
     * value, so pass the full accumulated response.
     */
    void callWindowApiWithArg(
      "push_bubble_text",
      streamingAssistantText,
    );

    return;
  }

  if (event.type === "status") {
    setTransientStatus(
      event.label || "Working...",
    );

    setStatus({
      loading: false,
      loaded: true,
    });

    return;
  }

  if (event.type === "done") {
    setTransientStatus("");
    bubbleStreaming = false;

    /*
     * Prefer the server's finalized message list.
     * Do not append event.reply to the streamed
     * response because it is normally the complete
     * final response.
     */
    if (Array.isArray(event.messages)) {
      renderMessages(event.messages);
    } else if (event.reply) {
      streamingAssistantText = String(
        event.reply,
      );

      const next = [...currentMessages];

      const lastMessage =
        next[next.length - 1];

      if (
        lastMessage?.role === "assistant"
      ) {
        next[next.length - 1] = {
          ...lastMessage,
          content: streamingAssistantText,
        };
      } else {
        next.push({
          role: "assistant",
          content: streamingAssistantText,
        });
      }

      renderMessages(next);
    }

    if (
      Number.isFinite(
        Number(event.version),
      )
    ) {
      lastKnownStateVersion = Number(
        event.version,
      );
    }

    hideBubbleSoon(10000);

    window.setTimeout(() => {
      void callWindowApiWithArg(
        "push_bubble_text",
        "",
      );
    }, 10000);

    return;
  }

  if (event.type === "error") {
    setTransientStatus("");
    bubbleStreaming = false;
    streamingAssistantText = "";

    hideBubbleSoon(1200);

    window.setTimeout(() => {
      void callWindowApiWithArg(
        "push_bubble_text",
        "",
      );
    }, 1200);

    throw new Error(
      event.error ||
        event.detail ||
        "Streaming failed.",
    );
  }
}

async function refreshStatus() {
  if (isSending) {
    return;
  }

  try {
    const payload =
      await loadStatePayload(false);

    const nextVersion = Number(
      payload.version || 0,
    );

    if (
      !preservePreview &&
      nextVersion !== lastKnownStateVersion
    ) {
      const fullPayload =
        await loadStatePayload(true);

      lastKnownStateVersion = Number(
        fullPayload.version ||
          nextVersion,
      );

      const nextMessages =
        Array.isArray(
          fullPayload.messages,
        )
          ? fullPayload.messages
          : [];

      const nextMessagesKey =
        JSON.stringify(nextMessages);

      if (
        nextMessagesKey !==
        lastRenderedMessagesKey
      ) {
        renderMessages(nextMessages);
      }

      setStatus(fullPayload.model || {});

      syncBackendControls(
        fullPayload.model || {},
      );

      return;
    }

    lastKnownStateVersion =
      nextVersion;

    setStatus(payload.model || {});

    syncBackendControls(
      payload.model || {},
    );
  } catch (error) {
    setStatus({
      error:
        error?.message ||
        "Could not refresh status.",
    });
  }
}

async function handleComposerSubmit(event) {
  event.preventDefault();

  if (elements.send) {
    elements.send.disabled = false;
  }

  const message =
    elements.input?.value?.trim() || "";

  if (!message || isSending) {
    return;
  }

  elements.input.value = "";
  autosizeInput();

  try {
    await sendMessage(message);
  } catch (error) {
    showNotice(
      error?.message ||
        "Could not send message.",
      "error",
    );
  } finally {
    if (elements.send) {
      elements.send.disabled = false;
    }

    requestComposerFocus();
    focusComposerInput();
  }
}

elements.composer?.addEventListener(
  "submit",
  handleComposerSubmit,
);

elements.input?.addEventListener(
  "input",
  () => {
    autosizeInput();
  },
);

elements.input?.addEventListener(
  "pointerdown",
  () => {
    if (!isSending) {
      requestComposerFocus();

      if (elements.send) {
        elements.send.disabled = false;
      }

      focusComposerInput();
    }
  },
);

elements.input?.addEventListener(
  "mousedown",
  () => {
    if (!isSending) {
      requestComposerFocus();
      focusComposerInput();
    }
  },
);

elements.input?.addEventListener(
  "focus",
  () => {
    if (
      !isSending &&
      elements.send
    ) {
      elements.send.disabled = false;
    }
  },
);

elements.send?.addEventListener(
  "click",
  async (event) => {
    if (elements.composer) {
      return;
    }

    await handleComposerSubmit(event);
  },
);

elements.minimizeButton?.addEventListener(
  "click",
  () => {
    void callWindowApi(
      "minimize_window",
    );
  },
);

elements.closeButton?.addEventListener(
  "click",
  () => {
    void callWindowApi(
      "close_window",
    );
  },
);

elements.messageActionButton?.addEventListener(
  "click",
  () => {
    void callWindowApi(
      "toggle_composer",
    );
  },
);

elements.backendSelect?.addEventListener(
  "change",
  () => {
    const backend =
      elements.backendSelect.value;

    elements.modelInput.placeholder =
      backendPlaceholder(backend);

    elements.modelInput.value =
      backend === "openrouter"
        ? backendValues.openrouter
        : backendValues.llama_cpp;
  },
);

elements.backendForm?.addEventListener(
  "submit",
  async (event) => {
    event.preventDefault();

    if (isSending) {
      return;
    }

    try {
      await switchBackend();
    } catch (error) {
      showNotice(
        error?.message ||
          "Could not switch backend.",
        "error",
      );
    } finally {
      await refreshStatus();
    }
  },
);

elements.input?.addEventListener(
  "keydown",
  (event) => {
    if (
      event.key !== "Enter" ||
      event.shiftKey
    ) {
      return;
    }

    event.preventDefault();

    if (
      elements.composer
        ?.requestSubmit
    ) {
      elements.composer.requestSubmit();
    } else if (elements.composer) {
      elements.composer.dispatchEvent(
        new Event("submit", {
          cancelable: true,
          bubbles: true,
        }),
      );
    } else {
      void handleComposerSubmit(event);
    }
  },
);

let memoryPanelVisible = false;

async function openMemoryPanel() {
  if (
    !elements.memoryPanel ||
    !elements.memoryContent ||
    memoryPanelVisible
  ) {
    return;
  }

  memoryPanelVisible = true;

  elements.memoryPanel.removeAttribute(
    "hidden",
  );

  elements.memoryContent.innerHTML =
    '<div class="memory-loading">Loading memory...</div>';

  try {
    const response = await fetch(
      apiUrl("/api/memory"),
    );

    if (!response.ok) {
      throw new Error(
        "Failed to load memory.",
      );
    }

    const memory =
      await response.json();

    elements.memoryContent.innerHTML =
      formatMemoryForDisplay(memory);
  } catch {
    elements.memoryContent.innerHTML =
      '<div class="memory-empty">Could not load memory.</div>';
  }
}

function closeMemoryPanel() {
  if (
    !elements.memoryPanel ||
    !memoryPanelVisible
  ) {
    return;
  }

  memoryPanelVisible = false;

  elements.memoryPanel.setAttribute(
    "hidden",
    "",
  );
}

function toggleMemoryPanel() {
  if (memoryPanelVisible) {
    closeMemoryPanel();
  } else {
    void openMemoryPanel();
  }
}

window.__akaneToggleMemory = () => {
  const wantVisible =
    document.body.getAttribute(
      "data-memory-visible",
    ) === "true";

  if (wantVisible) {
    void openMemoryPanel();
  } else {
    closeMemoryPanel();
  }
};

async function boot() {
  document.body.dataset.popupRole =
    POPUP_ROLE;

  document.documentElement.dataset.popupRole =
    POPUP_ROLE;

  if (
    POPUP_ROLE === "composer" ||
    POPUP_ROLE === "avatar"
  ) {
    setComposerOpenState(false);
  }

  if (POPUP_ROLE === "bubble") {
    setBubbleVisible(false);
    bubbleStreaming = false;
    streamingAssistantText = "";

    ensureBubbleResizeObserver();

    document.fonts?.ready
      ?.then(() => {
        scheduleBubbleSync();
      })
      .catch(() => {});
  }

  autosizeInput();

  try {
    await fetchState();
  } catch (error) {
    showNotice(
      error?.message ||
        "Could not load chat state.",
      "error",
    );

    setStatus({
      error:
        error?.message ||
        "Could not load chat state.",
    });
  }

  statusPoll = window.setInterval(
    refreshStatus,
    2500,
  );

  if (POPUP_ROLE === "composer") {
    requestComposerFocus();
  } else if (POPUP_ROLE !== "avatar") {
    focusComposerInput();
  }

  elements.memoryActionButton
    ?.addEventListener(
      "click",
      (event) => {
        event.stopPropagation();
        toggleMemoryPanel();
      },
    );

  elements.memoryCloseBtn
    ?.addEventListener(
      "click",
      (event) => {
        event.stopPropagation();
        closeMemoryPanel();
      },
    );

  elements.memoryPanel
    ?.addEventListener(
      "click",
      (event) => {
        if (
          event.target ===
          elements.memoryPanel
        ) {
          closeMemoryPanel();
        }
      },
    );
}

void boot();
