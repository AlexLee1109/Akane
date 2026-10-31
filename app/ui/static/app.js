const SEARCH_PARAMS = new URLSearchParams(window.location.search);
const POPUP_ROLE = SEARCH_PARAMS.get("popup_role") || "full";
const SESSION_ID = SEARCH_PARAMS.get("session_id") || "popup";

const elements = {
  messages:
    document.getElementById("messages") ||
    document.getElementById("speechBody") ||
    document.getElementById("dialogueText"),
  dialoguePanel: document.getElementById("dialoguePanel"),
  dialogueText: document.getElementById("dialogueText"),
  dialogueOpen: document.getElementById("dialogueOpen"),
  dialoguePrevious: document.getElementById("dialoguePrevious"),
  dialogueNext: document.getElementById("dialogueNext"),
  dialogueLatest: document.getElementById("dialogueLatest"),
  dialogueExpand: document.getElementById("dialogueExpand"),
  dialogueToolbar: document.querySelector(".dialogue-toolbar"),
  dialogueDismiss: document.getElementById("dialogueDismiss"),
  responsePanel: document.getElementById("responsePanel"),
  responseContent: document.getElementById("responseContent"),
  responsePrevious: document.getElementById("responsePrevious"),
  responseNext: document.getElementById("responseNext"),
  responseClose: document.getElementById("responseClose"),
  responsePosition: document.getElementById("responsePosition"),
  composer:
    document.getElementById("composer") ||
    document.getElementById("composerWrapper"),
  input: document.getElementById("messageInput"),
  send: document.getElementById("sendButton"),
  notice: document.getElementById("notice"),
  minimizeButton: document.getElementById("minimizeButton"),
  closeButton: document.getElementById("closeButton"),
  messageActionButton: document.getElementById("messageActionButton"),
  emptyState: document.getElementById("emptyState"),
  memoryActionButton: document.getElementById("memoryActionButton"),
  memoryCloseBtn: document.getElementById("memoryCloseBtn"),
  memoryPanel: document.getElementById("memoryPanel"),
  memoryContent: document.getElementById("memoryContent"),
  avatarHitbox: document.getElementById("avatarHitbox"),
  actionDock: document.querySelector(".avatar-action-dock"),
  characterArt: document.getElementById("characterArt"),
};

let currentMessages = [];
let isSending = false;
let isQuitting = false;
let refreshStatusIntervalId = null;
const activeFetchControllers = new Set();
let preservePreview = false;
let lastRenderedMessagesKey = "";
let lastKnownStateVersion = -1;
let lastInitiativeDeliveryId = "";

let streamingAssistantText = "";
let subtitles = null;
let activeSubtitleId = null;
let subtitleSequence = 0;
let subtitleFadeTimer = null;
let expandedReplyId = null;
let subtitleLayoutFrame = null;
let subtitleScrollFrame = null;
let subtitleScrollExpected = null;
let renderedSubtitleId = null;
let visibleViewport = null;
let lastInteractiveRegionsKey = "";

// Opt in from the popup console; independent of native health diagnostics.
let activeTTFT = null;
window.__akaneTTFTSamples = [];

function beginTTFT() {
  activeTTFT = null;
  if (!window.__akaneTTFTEnabled) {
    return "";
  }
  activeTTFT = {
    request_id: window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`,
    send_at: performance.now(),
    client_ttft_ms: null,
    status: "pending",
  };
  window.__akaneTTFTSamples.push(activeTTFT);
  if (window.__akaneTTFTSamples.length > 100) {
    window.__akaneTTFTSamples.shift();
  }
  return activeTTFT.request_id;
}

function recordTTFT(event, receivedAt) {
  const sample = activeTTFT;
  if (!sample || event.request_id !== sample.request_id) {
    return;
  }
  if (["done", "error", "cancelled"].includes(event.type)) {
    sample.status = event.type;
    sample.generation_id = event.generation_id;
    sample.server = event.ttft || null;
    sample.popup = event.popup_ttft || null;
  }
  if (event.type !== "delta" || !String(event.content || "").trim()) {
    return;
  }
  sample.first_js_delta_ms ??= receivedAt - sample.send_at;
  if (subtitles && subtitles.snapshot().responseId !== activeSubtitleId) {
    return;
  }
  const responseElement = POPUP_ROLE === "companion" ? elements.messages :
    elements.messages?.lastElementChild?.querySelector(".bubble-body");
  if (sample.first_dom_update_ms !== undefined || !responseElement?.textContent?.trim()) {
    return;
  }
  sample.first_dom_update_ms = performance.now() - receivedAt;
  const visible = () => document.visibilityState === "visible" &&
    responseElement.getClientRects().length > 0 &&
    getComputedStyle(responseElement).visibility === "visible";
  if (!visible()) {
    sample.render_unavailable = "document or response is hidden";
    return;
  }
  // Two frames bracket a render opportunity; this is not a pixel-paint sensor.
  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(() => {
      if (activeTTFT !== sample || !visible()) {
        sample.render_unavailable = "response replaced or hidden before frame";
        return;
      }
      sample.client_ttft_ms = performance.now() - sample.send_at;
    });
  });
}

const messageTimestampCache = new Map();
const API_BASE =
  SEARCH_PARAMS.get("api_base")?.replace(/\/$/, "") || "";
const API_TOKEN = SEARCH_PARAMS.get("api_token") || "";
let apiRequestHeaders = API_TOKEN
  ? { Authorization: `Bearer ${API_TOKEN}` }
  : {};

function apiUrl(path) {
  return API_BASE ? `${API_BASE}${path}` : path;
}

async function initializeRequestHeaders() {
  if (isQuitting) {
    return;
  }
  if (
    POPUP_ROLE === "companion" &&
    !window.pywebview?.api?.request_headers
  ) {
    await Promise.race([
      new Promise((resolve) => {
        window.addEventListener(
          "pywebviewready",
          resolve,
          { once: true },
        );
      }),
      new Promise((resolve) => {
        window.setTimeout(resolve, 2000);
      }),
    ]);
  }

  if (isQuitting) {
    return;
  }
  try {
    const bridgeHeaders =
      (await window.pywebview?.api?.request_headers?.()) || {};
    apiRequestHeaders = { ...apiRequestHeaders, ...bridgeHeaders };
  } catch {}
}

function apiFetch(path, options = {}) {
  if (isQuitting) {
    return Promise.reject(
      new DOMException("Popup is closing.", "AbortError"),
    );
  }
  const controller = new AbortController();
  activeFetchControllers.add(controller);
  return fetch(apiUrl(path), {
    ...options,
    signal: controller.signal,
    headers: {
      ...apiRequestHeaders,
      ...(options.headers || {}),
    },
  }).finally(() => {
    activeFetchControllers.delete(controller);
  });
}

async function callWindowApi(methodName) {
  if (isQuitting && methodName !== "close_window") {
    return;
  }
  if (!window.pywebview?.api?.[methodName]) {
    return;
  }

  try {
    await window.pywebview.api[methodName]();
  } catch {
    if (!isQuitting) {
      showNotice("Window action failed.", "error");
    }
  }
}

function setComposerOpenState(open) {
  document.body.dataset.composerOpen = open ? "true" : "false";

  if (elements.composer) {
    elements.composer.hidden = !open;
  }

  scheduleDialogueLayout();
  syncInteractiveRegions();
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

function clearBubbleHideTimer() {
  if (subtitleFadeTimer !== null) {
    window.clearTimeout(subtitleFadeTimer);
    subtitleFadeTimer = null;
  }
}

function layoutDialogue() {
  if (!elements.composer) return;
  const rect = elements.composer.getBoundingClientRect();
  const root = document.documentElement;
  const gap = parseFloat(getComputedStyle(root).getPropertyValue("--dialogue-gap"));
  const left = Math.max(0, visibleViewport?.left || 0, rect.left);
  const right = Math.min(window.innerWidth, visibleViewport?.right ?? window.innerWidth, rect.right);
  const width = Math.max(1, right - left);
  const bottom = Math.min(rect.top - gap, visibleViewport?.bottom ?? window.innerHeight, window.innerHeight);
  root.style.setProperty("--dialogue-left", `${left}px`);
  root.style.setProperty("--dialogue-width", `${width}px`);
  root.style.setProperty("--dialogue-bottom", `${window.innerHeight - bottom}px`);
  root.style.setProperty("--dialogue-max-height", `${Math.max(1, bottom - (visibleViewport?.top || 0) - 26)}px`);
  root.style.setProperty("--response-max-height", `${Math.max(1, bottom - (visibleViewport?.top || 0) - 12)}px`);
  scheduleSubtitleScroll();
  syncInteractiveRegions();
}

async function refreshVisibleViewport() {
  const bounds = await window.pywebview?.api?.viewport_bounds?.();
  if (bounds) visibleViewport = bounds;
  scheduleDialogueLayout();
}

window.__akaneViewportChanged = refreshVisibleViewport;

function scheduleDialogueLayout() {
  if (subtitleLayoutFrame !== null) return;
  subtitleLayoutFrame = window.requestAnimationFrame(() => {
    subtitleLayoutFrame = null;
    layoutDialogue();
  });
}

function stopSubtitleScroll() {
  if (subtitleScrollFrame !== null) window.cancelAnimationFrame(subtitleScrollFrame);
  subtitleScrollFrame = null;
}

function subtitleAtBottom() {
  const viewport = elements.dialogueOpen;
  return viewport.scrollHeight - viewport.clientHeight - viewport.scrollTop <= 6;
}

function scheduleSubtitleScroll() {
  const state = subtitles?.snapshot();
  if (!state?.visible || state.followPaused || elements.responsePanel && !elements.responsePanel.hidden) {
    stopSubtitleScroll();
    return;
  }
  if (subtitleScrollFrame !== null) return;
  let previousTime;
  const step = time => {
    subtitleScrollFrame = null;
    if (subtitles.snapshot().followPaused) return;
    const viewport = elements.dialogueOpen;
    const target = Math.max(0, viewport.scrollHeight - viewport.clientHeight);
    const distance = target - viewport.scrollTop;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const fraction = reduced ? 1 : 1 - Math.exp(-Math.min(50, time - (previousTime ?? time - 16)) / 65);
    const movement = Math.sign(distance) * Math.max(1, Math.abs(distance * fraction));
    subtitleScrollExpected = Math.abs(distance) <= 1 ? target : viewport.scrollTop + movement;
    viewport.scrollTop = subtitleScrollExpected;
    subtitleScrollExpected = viewport.scrollTop;
    previousTime = time;
    if (Math.abs(target - viewport.scrollTop) > 0.75) subtitleScrollFrame = window.requestAnimationFrame(step);
  };
  subtitleScrollFrame = window.requestAnimationFrame(step);
}

function availableReplies() {
  const live = subtitles?.snapshot().replies || [];
  const liveTexts = new Set(live.map(reply => reply.text));
  return [
    ...currentMessages.filter(message => message.role === "assistant" && message.content &&
      !liveTexts.has(message.content)).map((message, index) => ({
        id: `history-${index}`, text: message.content, complete: true,
      })),
    ...live.filter(reply => reply.text.trim()),
  ];
}

function renderCompleteResponse() {
  if (!elements.responsePanel || elements.responsePanel.hidden) return;
  const replies = availableReplies();
  let index = replies.findIndex(reply => reply.id === expandedReplyId);
  if (index < 0) index = replies.length - 1;
  const reply = replies[index];
  if (!reply) return;
  expandedReplyId = reply.id;
  if (elements.responseContent.textContent !== reply.text) {
    const scroll = elements.responseContent.scrollTop;
    const previous = elements.responseContent.textContent;
    const textNode = elements.responseContent.firstChild;
    // Appending to the existing text node preserves an active selection.
    if (textNode && reply.text.startsWith(previous)) textNode.appendData(reply.text.slice(previous.length));
    else elements.responseContent.textContent = reply.text;
    elements.responseContent.scrollTop = scroll;
  }
  elements.responsePosition.textContent = `${index + 1} / ${replies.length}`;
  elements.responsePrevious.disabled = index <= 0;
  elements.responseNext.disabled = index >= replies.length - 1;
}

function renderSubtitle(snapshot) {
  if (!elements.dialoguePanel) return;
  const panel = elements.dialoguePanel;
  const expanded = !elements.responsePanel.hidden;
  clearBubbleHideTimer();
  if (snapshot.visible && !expanded) {
    const appearing = panel.hidden || panel.dataset.visible !== "true";
    panel.hidden = false;
    if (renderedSubtitleId !== snapshot.responseId) {
      renderedSubtitleId = snapshot.responseId;
      stopSubtitleScroll();
      elements.dialogueOpen.scrollTop = 0;
      subtitleScrollExpected = 0;
    }
    if (elements.dialogueText.textContent !== snapshot.text) {
      const previous = elements.dialogueText.textContent;
      const textNode = elements.dialogueText.firstChild;
      if (textNode && snapshot.text.startsWith(previous)) textNode.appendData(snapshot.text.slice(previous.length));
      else elements.dialogueText.textContent = snapshot.text;
      if (activeTTFT?.first_js_delta_ms !== undefined && snapshot.responseId === activeSubtitleId) {
        recordTTFT({ type: "delta", content: snapshot.text, request_id: activeTTFT.request_id }, performance.now());
      }
    }
    if (appearing) {
      // A layout flush establishes the start of the inexpensive opacity fade.
      panel.getBoundingClientRect();
    }
    panel.dataset.visible = "true";
  } else {
    panel.dataset.visible = "false";
    if (expanded || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
      panel.hidden = true;
    } else if (!panel.hidden) {
      subtitleFadeTimer = window.setTimeout(() => {
        panel.hidden = true;
        syncInteractiveRegions();
      }, 180);
    }
  }
  elements.dialoguePrevious.disabled = !snapshot.hasPrevious;
  elements.dialogueNext.disabled = !snapshot.hasNext;
  elements.dialogueLatest.hidden = !snapshot.manual;
  scheduleSubtitleScroll();
  renderCompleteResponse();
  syncInteractiveRegions();
}

function openCompleteResponse() {
  if (!availableReplies().length) return;
  expandedReplyId = subtitles?.snapshot().responseId || availableReplies().at(-1).id;
  elements.responsePanel.hidden = false;
  elements.dialogueExpand.setAttribute("aria-expanded", "true");
  subtitles?.pause("expanded", true);
  // Hiding a hovered/focused caption does not reliably dispatch leave events.
  subtitles?.pause("hover", false);
  subtitles?.pause("focus", false);
  renderCompleteResponse();
  elements.responseContent.focus({ preventScroll: true });
  syncInteractiveRegions();
}

function closeCompleteResponse() {
  elements.responsePanel.hidden = true;
  elements.dialogueExpand.setAttribute("aria-expanded", "false");
  subtitles?.pause("hover", false);
  subtitles?.pause("focus", false);
  subtitles?.pause("expanded", false);
  const target = subtitles?.snapshot().visible ? elements.dialogueOpen : elements.input;
  target?.focus({ preventScroll: true });
  syncInteractiveRegions();
}

function beginSubtitle() {
  if (!subtitles || activeSubtitleId !== null) return;
  activeSubtitleId = `reply-${++subtitleSequence}`;
  subtitles.begin(activeSubtitleId);
}

function finishSubtitle(finalText) {
  if (activeSubtitleId !== null) {
    subtitles?.finish(activeSubtitleId, finalText);
    activeSubtitleId = null;
  }
}

// Kept for native integrations; complete replies use the same reading lifecycle.
function setBubbleText(text) {
  const value = String(text || "");
  if (!value.trim() || !subtitles) return;
  const id = `reply-${++subtitleSequence}`;
  subtitles.begin(id);
  subtitles.append(id, value);
  subtitles.finish(id);
}

function showBubbleNow() {
  beginSubtitle();
}

function initializeSubtitles() {
  if (!elements.dialoguePanel || !window.AkaneSubtitles) return;
  layoutDialogue();
  subtitles = new window.AkaneSubtitles({ onChange: renderSubtitle });
  window.__akaneSubtitles = subtitles;
  elements.dialoguePrevious.addEventListener("click", () => subtitles.previous());
  elements.dialogueNext.addEventListener("click", () => subtitles.next());
  elements.dialogueDismiss.addEventListener("click", () => subtitles.dismiss());
  elements.dialogueExpand.addEventListener("click", openCompleteResponse);
  elements.avatarHitbox?.addEventListener("dblclick", openCompleteResponse);
  elements.messageActionButton?.addEventListener("contextmenu", event => {
    event.preventDefault();
    openCompleteResponse();
  });
  window.addEventListener("keydown", event => {
    if (event.altKey && event.key === "ArrowUp") {
      event.preventDefault();
      openCompleteResponse();
    }
  });
  elements.dialogueLatest.addEventListener("click", () => {
    subtitles.pause("manual", false);
    stopSubtitleScroll();
    const viewport = elements.dialogueOpen;
    subtitleScrollExpected = Math.max(0, viewport.scrollHeight - viewport.clientHeight);
    viewport.scrollTop = subtitleScrollExpected;
    subtitleScrollExpected = viewport.scrollTop;
  });
  elements.dialogueOpen.addEventListener("keydown", event => {
    if (event.key === "Enter") {
      event.preventDefault();
      openCompleteResponse();
    }
    if (event.key === "ArrowLeft") { event.preventDefault(); subtitles.previous(); }
    if (event.key === "ArrowRight") { event.preventDefault(); subtitles.next(); }
  });
  const viewport = elements.dialogueOpen;
  let previousScrollTop = viewport.scrollTop;
  const manualScroll = upward => {
    stopSubtitleScroll();
    subtitleScrollExpected = null;
    if (upward && viewport.scrollHeight > viewport.clientHeight + 1) subtitles.pause("manual", true);
  };
  viewport.addEventListener("wheel", event => manualScroll(event.deltaY < 0), { passive: true });
  viewport.addEventListener("touchstart", () => manualScroll(true), { passive: true });
  viewport.addEventListener("keydown", event => {
    if (["ArrowUp", "PageUp", "Home"].includes(event.key) || event.key === " " && event.shiftKey) manualScroll(true);
    if (["ArrowDown", "PageDown", "End", " "].includes(event.key)) manualScroll(false);
  });
  viewport.addEventListener("scroll", () => {
    const previous = previousScrollTop;
    previousScrollTop = viewport.scrollTop;
    if (subtitleScrollExpected !== null && Math.abs(viewport.scrollTop - subtitleScrollExpected) <= 1) return;
    if (subtitleAtBottom()) subtitles.pause("manual", false);
    else if (viewport.scrollTop < previous - 1) {
      manualScroll(true);
    }
  }, { passive: true });
  viewport.addEventListener("pointerdown", () => {
    stopSubtitleScroll();
    subtitles.pause("pointer", true);
  });
  window.addEventListener("pointerup", () => subtitles.pause("pointer", false));
  window.addEventListener("pointercancel", () => subtitles.pause("pointer", false));
  elements.responseClose.addEventListener("click", closeCompleteResponse);
  for (const [button, direction] of [[elements.responsePrevious, -1], [elements.responseNext, 1]]) {
    button.addEventListener("click", () => {
      const replies = availableReplies();
      const index = replies.findIndex(reply => reply.id === expandedReplyId);
      expandedReplyId = replies[index + direction]?.id || expandedReplyId;
      elements.responseContent.scrollTop = 0;
      renderCompleteResponse();
    });
  }
  for (const panel of [elements.dialoguePanel, elements.responsePanel]) {
    panel.addEventListener("pointerenter", () => { subtitles.pause("hover", true); syncInteractiveRegions(); });
    panel.addEventListener("pointerleave", () => { subtitles.pause("hover", false); syncInteractiveRegions(); });
    panel.addEventListener("focusin", () => { subtitles.pause("focus", true); syncInteractiveRegions(); });
    panel.addEventListener("focusout", event => {
      if (!panel.contains(event.relatedTarget)) subtitles.pause("focus", false);
      syncInteractiveRegions();
    });
    panel.addEventListener("keydown", event => {
      if (event.key === "Escape") {
        event.preventDefault();
        if (!elements.responsePanel.hidden) closeCompleteResponse();
        else subtitles.dismiss();
      }
    });
  }
  document.addEventListener("selectionchange", () => {
    const selection = window.getSelection();
    const selected = !selection?.isCollapsed && [elements.dialoguePanel, elements.responsePanel]
      .some(panel => panel.contains(selection?.anchorNode) || panel.contains(selection?.focusNode));
    subtitles.pause("selection", Boolean(selected));
  });
  document.addEventListener("visibilitychange", () => {
    subtitles.pause("hidden", document.visibilityState !== "visible");
  });
  if (window.ResizeObserver) {
    const observer = new ResizeObserver(scheduleDialogueLayout);
    observer.observe(elements.composer);
    observer.observe(document.querySelector(".shell"));
    observer.observe(elements.dialoguePanel);
    observer.observe(elements.responsePanel);
    observer.observe(elements.dialogueText);
  }
  document.fonts?.ready.then(layoutDialogue);
  void refreshVisibleViewport();
  renderSubtitle(subtitles.snapshot());
}

window.__akaneSetBubbleText = setBubbleText;

window.__akanePendingMessage = "";

window.__akaneStreamEvent = (event) => {
  if (isQuitting) {
    return;
  }
  if (
    event.type === "done" ||
    event.type === "error" ||
    event.type === "cancelled"
  ) {
    setSendingState(false);
  }

  try {
    handleStreamEvent(
      event,
      window.__akanePendingMessage,
    );
  } catch (error) {
    showNotice(
      error?.message ||
        "Could not receive Akane's reply.",
      "error",
    );
  }
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

function rememberMessages(messages) {
  currentMessages = Array.isArray(messages)
    ? messages
    : [];

  lastRenderedMessagesKey =
    JSON.stringify(currentMessages);
}

function renderMessages(messages) {
  const previous = currentMessages.at(-1);
  const initialized = lastRenderedMessagesKey !== "";
  rememberMessages(messages);

  if (subtitles) {
    const latest = currentMessages.at(-1);
    if (initialized && !isSending && latest?.role === "assistant" && latest.content &&
        (previous?.role !== "assistant" || previous.content !== latest.content) &&
        !subtitles.snapshot().replies.some(reply => reply.text === latest.content)) {
      setBubbleText(latest.content);
    }
    renderSubtitle(subtitles.snapshot());
    return;
  }

  if (!elements.messages) {
    return;
  }

  if (POPUP_ROLE === "companion") {
    const lastMessage =
      currentMessages[
        currentMessages.length - 1
      ];

    if (
      lastMessage?.role === "assistant" &&
      lastMessage.content
    ) {
      setBubbleText(lastMessage.content);
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

function updateStreamingAssistant(
  assistantText,
  userMessage,
) {
  const lastMessage =
    currentMessages[
      currentMessages.length - 1
    ];

  if (lastMessage?.role === "assistant") {
    lastMessage.content = assistantText;
  } else {
    if (!currentMessages.length && userMessage) {
      currentMessages.push({
        role: "user",
        content: userMessage,
      });
    }

    currentMessages.push({
      role: "assistant",
      content: assistantText,
    });
  }

  if (subtitles) {
    renderCompleteResponse();
    return;
  }
  if (POPUP_ROLE === "companion") return;

  const row = elements.messages?.lastElementChild;
  const body = row?.classList.contains(
    "message-row-assistant",
  )
    ? row.querySelector(".bubble-body")
    : null;

  if (!body) {
    renderMessages(currentMessages);
    return;
  }

  body.innerHTML = renderMessageBody(
    assistantText,
  );
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

  if (elements.send) {
    elements.send.disabled = sending;
  }
}

async function fetchState() {
  const payload = await loadStatePayload();

  renderMessages(payload.messages || []);

  lastKnownStateVersion = Number(
    payload.version || 0,
  );

}

async function loadStatePayload(
  includeMessages = true,
) {
  const response = await apiFetch(
    `/api/state?session_id=${encodeURIComponent(
        SESSION_ID,
      )}&include_messages=${
        includeMessages ? "1" : "0"
      }`,
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
  const requestId = message.startsWith("/") ? "" : beginTTFT();
  preservePreview = false;

  if (message.startsWith("/")) {
    return sendCommand(message);
  }

  setSendingState(true);

  streamingAssistantText = "";
  beginSubtitle();

  upsertStreamingMessages(message, "");

  if (
    window.pywebview?.api
      ?.send_message_stream
  ) {
    window.__akanePendingMessage = message;

    try {
      await window.pywebview.api
        .send_message_stream(message, requestId, Boolean(requestId));
    } catch (error) {
      showNotice(
        error?.message ||
          "Could not start streaming.",
        "error",
      );

      finishSubtitle();
      setSendingState(false);
    }

    return;
  }

  try {
    const response = await apiFetch(
      "/api/chat/stream",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message,
          session_id: SESSION_ID,
          source: "popup",
          request_id: requestId,
          trace_ttft: Boolean(requestId),
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
    try {
      while (true) {
        const {
          value,
          done,
        } = await reader.read();

        buffer += decoder.decode(value, {
          stream: !done,
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
        if (done) {
          break;
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
      await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
  } finally {
    finishSubtitle();
    setSendingState(false);
  }
}

async function sendCommand(message) {
  setSendingState(true);

  try {
    const response = await apiFetch(
      "/api/chat",
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

function handleStreamEvent(
  event,
  userMessage,
) {
  const receivedAt = activeTTFT ? performance.now() : 0;
  if (event.type !== "delta") {
    recordTTFT(event, receivedAt);
  }
  if (event.type === "start") {
    streamingAssistantText = "";
    beginSubtitle();

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

    streamingAssistantText += incoming;
    subtitles?.append(activeSubtitleId, incoming);
    updateStreamingAssistant(
      streamingAssistantText,
      userMessage,
    );
    recordTTFT(event, receivedAt);

    return;
  }

  if (event.type === "done") {

    // Store final history without replacing the subtitle being read.
    if (Array.isArray(event.messages)) {
      if (POPUP_ROLE === "companion") {
        rememberMessages(event.messages);
      } else {
        renderMessages(event.messages);
      }
    } else if (event.reply) {
      const finalizedAssistantText = String(
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
          content: finalizedAssistantText,
        };
      } else {
        next.push({
          role: "assistant",
          content: finalizedAssistantText,
        });
      }

      if (POPUP_ROLE === "companion") {
        rememberMessages(next);
      } else {
        renderMessages(next);
      }
    }

    // Only the visible reply is queued; history and semantic sidecars never enter playback.
    const finalReply = event.reply || (Array.isArray(event.messages)
      ? event.messages.at(-1)?.role === "assistant" && event.messages.at(-1).content : "");
    finishSubtitle(finalReply || streamingAssistantText);

    return;
  }

  if (event.type === "error") {
    finishSubtitle();
    streamingAssistantText = "";

    throw new Error(
      event.error ||
        event.detail ||
      "Streaming failed.",
    );
  }

  if (event.type === "cancelled") {
    finishSubtitle();
    streamingAssistantText = "";
  }
}

async function refreshStatus() {
  if (isQuitting || isSending) {
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

      return;
    }

    lastKnownStateVersion =
      nextVersion;

  } catch {
    return;
  }
}

async function reportPopupAvailability(available) {
  if (isQuitting) {
    return null;
  }
  return apiFetch("/api/initiative/delivery/claim", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    keepalive: !available,
    body: JSON.stringify({
      adapter: "popup",
      conversation_id: SESSION_ID,
      available,
      wait_seconds: available ? 25 : 0,
    }),
  });
}

async function acknowledgeInitiative(
  delivery,
  success,
) {
  const response = await apiFetch(
    "/api/initiative/delivery/ack",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        adapter: "popup",
        conversation_id: SESSION_ID,
        opportunity_id:
          delivery.opportunity_id,
        claim_token: delivery.claim_token,
        success,
        message_id:
          delivery.opportunity_id,
      }),
    },
  );
  return response.ok;
}

async function initiativeDeliveryLoop() {
  if (POPUP_ROLE !== "companion") {
    return;
  }
  while (!isQuitting) {
    if (document.visibilityState !== "visible") {
      try {
        await reportPopupAvailability(false);
      } catch {}
      await new Promise((resolve) => {
        document.addEventListener(
          "visibilitychange",
          resolve,
          { once: true },
        );
      });
      continue;
    }
    try {
      const response =
        await reportPopupAvailability(true);
      if (isQuitting || !response) {
        return;
      }
      if (response.status === 404 || response.status === 405) {
        return;
      }
      if (!response.ok) {
        throw new Error("Initiative delivery is unavailable.");
      }
      const payload = await response.json();
      if (isQuitting) {
        return;
      }
      const delivery = payload.delivery;
      if (!delivery?.message) {
        continue;
      }
      if (document.visibilityState !== "visible") {
        await acknowledgeInitiative(
          delivery,
          false,
        );
        continue;
      }
      if (
        lastInitiativeDeliveryId !==
        delivery.opportunity_id
      ) {
        lastInitiativeDeliveryId =
          delivery.opportunity_id;
        rememberMessages([
          ...currentMessages,
          {
            role: "assistant",
            content: delivery.message,
          },
        ]);
        setBubbleText(delivery.message);
      }
      await acknowledgeInitiative(
        delivery,
        true,
      );
      lastKnownStateVersion = -1;
    } catch {
      if (isQuitting) {
        return;
      }
      await new Promise((resolve) => {
        window.setTimeout(resolve, 1000);
      });
    }
  }
}

async function handleComposerSubmit(event) {
  event.preventDefault();
  if (isQuitting) {
    return;
  }

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
  async () => {
    if (isQuitting) {
      return;
    }
    isQuitting = true;
    elements.closeButton.disabled = true;
    elements.closeButton.setAttribute(
      "aria-disabled",
      "true",
    );
    clearBubbleHideTimer();
    subtitles?.destroy();
    stopSubtitleScroll();
    for (const controller of activeFetchControllers) {
      controller.abort();
    }
    activeFetchControllers.clear();
    if (refreshStatusIntervalId !== null) {
      window.clearInterval(
        refreshStatusIntervalId,
      );
      refreshStatusIntervalId = null;
    }
    window.speechSynthesis?.cancel?.();
    try {
      await window.pywebview?.api
        ?.close_window?.();
    } catch {
      // Native teardown can dispose the bridge before its promise settles.
    }
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

function syncInteractiveRegions() {
  if (
    isQuitting ||
    POPUP_ROLE !== "companion" ||
    !window.pywebview?.api
      ?.update_interactive_regions
  ) {
    return;
  }

  const captionVisible = elements.dialoguePanel?.dataset.visible === "true";
  const dockVisible = elements.actionDock && getComputedStyle(elements.actionDock).pointerEvents !== "none";
  const interactiveElements = [
    captionVisible ? elements.dialoguePanel : null,
    captionVisible ? elements.dialogueToolbar : null,
    elements.responsePanel,
    elements.avatarHitbox,
    dockVisible ? elements.actionDock : null,
    elements.composer,
    elements.memoryPanel,
  ];
  const clippedRect = (element) => {
    if (!element || element.hidden || element.closest?.("[hidden]")) return null;
    const style = getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden") return null;
    const rect = element.getBoundingClientRect();
    const left = Math.max(0, rect.left);
    const top = Math.max(0, rect.top);
    const right = Math.min(
      window.innerWidth,
      rect.right,
    );
    const bottom = Math.min(
      window.innerHeight,
      element === elements.dialogueToolbar
        ? Math.max(rect.bottom, elements.dialoguePanel.getBoundingClientRect().top)
        : rect.bottom,
    );

    if (right <= left || bottom <= top) {
      return null;
    }

    return {
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
    };
  };
  const interactiveRegions =
    interactiveElements
      .map((element) => clippedRect(element))
      .filter(Boolean);
  const key = JSON.stringify(interactiveRegions);
  if (key === lastInteractiveRegionsKey) return;
  lastInteractiveRegionsKey = key;
  // Streaming words usually leave these rectangles unchanged. Avoid crossing
  // the native bridge again until the actual interactive geometry changes.
  window.pywebview.api.update_interactive_regions(interactiveRegions, [])?.catch?.(() => {
    if (lastInteractiveRegionsKey === key) lastInteractiveRegionsKey = "";
  });
}

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
  syncInteractiveRegions();

  elements.memoryContent.innerHTML =
    '<div class="memory-loading">Loading memory...</div>';

  try {
    const response = await apiFetch("/api/memory");

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
  syncInteractiveRegions();
}

function toggleMemoryPanel() {
  if (memoryPanelVisible) {
    closeMemoryPanel();
  } else {
    void openMemoryPanel();
  }
}

async function boot() {
  document.body.dataset.popupRole =
    POPUP_ROLE;

  document.documentElement.dataset.popupRole =
    POPUP_ROLE;

  initializeSubtitles();
  autosizeInput();
  focusComposerInput();
  syncInteractiveRegions();
  elements.characterArt?.addEventListener(
    "load",
    syncInteractiveRegions,
  );

  const syncDockRegions = () => {
    window.requestAnimationFrame(
      syncInteractiveRegions,
    );
    window.setTimeout(
      syncInteractiveRegions,
      220,
    );
  };

  // Reveal only in the right-side button column, including over the subtitle,
  // without placing an invisible hitbox over text.
  document.addEventListener("pointermove", event => {
    const rect = elements.actionDock?.getBoundingClientRect();
    const visible = rect && event.clientX >= rect.left && event.clientX <= rect.right &&
      event.clientY >= rect.top && event.clientY <= rect.bottom;
    const value = visible ? "true" : "false";
    if (elements.actionDock && elements.actionDock.dataset.visible !== value) {
      elements.actionDock.dataset.visible = value;
      syncInteractiveRegions();
    }
  });
  document.addEventListener("pointerleave", () => {
    if (elements.actionDock) elements.actionDock.dataset.visible = "false";
    syncInteractiveRegions();
  });

  for (const element of [
    elements.avatarHitbox,
    elements.actionDock,
  ]) {
    element?.addEventListener(
      "pointerenter",
      syncDockRegions,
    );
    element?.addEventListener(
      "pointerleave",
      syncDockRegions,
    );
    element?.addEventListener("focusin", syncDockRegions);
    element?.addEventListener("focusout", syncDockRegions);
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

  await initializeRequestHeaders();
  if (isQuitting) {
    return;
  }

  try {
    await fetchState();
  } catch (error) {
    showNotice(
      error?.message ||
        "Could not load chat state.",
      "error",
    );

  }

  if (isQuitting) {
    return;
  }

  refreshStatusIntervalId = window.setInterval(
    refreshStatus,
    2500,
  );

  void initiativeDeliveryLoop();

  window.addEventListener("pagehide", () => {
    if (!isQuitting) {
      void reportPopupAvailability(false);
    }
  });


}

window.addEventListener(
  "pywebviewready",
  () => { syncInteractiveRegions(); void refreshVisibleViewport(); },
);
window.addEventListener(
  "resize",
  () => { scheduleDialogueLayout(); void refreshVisibleViewport(); },
);

void boot();
