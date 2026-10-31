/* Complete replies stay available independently of the three-line viewport. */
(function (root) {
  "use strict";

  class AkaneSubtitles {
    constructor({ onChange = () => {}, now = () => Date.now(),
      setTimeout = (fn, delay) => globalThis.setTimeout(fn, delay),
      clearTimeout = timer => globalThis.clearTimeout(timer) } = {}) {
      this.onChange = onChange;
      this.now = now;
      this.setTimer = setTimeout;
      this.clearTimer = clearTimeout;
      this.replies = [];
      this.cursor = -1;
      this.visible = false;
      this.pinned = false;
      this.pauses = new Set();
      this.deadline = 0;
      this.remaining = 0;
      this.readTimer = null;
    }

    begin(id) {
      if (!this._reply(id)) this.replies.push({ id, text: "", complete: false, dismissed: false });
      return id;
    }

    append(id, text) {
      if (!text) return;
      this.begin(id);
      const reply = this._reply(id);
      if (reply.complete) return;
      const previous = this._readableText(reply);
      reply.text += text;
      if (!this.visible) this._showNext();
      else if (previous !== this._readableText(reply)) this._emit();
    }

    finish(id, finalText) {
      this.begin(id);
      const reply = this._reply(id);
      if (reply.complete) return;
      // Final-only transports may supply a suffix, but cannot erase streamed text.
      if (typeof finalText === "string" && finalText.startsWith(reply.text)) reply.text = finalText;
      reply.complete = true;
      if (!this.visible) this._showNext();
      else if (this.replies[this.cursor] === reply) {
        this._startReading();
        this._emit();
      } else this._emit();
    }

    pause(reason, paused) {
      if (this.pauses.has(reason) === Boolean(paused)) return;
      const wasPaused = this._paused();
      if (paused) this.pauses.add(reason);
      else this.pauses.delete(reason);
      this._pauseChanged(wasPaused);
    }

    setPinned(pinned) {
      const wasPaused = this._paused();
      this.pinned = Boolean(pinned);
      this._pauseChanged(wasPaused);
    }

    togglePinned() { this.setPinned(!this.pinned); }

    previous() {
      const index = this.replies.findLastIndex((reply, i) => i < this.cursor && reply.text.trim());
      if (index < 0) return;
      this.pauses.add("history");
      this._show(index);
    }

    next() {
      const index = this._nextIndex();
      if (index < 0) return;
      if (!this.replies.some((reply, i) => i > index && reply.text.trim())) this.pauses.delete("history");
      this._show(index);
    }

    dismiss() {
      const reply = this.replies[this.cursor];
      if (reply) reply.dismissed = true;
      this.pauses.delete("history");
      this.pinned = false;
      this._clearRead();
      this.visible = false;
      if (!this._showNext()) this._emit();
    }

    snapshot() {
      const reply = this.replies[this.cursor];
      return {
        visible: this.visible,
        text: reply ? this._readableText(reply) : "",
        responseId: reply?.id ?? null,
        hasPrevious: this.replies.some((reply, i) => i < this.cursor && reply.text.trim()),
        hasNext: this._nextIndex() >= 0,
        pinned: this.pinned,
        paused: this._paused(),
        followPaused: this._paused(),
        manual: this.pauses.has("manual"),
        streaming: Boolean(reply && !reply.complete),
        fullText: reply?.text || "",
        replies: this.replies.map(item => ({ id: item.id, text: item.text, complete: item.complete })),
      };
    }

    destroy() { this._clearRead(); }

    _reply(id) { return this.replies.find(reply => reply.id === id); }
    _paused() { return this.pinned || this.pauses.size > 0; }
    _emit() { this.onChange(this.snapshot()); }

    _readableText(reply) {
      const text = reply.text;
      if (reply.complete || /[.!?。！？…][”’"')\]]*$/u.test(text)) return text.trim();
      const boundaries = Array.from(text.matchAll(/\s+/gu), match => match.index + match[0].length);
      const end = boundaries.at(-1) || 0;
      // Show the first chunk immediately; buffer subsequent incomplete words.
      return (text.slice(0, end).trim() ? text.slice(0, end) : text).trim();
    }

    _pauseChanged(wasPaused) {
      const paused = this._paused();
      if (!wasPaused && paused) {
        if (this.readTimer !== null) this.remaining = Math.max(0, this.deadline - this.now());
        this._clearRead();
      } else if (wasPaused && !paused) this._scheduleRead();
      this._emit();
    }

    _clearRead() {
      if (this.readTimer !== null) this.clearTimer(this.readTimer);
      this.readTimer = null;
    }

    _startReading() {
      const text = this.replies[this.cursor].text;
      const words = text.match(/\S+/gu)?.length || 1;
      // Count unspaced text too; long replies get enough time to be read in full.
      this.remaining = Math.max(5000, words * 300 + 1500, Array.from(text).length * 60);
      this._scheduleRead();
    }

    _scheduleRead() {
      this._clearRead();
      if (!this.visible || this._paused() || !this.replies[this.cursor]?.complete) return;
      this.deadline = this.now() + this.remaining;
      this.readTimer = this.setTimer(() => {
        this.readTimer = null;
        this.dismiss();
      }, this.remaining);
    }

    _show(index) {
      this._clearRead();
      this.pauses.delete("manual");
      this.cursor = index;
      this.visible = Boolean(this.replies[index].text.trim());
      if (this.replies[index].complete) this._startReading();
      this._emit();
    }

    _showNext() {
      const index = this._nextIndex(true);
      if (index < 0) return false;
      this._show(index);
      return true;
    }

    _nextIndex(skipDismissed = false) {
      for (let index = this.cursor + 1; index < this.replies.length; index++) {
        const reply = this.replies[index];
        if (skipDismissed && reply.dismissed) continue;
        if (reply.text.trim()) return index;
        // A later initiative must not jump ahead of a reply awaiting its first token.
        if (!reply.complete) return -1;
      }
      return -1;
    }
  }

  if (typeof module !== "undefined" && module.exports) module.exports = AkaneSubtitles;
  if (root) root.AkaneSubtitles = AkaneSubtitles;
})(typeof window === "undefined" ? null : window);
