export function Architecture({ compact = false }: { compact?: boolean }) {
  return <figure className={`architecture ${compact ? "compact" : ""}`} aria-label="Authorized interfaces connect to a shared local inference runtime. Selected Character, Memory, and Self context informs replies. Validated changes are saved atomically; guest profiles stay isolated.">
    <figcaption><span className="diagram-dot" /> AKANE SYSTEM <span>LOCAL / PERSISTENT</span></figcaption>
    <div className="architecture-clients"><span>Desktop</span><span>Discord</span><span>Web guest</span></div>
    <div className="architecture-connector" aria-hidden="true"><span />↓<span /></div>
    <div className="architecture-core"><span className="eyebrow">Authorized profile · Selected context</span><strong>Local inference runtime</strong><span>FastAPI / llama.cpp / GGUF</span></div>
    <div className="architecture-connector" aria-hidden="true">↕</div>
    <div className="architecture-state"><span>Character<small>Identity & voice</small></span><span>Memory<small>Useful context</small></span><span>Self<small>A perspective</small></span></div>
    <div className="architecture-store"><span aria-hidden="true">↳</span> Validated state <span aria-hidden="true">→</span> Atomic Store</div>
    {!compact && <p className="architecture-note">Grounded Experience supports development. Each public guest has a temporary profile, separate from the owner.</p>}
  </figure>;
}
