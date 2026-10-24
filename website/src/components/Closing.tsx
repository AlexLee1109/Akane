import { Link } from "react-router-dom";
import { projectConfig } from "../config/project";

export function Closing() {
  return <section className="closing shell" aria-labelledby="closing-title"><div className="closing-copy"><p className="eyebrow">Every story starts somewhere</p><h2 id="closing-title">Start with hello.<br /><em>See where it goes.</em></h2><p>A familiar face. A new conversation.</p><div className="actions"><Link className="button primary" to="/demo">Talk to Akane <span aria-hidden="true">↗</span></Link><a className="text-link" href={projectConfig.githubUrl} target="_blank" rel="noreferrer">Explore GitHub ↗</a></div></div><div className="closing-portrait"><span className="closing-name" aria-hidden="true">茜</span><img src={`${projectConfig.basePath}assets/akane-standing.webp`} width="335" height="1147" alt="Akane, with blue eyes, long blue hair, and her white-and-blue jacket" loading="lazy" /></div></section>;
}
