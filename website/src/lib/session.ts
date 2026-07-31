const sessionKey = "akane-web-demo-session";

export function getWebSessionId() {
  const existing = sessionStorage.getItem(sessionKey);
  if (existing) return existing;
  const id = `public:web:${crypto.randomUUID()}`;
  sessionStorage.setItem(sessionKey, id);
  return id;
}

export function resetWebSessionId() {
  const id = `public:web:${crypto.randomUUID()}`;
  sessionStorage.setItem(sessionKey, id);
  return id;
}
