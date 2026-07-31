const guestTokenKey = "akane-public-guest-token";

export function getGuestToken() {
  try {
    return sessionStorage.getItem(guestTokenKey)?.trim() || null;
  } catch {
    return null;
  }
}

export function storeGuestToken(token: string) {
  try {
    sessionStorage.setItem(guestTokenKey, token);
  } catch {
    // The active tab can still use the token when storage is unavailable.
  }
}

export function clearGuestToken() {
  try {
    sessionStorage.removeItem(guestTokenKey);
  } catch {
    // There is no stored token to remove when storage is unavailable.
  }
}
