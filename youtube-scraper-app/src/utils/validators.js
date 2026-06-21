// Validation et reconnaissance des URLs YouTube.

const POST_PATTERNS = [
  /youtube\.com\/post\//i,
  /youtube\.com\/channel\/[^/]+\/community/i,
  /youtube\.com\/@[^/]+\/community/i,
  /youtube\.com\/.*\/community/i,
];

export function isYouTubeUrl(url = "") {
  return /(^|\.)youtube\.com/i.test(url) || /youtu\.be/i.test(url);
}

/** Vrai si l'URL pointe vers un post / une publication communautaire YouTube. */
export function isYouTubePostUrl(url = "") {
  return POST_PATTERNS.some((re) => re.test(url));
}

/** Extrait l'identifiant de post (Ug...) d'une URL si présent. */
export function extractPostId(url = "") {
  const m = url.match(/\/post\/([A-Za-z0-9_-]+)/);
  if (m) return m[1];
  const lb = url.match(/[?&]lb=([A-Za-z0-9_-]+)/);
  return lb ? lb[1] : null;
}

/** Normalise / valide une chaîne d'URL ; renvoie null si invalide. */
export function normalizeUrl(url = "") {
  const trimmed = String(url || "").trim();
  if (!trimmed) return null;
  try {
    return new URL(trimmed).href;
  } catch {
    try {
      return new URL(`https://${trimmed}`).href;
    } catch {
      return null;
    }
  }
}
