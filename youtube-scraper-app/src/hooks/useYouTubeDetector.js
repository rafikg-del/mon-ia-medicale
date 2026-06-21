import { useEffect } from "react";
import { extractPost } from "../utils/scraper.js";
import { cleanText } from "../utils/parser.js";
import { isYouTubePostUrl, isYouTubeUrl } from "../utils/validators.js";

// Décode un payload de capture provenant du bookmarklet (#capture=...).
function parseHashCapture() {
  const hash = window.location.hash || "";
  const m = hash.match(/[#&]capture=([^&]+)/);
  if (!m) return null;
  try {
    const obj = JSON.parse(decodeURIComponent(m[1]));
    if (!obj || !obj.content) return null;
    return {
      content: cleanText(obj.content),
      author: cleanText(obj.author || ""),
      date: cleanText(obj.date || ""),
      url: obj.url || "",
      capturedAt: obj.capturedAt || Date.now(),
      source: "bookmarklet",
    };
  } catch {
    return null;
  }
}

// Décode une cible de partage (Web Share Target : ?title=&text=&url=).
function parseShareTarget() {
  const params = new URLSearchParams(window.location.search);
  if (!params.has("title") && !params.has("text") && !params.has("url")) return null;
  const sharedUrl = params.get("url") || "";
  const text = params.get("text") || "";
  const title = params.get("title") || "";

  // Sur mobile, le texte partagé contient souvent le post lui-même.
  const url = sharedUrl || (isYouTubeUrl(text) ? text.match(/https?:\/\/\S+/)?.[0] : "") || "";
  const content = cleanText(text || title);
  if (!content && !url) return null;

  return {
    content,
    author: cleanText(title),
    date: "",
    url,
    capturedAt: Date.now(),
    source: "share-target",
  };
}

// Cas limite : l'app est ouverte directement dans le contexte d'un post
// YouTube (WebView intégrée) -> on tente l'extraction DOM locale.
function parseSelfContext() {
  if (!isYouTubePostUrl(window.location.href)) return null;
  const post = extractPost(document, { url: window.location.href });
  if (!post) return null;
  return { ...post, capturedAt: Date.now(), source: "self" };
}

/**
 * Détecte une capture entrante au chargement et appelle `onCapture`.
 * Nettoie ensuite l'URL pour éviter les ré-enregistrements au refresh.
 */
export function useYouTubeDetector(onCapture) {
  useEffect(() => {
    const incoming = parseHashCapture() || parseShareTarget() || parseSelfContext();
    if (incoming) {
      onCapture(incoming);
      // Nettoie hash + query sans recharger.
      const clean = window.location.origin + window.location.pathname;
      window.history.replaceState(null, "", clean);
    }
    // On ne dépend volontairement de rien : exécution unique au montage.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}
