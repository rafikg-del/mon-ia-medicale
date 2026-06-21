// Extraction DOM d'un post YouTube.
//
// IMPORTANT — limite technique du web : une PWA hébergée sur son propre domaine
// ne peut PAS lire le DOM de youtube.com (politique same-origin du navigateur).
// L'extraction réelle s'exécute donc DANS le contexte de la page YouTube via le
// bookmarklet (voir bookmarklet.js), qui réutilise exactement la même logique.
// Cette fonction est écrite pour fonctionner sur n'importe quel `document`
// (celui de YouTube côté bookmarklet, ou un document parsé côté app).

import { cleanText } from "./parser.js";

// Sélecteurs par ordre de priorité (cf. spécification).
const CONTENT_SELECTORS = [
  'meta[property="og:description"]',
  'meta[name="description"]',
  "ytd-backstage-post-renderer #content-text",
  "ytd-backstage-post-renderer yt-formatted-string#content-text",
  "#content-text",
  "[data-comment-text]",
  '[role="article"]',
  ".post-content",
];

const AUTHOR_SELECTORS = [
  'ytd-backstage-post-renderer #author-text',
  "#author-text",
  'link[itemprop="name"]',
  'meta[property="og:title"]',
  "ytd-channel-name #text",
  "#channel-name #text",
];

const DATE_SELECTORS = [
  "ytd-backstage-post-renderer #published-time-text a",
  "#published-time-text",
  'meta[itemprop="datePublished"]',
  "time",
];

function readSelector(doc, selector) {
  const el = doc.querySelector(selector);
  if (!el) return "";
  if (el.tagName === "META") return el.getAttribute("content") || "";
  if (el.tagName === "LINK") return el.getAttribute("content") || el.getAttribute("href") || "";
  if (el.tagName === "TIME") return el.getAttribute("datetime") || el.textContent || "";
  if (el.hasAttribute("data-comment-text"))
    return el.getAttribute("data-comment-text") || el.textContent || "";
  return el.innerText || el.textContent || "";
}

function firstMatch(doc, selectors) {
  for (const sel of selectors) {
    try {
      const value = cleanText(readSelector(doc, sel));
      if (value) return value;
    } catch {
      /* sélecteur invalide -> on continue */
    }
  }
  return "";
}

/**
 * Extrait un post depuis un document.
 * @param {Document} doc - document YouTube (ou parsé).
 * @param {object} ctx   - contexte additionnel { url }.
 * @returns {{content, author, date, url, engagement} | null}
 */
export function extractPost(doc = document, ctx = {}) {
  const url = ctx.url || doc?.location?.href || "";

  let content = firstMatch(doc, CONTENT_SELECTORS);

  // Fallback ultime : texte brut de la zone d'article, borné.
  if (!content) {
    const body = doc.querySelector('[role="main"]') || doc.body;
    if (body) content = cleanText(body.innerText || body.textContent || "").slice(0, 4000);
  }

  if (!content) return null;

  const author = cleanAuthor(firstMatch(doc, AUTHOR_SELECTORS));
  const date = firstMatch(doc, DATE_SELECTORS);
  const engagement = extractEngagement(doc);

  return { content, author, date, url, engagement };
}

function cleanAuthor(raw) {
  if (!raw) return "";
  // "Nom - YouTube" / "Nom a publié…" -> "Nom"
  return raw
    .replace(/\s*[-–—]\s*YouTube\s*$/i, "")
    .replace(/\s+a publié.*/i, "")
    .trim();
}

function extractEngagement(doc) {
  const out = {};
  const like =
    doc.querySelector('#vote-count-middle, [aria-label*="J’aime"], [aria-label*="like" i]') || null;
  if (like) {
    const txt = cleanText(like.getAttribute("aria-label") || like.textContent || "");
    if (txt) out.likes = txt;
  }
  const comments = doc.querySelector('[aria-label*="commentaire" i], [aria-label*="comment" i]');
  if (comments) {
    const txt = cleanText(comments.getAttribute("aria-label") || "");
    if (txt) out.comments = txt;
  }
  return out;
}

export { CONTENT_SELECTORS, AUTHOR_SELECTORS, DATE_SELECTORS };
