// Formatage Markdown / TXT / JSON d'un post capturé.

import { wordCount } from "./parser.js";

function fmtDate(ts) {
  const d = ts ? new Date(ts) : new Date();
  return d.toLocaleDateString("fr-FR", {
    day: "2-digit",
    month: "long",
    year: "numeric",
  });
}

function fmtTime(ts) {
  const d = ts ? new Date(ts) : new Date();
  return d.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
}

/**
 * Construit le Markdown d'un post.
 * @param {object} post   - { content, author, date, url, capturedAt, tags }
 * @param {object} options - { includeMeta, includeTags, title }
 */
export function toMarkdown(post = {}, options = {}) {
  const {
    includeMeta = true,
    includeTags = true,
    title = "📺 Post YouTube",
  } = options;

  const lines = [`# ${title}`, ""];

  if (post.author) lines.push(`**Auteur :** ${post.author}`);
  if (post.url) lines.push(`**URL :** ${post.url}`);
  if (post.date) lines.push(`**Date du post :** ${post.date}`);
  if (post.engagement?.likes) lines.push(`**J'aime :** ${post.engagement.likes}`);
  if (lines.length > 2) lines.push("");

  lines.push("## Contenu", "", (post.content || "").trim(), "");

  if (includeTags && Array.isArray(post.tags) && post.tags.length) {
    lines.push("**Tags :** " + post.tags.map((t) => `#${t}`).join(" "), "");
  }

  if (includeMeta) {
    lines.push("---", "");
    lines.push(
      `_Extrait le ${fmtDate(post.capturedAt)} à ${fmtTime(
        post.capturedAt,
      )} via YouTube Scraper_`,
    );
  }

  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim() + "\n";
}

/** Version texte brut (sans syntaxe Markdown). */
export function toPlainText(post = {}) {
  const parts = [];
  if (post.author) parts.push(`Auteur : ${post.author}`);
  if (post.url) parts.push(`URL : ${post.url}`);
  if (post.date) parts.push(`Date : ${post.date}`);
  parts.push("", (post.content || "").trim(), "");
  parts.push(`Extrait le ${fmtDate(post.capturedAt)} à ${fmtTime(post.capturedAt)}`);
  return parts.join("\n").trim() + "\n";
}

/** Version JSON complète (métadonnées incluses). */
export function toJson(post = {}) {
  return JSON.stringify(
    {
      author: post.author || null,
      url: post.url || null,
      date: post.date || null,
      content: post.content || "",
      tags: post.tags || [],
      engagement: post.engagement || {},
      capturedAt: post.capturedAt || Date.now(),
      wordCount: wordCount(post.content || ""),
      source: "youtube-scraper-pwa",
    },
    null,
    2,
  );
}

export function exportPost(post, format, options) {
  switch (format) {
    case "txt":
      return { data: toPlainText(post), mime: "text/plain", ext: "txt" };
    case "json":
      return { data: toJson(post), mime: "application/json", ext: "json" };
    case "md":
    default:
      return {
        data: toMarkdown(post, options),
        mime: "text/markdown",
        ext: "md",
      };
  }
}
