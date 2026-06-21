// Nettoyage de texte et décodage des entités HTML.
// Ces fonctions ne dépendent PAS du DOM pour le décodage (sauf fallback),
// afin de pouvoir tourner dans un bookmarklet minimal comme dans l'app.

const NAMED_ENTITIES = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
  nbsp: " ",
  hellip: "…",
  mdash: "—",
  ndash: "–",
  rsquo: "’",
  lsquo: "‘",
  ldquo: "“",
  rdquo: "”",
  copy: "©",
  reg: "®",
  trade: "™",
  deg: "°",
};

/** Décode les entités HTML (&amp; &#39; &#x2026; &nbsp; …). */
export function decodeHtmlEntities(input = "") {
  return String(input)
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) =>
      String.fromCodePoint(parseInt(hex, 16)),
    )
    .replace(/&#(\d+);/g, (_, dec) => String.fromCodePoint(parseInt(dec, 10)))
    .replace(/&([a-zA-Z][a-zA-Z0-9]*);/g, (m, name) =>
      Object.prototype.hasOwnProperty.call(NAMED_ENTITIES, name)
        ? NAMED_ENTITIES[name]
        : m,
    );
}

/** Supprime les balises HTML résiduelles d'une chaîne. */
export function stripTags(input = "") {
  return String(input).replace(/<[^>]*>/g, " ");
}

/**
 * Nettoie un texte extrait : décode les entités, normalise les espaces et
 * les retours à la ligne, supprime les espaces superflus.
 */
export function cleanText(input = "") {
  let text = decodeHtmlEntities(stripTags(String(input)));
  text = text
    .replace(/\r\n?/g, "\n") // CRLF -> LF
    .replace(/[ \t ]+/g, " ") // espaces multiples / insécables
    .replace(/ *\n */g, "\n") // espaces autour des sauts de ligne
    .replace(/\n{3,}/g, "\n\n"); // max une ligne vide
  return text.trim();
}

/** Tronque proprement un texte pour les aperçus. */
export function truncate(text = "", max = 140) {
  const t = String(text).trim();
  if (t.length <= max) return t;
  return t.slice(0, max - 1).trimEnd() + "…";
}

/** Compte les mots d'un texte. */
export function wordCount(text = "") {
  const t = String(text).trim();
  if (!t) return 0;
  return t.split(/\s+/).length;
}
