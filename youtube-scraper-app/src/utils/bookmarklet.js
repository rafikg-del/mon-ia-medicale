// Génère le bookmarklet qui s'exécute DANS la page YouTube.
//
// C'est la pièce qui contourne la limite same-origin : impossible pour une PWA
// de lire le DOM de youtube.com à distance, mais un bookmarklet exécuté sur la
// page YouTube a un accès complet au DOM, sans fetch ni proxy ni CORS. Il
// extrait le post puis ouvre la PWA avec les données dans le hash de l'URL.

// Le corps est volontairement écrit en ES5 compact et auto-suffisant.
function bookmarkletSource(appUrl) {
  // NOTE: ce code est sérialisé tel quel dans le bookmarklet.
  const fn = function () {
    var SEL = [
      'meta[property="og:description"]',
      'meta[name="description"]',
      "ytd-backstage-post-renderer #content-text",
      "#content-text",
      "[data-comment-text]",
      '[role="article"]',
      ".post-content",
    ];
    function read(el) {
      if (!el) return "";
      if (el.tagName === "META") return el.getAttribute("content") || "";
      return el.innerText || el.textContent || "";
    }
    function clean(s) {
      var d = document.createElement("textarea");
      d.innerHTML = String(s || "");
      return d.value
        .replace(/\r\n?/g, "\n")
        .replace(/[ \t ]+/g, " ")
        .replace(/ *\n */g, "\n")
        .replace(/\n{3,}/g, "\n\n")
        .trim();
    }
    var content = "";
    for (var i = 0; i < SEL.length; i++) {
      try {
        var v = clean(read(document.querySelector(SEL[i])));
        if (v) {
          content = v;
          break;
        }
      } catch (e) {}
    }
    if (!content) {
      var main = document.querySelector('[role="main"]') || document.body;
      content = clean(main ? main.innerText : "").slice(0, 4000);
    }
    if (!content) {
      alert("Aucun post YouTube détecté sur cette page.");
      return;
    }
    var author = clean(
      read(document.querySelector("#author-text")) ||
        (document.querySelector('meta[property="og:title"]') || {}).content ||
        "",
    ).replace(/\s*[-–—]\s*YouTube\s*$/i, "");
    var dateEl = document.querySelector("#published-time-text a, #published-time-text, time");
    var date = dateEl ? clean(dateEl.getAttribute("datetime") || dateEl.textContent) : "";
    var payload = {
      content: content,
      author: author,
      date: date,
      url: location.href,
      capturedAt: Date.now(),
    };
    var hash = "#capture=" + encodeURIComponent(JSON.stringify(payload));
    window.open("__APP_URL__" + hash, "_blank");
  };

  const body = `(${fn.toString()})();`;
  return body.replace("__APP_URL__", appUrl);
}

export function buildBookmarklet(appUrl) {
  return "javascript:" + encodeURIComponent(bookmarkletSource(appUrl));
}
