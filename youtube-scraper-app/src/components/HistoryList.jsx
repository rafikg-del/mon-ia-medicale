import { useMemo, useState } from "react";
import { truncate, wordCount } from "../utils/parser.js";

function relTime(ts) {
  const diff = Date.now() - (ts || 0);
  const min = Math.round(diff / 60000);
  if (min < 1) return "à l'instant";
  if (min < 60) return `il y a ${min} min`;
  const h = Math.round(min / 60);
  if (h < 24) return `il y a ${h} h`;
  const d = Math.round(h / 24);
  return `il y a ${d} j`;
}

export default function HistoryList({ posts, loading, onOpen, onDelete, onClearAll }) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return posts;
    return posts.filter(
      (p) =>
        (p.content || "").toLowerCase().includes(q) ||
        (p.author || "").toLowerCase().includes(q),
    );
  }, [posts, query]);

  if (loading) {
    return <p className="px-4 py-10 text-center text-slate-400">Chargement…</p>;
  }

  if (!posts.length) {
    return (
      <div className="flex flex-col items-center gap-3 px-6 py-16 text-center">
        <span className="text-5xl">🗂️</span>
        <h2 className="text-lg font-semibold">Aucun post capturé</h2>
        <p className="max-w-xs text-sm text-slate-500 dark:text-slate-400">
          Va dans l'onglet <strong>Capturer</strong> pour installer l'outil de
          capture, puis ouvre un post YouTube.
        </p>
      </div>
    );
  }

  return (
    <div className="px-3 pb-4">
      <div className="sticky top-0 z-10 -mx-3 mb-2 bg-slate-50/90 px-3 pb-2 pt-1 backdrop-blur dark:bg-slate-950/90">
        <input
          type="search"
          inputMode="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Rechercher (texte, auteur)…"
          className="w-full rounded-xl border border-black/10 bg-white px-3 py-2 text-sm outline-none focus:border-[var(--color-brand)] dark:border-white/10 dark:bg-slate-800"
        />
        <div className="mt-2 flex items-center justify-between px-1 text-xs text-slate-500 dark:text-slate-400">
          <span>
            {filtered.length} / {posts.length} post{posts.length > 1 ? "s" : ""}
          </span>
          <button
            type="button"
            onClick={onClearAll}
            className="font-medium text-[var(--color-brand)]"
          >
            Tout effacer
          </button>
        </div>
      </div>

      <ul className="flex flex-col gap-2">
        {filtered.map((p) => (
          <li key={p.id}>
            <article className="rounded-2xl border border-black/5 bg-white p-3 shadow-sm dark:border-white/5 dark:bg-slate-800">
              <button
                type="button"
                onClick={() => onOpen(p)}
                className="w-full text-left"
              >
                <header className="mb-1 flex items-center justify-between gap-2">
                  <span className="truncate text-sm font-semibold">
                    {p.author || "Post YouTube"}
                  </span>
                  <span className="shrink-0 text-[11px] text-slate-400">
                    {relTime(p.capturedAt)}
                  </span>
                </header>
                <p className="text-sm text-slate-600 dark:text-slate-300">
                  {truncate(p.content, 160)}
                </p>
                <footer className="mt-2 flex items-center gap-2 text-[11px] text-slate-400">
                  <span>{wordCount(p.content)} mots</span>
                  {p.tags?.length ? <span>· {p.tags.map((t) => `#${t}`).join(" ")}</span> : null}
                </footer>
              </button>
              <div className="mt-2 flex justify-end">
                <button
                  type="button"
                  onClick={() => onDelete(p.id)}
                  className="rounded-lg px-2 py-1 text-xs text-slate-400 hover:text-[var(--color-brand)]"
                >
                  Supprimer
                </button>
              </div>
            </article>
          </li>
        ))}
      </ul>
    </div>
  );
}
