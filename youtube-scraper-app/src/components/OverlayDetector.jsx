import { useMemo } from "react";
import { buildBookmarklet } from "../utils/bookmarklet.js";
import { useClipboard } from "../hooks/useClipboard.js";

const DEMO = {
  content:
    "Merci à tous pour vos 100 000 abonnés ! 🎉 On prépare une grosse vidéo sur la micronutrition et la prévention santé. Dites-moi en commentaire les sujets qui vous intéressent.",
  author: "Ma Chaîne Santé",
  date: "il y a 2 jours",
  url: "https://www.youtube.com/post/UgkxDEMO_EXAMPLE_1234567890",
};

export default function OverlayDetector({ autoMode, onToggleAuto, onDemoCapture, onToast }) {
  const { copy } = useClipboard();

  const appUrl = useMemo(() => window.location.origin + window.location.pathname, []);
  const bookmarklet = useMemo(() => buildBookmarklet(appUrl), [appUrl]);

  async function copyBookmarklet() {
    const ok = await copy(bookmarklet);
    onToast(
      ok ? "Code copié — colle-le dans un favori" : "Échec de la copie",
      ok ? "success" : "error",
    );
  }

  return (
    <div className="flex flex-col gap-4 px-4 pb-6 pt-3">
      {/* Mode actif */}
      <section className="rounded-2xl border border-black/5 bg-white p-4 shadow-sm dark:border-white/5 dark:bg-slate-800">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold">Mode Actif</h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Capture automatiquement à l'ouverture d'un post (toast + badge).
            </p>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={autoMode}
            onClick={() => onToggleAuto(!autoMode)}
            className={`relative h-7 w-12 shrink-0 rounded-full transition-colors ${
              autoMode ? "bg-[var(--color-brand)]" : "bg-slate-300 dark:bg-slate-600"
            }`}
          >
            <span
              className={`absolute top-0.5 h-6 w-6 rounded-full bg-white shadow transition-all ${
                autoMode ? "left-[1.375rem]" : "left-0.5"
              }`}
            />
          </button>
        </div>
      </section>

      {/* Bookmarklet : le coeur du scraping */}
      <section className="rounded-2xl border border-black/5 bg-white p-4 shadow-sm dark:border-white/5 dark:bg-slate-800">
        <h2 className="text-sm font-semibold">1 · Installer la capture</h2>
        <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
          YouTube bloque toute lecture externe (CORS). Pour extraire le post
          <strong> dans le contexte de la page</strong>, sans proxy ni copier-coller,
          utilise ce bouton de capture.
        </p>

        <div className="mt-3 flex flex-col gap-2">
          <a
            href={bookmarklet}
            onClick={(e) => e.preventDefault()}
            draggable
            className="block select-none rounded-xl bg-[var(--color-brand)] px-4 py-3 text-center text-sm font-bold text-white"
          >
            🎯 Capturer ce post
          </a>
          <p className="text-center text-[11px] text-slate-400">
            Ordinateur : glisse ce bouton dans la barre de favoris.
            <br />
            Mobile : copie le code ci-dessous et crée un favori avec.
          </p>
          <button
            type="button"
            onClick={copyBookmarklet}
            className="rounded-xl bg-black/5 py-2.5 text-xs font-medium dark:bg-white/10"
          >
            Copier le code du favori
          </button>
        </div>
      </section>

      {/* Utilisation */}
      <section className="rounded-2xl border border-black/5 bg-white p-4 shadow-sm dark:border-white/5 dark:bg-slate-800">
        <h2 className="text-sm font-semibold">2 · Capturer un post</h2>
        <ol className="mt-2 list-decimal space-y-1 pl-5 text-xs text-slate-600 dark:text-slate-300">
          <li>Ouvre un post YouTube (youtube.com/post/…).</li>
          <li>Touche le favori « Capturer ce post ».</li>
          <li>L'app s'ouvre et enregistre le post automatiquement.</li>
        </ol>
        <p className="mt-3 rounded-lg bg-black/5 p-2 text-[11px] text-slate-500 dark:bg-white/10 dark:text-slate-400">
          📲 Alternative mobile : utilise le menu <strong>Partager</strong> de
          YouTube et choisis <strong>YT Scraper</strong> (l'app s'enregistre comme
          cible de partage une fois installée).
        </p>
      </section>

      {/* Démo */}
      <section className="rounded-2xl border border-dashed border-black/15 p-4 dark:border-white/15">
        <h2 className="text-sm font-semibold">Tester sans YouTube</h2>
        <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
          Capture un post d'exemple pour voir le formatage Markdown et l'export.
        </p>
        <button
          type="button"
          onClick={() => onDemoCapture({ ...DEMO, capturedAt: Date.now(), source: "demo" })}
          className="mt-3 w-full rounded-xl bg-slate-800 py-2.5 text-xs font-semibold text-white dark:bg-slate-700"
        >
          Capturer un exemple
        </button>
      </section>
    </div>
  );
}
