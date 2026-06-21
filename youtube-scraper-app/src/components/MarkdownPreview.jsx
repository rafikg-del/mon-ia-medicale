import { useMemo, useState } from "react";
import { exportPost } from "../utils/markdown.js";
import { useClipboard } from "../hooks/useClipboard.js";

const FORMATS = [
  { id: "md", label: "Markdown" },
  { id: "txt", label: "Texte" },
  { id: "json", label: "JSON" },
];

function downloadFile(filename, data, mime) {
  const blob = new Blob([data], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function MarkdownPreview({ post, settings, onClose, onSaveTags, onToast }) {
  const [format, setFormat] = useState(settings.defaultFormat || "md");
  const [tagInput, setTagInput] = useState((post.tags || []).join(", "));
  const { copy } = useClipboard();

  const exported = useMemo(
    () =>
      exportPost(
        { ...post, tags: parseTags(tagInput) },
        format,
        {
          includeMeta: settings.includeMeta,
          includeTags: settings.includeTags,
        },
      ),
    [post, tagInput, format, settings.includeMeta, settings.includeTags],
  );

  const baseName = useMemo(() => {
    const a = (post.author || "post").replace(/[^a-z0-9]+/gi, "-").toLowerCase();
    return `yt-${a}-${post.id}`;
  }, [post]);

  async function handleCopy() {
    const ok = await copy(exported.data);
    onToast(ok ? "Copié dans le presse-papiers ✅" : "Échec de la copie", ok ? "success" : "error");
  }

  function handleDownload() {
    downloadFile(`${baseName}.${exported.ext}`, exported.data, exported.mime);
    onToast(`Téléchargé : ${baseName}.${exported.ext}`, "success");
  }

  async function handleShare() {
    if (navigator.share) {
      try {
        await navigator.share({ title: "Post YouTube", text: exported.data });
      } catch {
        /* annulé par l'utilisateur */
      }
    } else {
      const ok = await copy(exported.data);
      onToast(ok ? "Partage indisponible — copié à la place" : "Partage indisponible", "info");
    }
  }

  function saveTags() {
    onSaveTags(post.id, parseTags(tagInput));
    onToast("Tags enregistrés", "success");
  }

  return (
    <div className="fixed inset-0 z-40 flex flex-col bg-slate-50 dark:bg-slate-950">
      <header className="safe-top flex items-center gap-2 border-b border-black/10 px-3 py-3 dark:border-white/10">
        <button
          type="button"
          onClick={onClose}
          className="rounded-lg px-2 py-1 text-sm font-medium text-slate-500"
        >
          ← Retour
        </button>
        <h2 className="flex-1 truncate text-center text-sm font-semibold">
          {post.author || "Post YouTube"}
        </h2>
        <span className="w-14" />
      </header>

      <div className="flex gap-1 px-3 pt-3">
        {FORMATS.map((f) => (
          <button
            key={f.id}
            type="button"
            onClick={() => setFormat(f.id)}
            className={`flex-1 rounded-lg py-1.5 text-xs font-medium transition-colors ${
              format === f.id
                ? "bg-[var(--color-brand)] text-white"
                : "bg-black/5 text-slate-600 dark:bg-white/10 dark:text-slate-300"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-auto px-3 py-3">
        <pre className="md-raw rounded-xl border border-black/10 bg-white p-3 dark:border-white/10 dark:bg-slate-900">
          {exported.data}
        </pre>

        {post.url && (
          <a
            href={post.url}
            target="_blank"
            rel="noreferrer"
            className="mt-3 block truncate text-xs text-[var(--color-brand)] underline"
          >
            {post.url}
          </a>
        )}

        <div className="mt-4">
          <label className="text-xs font-medium text-slate-500 dark:text-slate-400">
            Tags (séparés par des virgules)
          </label>
          <div className="mt-1 flex gap-2">
            <input
              value={tagInput}
              onChange={(e) => setTagInput(e.target.value)}
              placeholder="santé, nutrition…"
              className="flex-1 rounded-lg border border-black/10 bg-white px-3 py-2 text-sm dark:border-white/10 dark:bg-slate-800"
            />
            <button
              type="button"
              onClick={saveTags}
              className="rounded-lg bg-black/5 px-3 text-xs font-medium dark:bg-white/10"
            >
              Enregistrer
            </button>
          </div>
        </div>
      </div>

      <div className="safe-bottom grid grid-cols-3 gap-2 border-t border-black/10 p-3 dark:border-white/10">
        <button
          type="button"
          onClick={handleCopy}
          className="rounded-xl bg-[var(--color-brand)] py-3 text-sm font-semibold text-white"
        >
          Copier
        </button>
        <button
          type="button"
          onClick={handleDownload}
          className="rounded-xl bg-slate-800 py-3 text-sm font-semibold text-white dark:bg-slate-700"
        >
          Télécharger
        </button>
        <button
          type="button"
          onClick={handleShare}
          className="rounded-xl bg-black/5 py-3 text-sm font-semibold dark:bg-white/10"
        >
          Partager
        </button>
      </div>
    </div>
  );
}

function parseTags(input) {
  return Array.from(
    new Set(
      String(input)
        .split(",")
        .map((t) => t.trim().replace(/^#/, ""))
        .filter(Boolean),
    ),
  );
}
