import { useCallback, useEffect, useRef, useState } from "react";
import Navbar from "./components/Navbar.jsx";
import HistoryList from "./components/HistoryList.jsx";
import OverlayDetector from "./components/OverlayDetector.jsx";
import ExportSettings from "./components/ExportSettings.jsx";
import MarkdownPreview from "./components/MarkdownPreview.jsx";
import Toast from "./components/Toast.jsx";
import { useScraper } from "./hooks/useScraper.js";
import { useYouTubeDetector } from "./hooks/useYouTubeDetector.js";
import { useLocalStorage } from "./hooks/useLocalStorage.js";
import { wordCount } from "./utils/parser.js";

const DEFAULT_SETTINGS = {
  defaultFormat: "md",
  includeMeta: true,
  includeTags: true,
};

export default function App() {
  const [tab, setTab] = useState("history");
  const [selected, setSelected] = useState(null);
  const [toast, setToast] = useState(null);
  const [unseen, setUnseen] = useState(0);

  const [theme, setTheme] = useLocalStorage("yt-theme", "dark");
  const [autoMode, setAutoMode] = useLocalStorage("yt-auto", true);
  const [settings, setSettings] = useLocalStorage("yt-settings", DEFAULT_SETTINGS);

  const { posts, loading, capture, remove, clearAll, setTags } = useScraper();
  const toastTimer = useRef(null);

  const showToast = useCallback((message, type = "info") => {
    setToast({ message, type, id: Date.now() });
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 2600);
  }, []);

  // Applique le thème au document.
  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  // Badge d'app (PWA) selon les posts non vus.
  useEffect(() => {
    if (!("setAppBadge" in navigator)) return;
    if (unseen > 0) navigator.setAppBadge(unseen).catch(() => {});
    else navigator.clearAppBadge?.().catch(() => {});
  }, [unseen]);

  const handleCapture = useCallback(
    async (data) => {
      const { record, duplicate } = await capture(data);
      if (duplicate) {
        showToast("Post déjà capturé — mis à jour", "info");
      } else {
        showToast("Nouveau post capturé ✅", "success");
        setUnseen((n) => n + 1);
      }
      setSelected(record);
      return record;
    },
    [capture, showToast],
  );

  // Détection d'une capture entrante (bookmarklet / partage / contexte YouTube).
  useYouTubeDetector(
    useCallback(
      (incoming) => {
        if (!autoMode) {
          showToast("Post détecté (Mode Actif désactivé)", "info");
          return;
        }
        handleCapture(incoming);
      },
      [autoMode, handleCapture, showToast],
    ),
  );

  const handleClearAll = useCallback(async () => {
    if (!posts.length) return;
    if (!window.confirm("Effacer définitivement tous les posts capturés ?")) return;
    await clearAll();
    setUnseen(0);
    showToast("Historique effacé", "success");
  }, [clearAll, posts.length, showToast]);

  function openTab(next) {
    if (next === "history") setUnseen(0);
    setTab(next);
  }

  const stats = {
    posts: posts.length,
    words: posts.reduce((sum, p) => sum + wordCount(p.content), 0),
  };

  return (
    <div className="mx-auto flex min-h-full max-w-md flex-col bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <header className="safe-top sticky top-0 z-20 flex items-center gap-2 border-b border-black/10 bg-white/90 px-4 py-3 backdrop-blur dark:border-white/10 dark:bg-slate-900/90">
        <span className="text-xl">📺</span>
        <h1 className="flex-1 text-base font-bold">YouTube Post Scraper</h1>
        {autoMode && (
          <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300">
            ● Actif
          </span>
        )}
      </header>

      <main className="flex-1">
        {tab === "history" && (
          <HistoryList
            posts={posts}
            loading={loading}
            onOpen={setSelected}
            onDelete={remove}
            onClearAll={handleClearAll}
          />
        )}
        {tab === "capture" && (
          <OverlayDetector
            autoMode={autoMode}
            onToggleAuto={setAutoMode}
            onDemoCapture={handleCapture}
            onToast={showToast}
          />
        )}
        {tab === "settings" && (
          <ExportSettings
            settings={settings}
            onChange={setSettings}
            theme={theme}
            onTheme={setTheme}
            stats={stats}
            onClearAll={handleClearAll}
          />
        )}
      </main>

      <Navbar active={tab} onChange={openTab} badge={unseen} />

      {selected && (
        <MarkdownPreview
          post={selected}
          settings={settings}
          onClose={() => setSelected(null)}
          onSaveTags={(id, tags) => {
            setTags(id, tags);
            setSelected((s) => (s && s.id === id ? { ...s, tags } : s));
          }}
          onToast={showToast}
        />
      )}

      <Toast toast={toast} />
    </div>
  );
}
