const FORMATS = [
  { id: "md", label: "Markdown (.md)" },
  { id: "txt", label: "Texte (.txt)" },
  { id: "json", label: "JSON (.json)" },
];

function Row({ title, desc, children }) {
  return (
    <div className="flex items-center justify-between gap-3 py-3">
      <div>
        <p className="text-sm font-medium">{title}</p>
        {desc && <p className="text-xs text-slate-500 dark:text-slate-400">{desc}</p>}
      </div>
      {children}
    </div>
  );
}

function Toggle({ checked, onChange }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={`relative h-7 w-12 shrink-0 rounded-full transition-colors ${
        checked ? "bg-[var(--color-brand)]" : "bg-slate-300 dark:bg-slate-600"
      }`}
    >
      <span
        className={`absolute top-0.5 h-6 w-6 rounded-full bg-white shadow transition-all ${
          checked ? "left-[1.375rem]" : "left-0.5"
        }`}
      />
    </button>
  );
}

export default function ExportSettings({ settings, onChange, theme, onTheme, stats, onClearAll }) {
  const set = (patch) => onChange({ ...settings, ...patch });

  return (
    <div className="flex flex-col gap-4 px-4 pb-6 pt-3">
      <section className="rounded-2xl border border-black/5 bg-white px-4 shadow-sm dark:border-white/5 dark:bg-slate-800">
        <Row title="Format d'export par défaut">
          <select
            value={settings.defaultFormat}
            onChange={(e) => set({ defaultFormat: e.target.value })}
            className="rounded-lg border border-black/10 bg-white px-2 py-1.5 text-sm dark:border-white/10 dark:bg-slate-700"
          >
            {FORMATS.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label}
              </option>
            ))}
          </select>
        </Row>
        <hr className="border-black/5 dark:border-white/5" />
        <Row title="Inclure les métadonnées" desc="Date et heure d'extraction">
          <Toggle checked={settings.includeMeta} onChange={(v) => set({ includeMeta: v })} />
        </Row>
        <hr className="border-black/5 dark:border-white/5" />
        <Row title="Inclure les tags" desc="Ajoute les #tags au Markdown">
          <Toggle checked={settings.includeTags} onChange={(v) => set({ includeTags: v })} />
        </Row>
      </section>

      <section className="rounded-2xl border border-black/5 bg-white px-4 shadow-sm dark:border-white/5 dark:bg-slate-800">
        <Row title="Thème sombre">
          <Toggle checked={theme === "dark"} onChange={(v) => onTheme(v ? "dark" : "light")} />
        </Row>
      </section>

      <section className="rounded-2xl border border-black/5 bg-white p-4 shadow-sm dark:border-white/5 dark:bg-slate-800">
        <h2 className="text-sm font-semibold">Statistiques</h2>
        <div className="mt-3 grid grid-cols-2 gap-3 text-center">
          <div className="rounded-xl bg-black/5 py-3 dark:bg-white/10">
            <p className="text-2xl font-bold">{stats.posts}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">posts</p>
          </div>
          <div className="rounded-xl bg-black/5 py-3 dark:bg-white/10">
            <p className="text-2xl font-bold">{stats.words}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">mots</p>
          </div>
        </div>
      </section>

      <button
        type="button"
        onClick={onClearAll}
        className="rounded-2xl border border-[var(--color-brand)] py-3 text-sm font-semibold text-[var(--color-brand)]"
      >
        Effacer tout l'historique
      </button>

      <p className="px-2 text-center text-[11px] text-slate-400">
        YouTube Post Scraper · 100 % local & hors ligne · aucune donnée envoyée.
      </p>
    </div>
  );
}
