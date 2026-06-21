const TABS = [
  { id: "history", label: "Historique", icon: "🗂️" },
  { id: "capture", label: "Capturer", icon: "🎯" },
  { id: "settings", label: "Réglages", icon: "⚙️" },
];

export default function Navbar({ active, onChange, badge = 0 }) {
  return (
    <nav className="safe-bottom sticky bottom-0 z-20 border-t border-black/10 bg-white/90 backdrop-blur dark:border-white/10 dark:bg-slate-900/90">
      <ul className="mx-auto flex max-w-md">
        {TABS.map((tab) => {
          const isActive = active === tab.id;
          return (
            <li key={tab.id} className="flex-1">
              <button
                type="button"
                onClick={() => onChange(tab.id)}
                aria-current={isActive ? "page" : undefined}
                className={`flex w-full flex-col items-center gap-0.5 py-2.5 text-xs font-medium transition-colors ${
                  isActive
                    ? "text-[var(--color-brand)]"
                    : "text-slate-500 dark:text-slate-400"
                }`}
              >
                <span className="relative text-lg leading-none">
                  {tab.icon}
                  {tab.id === "history" && badge > 0 && (
                    <span className="absolute -right-3 -top-1 min-w-4 rounded-full bg-[var(--color-brand)] px-1 text-[10px] font-bold leading-4 text-white">
                      {badge > 99 ? "99+" : badge}
                    </span>
                  )}
                </span>
                {tab.label}
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
