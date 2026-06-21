const STYLES = {
  success: "bg-emerald-600",
  error: "bg-[var(--color-brand)]",
  info: "bg-slate-800 dark:bg-slate-700",
};

export default function Toast({ toast }) {
  if (!toast) return null;
  return (
    <div className="safe-bottom pointer-events-none fixed inset-x-0 bottom-16 z-50 flex justify-center px-4">
      <div
        className={`animate-toast pointer-events-auto max-w-sm rounded-xl px-4 py-2.5 text-center text-sm font-medium text-white shadow-lg ${
          STYLES[toast.type] || STYLES.info
        }`}
        role="status"
      >
        {toast.message}
      </div>
    </div>
  );
}
