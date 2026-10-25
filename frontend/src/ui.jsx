import { Component, useState } from "react";
import { api } from "./api";

export const arr = (x) => (Array.isArray(x) ? x : []);
export const text = (x) => (x === null || x === undefined ? "" : String(x));

export function useRequest() {
  const [state, set] = useState({ loading: false, error: null, result: null });
  const run = async (path, opts) => {
    set({ loading: true, error: null, result: null });
    try { set({ loading: false, error: null, result: await api(path, opts) }); }
    catch (e) { set({ loading: false, error: e.message, result: null }); }
  };
  const reset = () => set({ loading: false, error: null, result: null });
  return { ...state, run, reset };
}

export const Panel = ({ title, children, className = "" }) => (
  <section className={`rounded-2xl border-2 border-ink bg-paper shadow-[5px_5px_0_0_#111] ${className}`}>
    {title && (
      <h2 className="flex items-center gap-2.5 px-5 pt-4 text-xl text-ink">
        <span aria-hidden="true" className="h-4 w-4 rounded-sm border-2 border-ink bg-tone" />{title}
      </h2>
    )}
    <div className={`p-5 ${title ? "pt-3" : ""}`}>{children}</div>
  </section>
);

const inputCls = "w-full rounded-lg border-2 border-ink bg-white px-3 py-2 text-base text-ink placeholder:text-mute/70 transition focus:outline-none focus:shadow-[3px_3px_0_0_var(--tone)]";
export const Field = ({ label, hint, children }) => (
  <label className="block">
    <span className="mb-1.5 block font-mono text-xs font-bold uppercase tracking-wider text-ink">{label}</span>
    {children}
    {hint && <span className="mt-1 block text-sm text-mute">{hint}</span>}
  </label>
);
export const Input = (p) => <input className={inputCls} {...p} />;
export const Select = ({ options, ...p }) => (
  <select className={inputCls} {...p}>
    {options.map((o) => <option key={o.value ?? o} value={o.value ?? o}>{o.label ?? o}</option>)}
  </select>
);
export const Textarea = (p) => <textarea className={inputCls} {...p} />;

export const Button = ({ loading, children, secondary, ...p }) => (
  <button
    disabled={loading || p.disabled}
    className={`btn inline-flex items-center gap-2 rounded-lg border-2 border-ink px-4 py-2 font-bold shadow-[3px_3px_0_0_#111] disabled:opacity-40 disabled:shadow-none ${
      secondary ? "bg-white text-ink hover:bg-tint" : "bg-tone text-on-tone"}`}
    {...p}
  >
    {loading && <Spinner small />}{children}
  </button>
);

export const Spinner = ({ small }) => (
  <span role="status" aria-label="Loading"
    className={`spin inline-block rounded-full border-2 border-current border-t-transparent ${small ? "h-4 w-4" : "h-6 w-6"}`} />
);

export const ErrorBox = ({ message }) => message ? (
  <div role="alert" className="rounded-xl border-2 border-bad bg-bad/10 px-4 py-3 font-medium text-bad">{message}</div>
) : null;

export const Notice = ({ children }) => children ? (
  <div role="status" className="rounded-xl border-2 border-warn bg-warn/10 px-4 py-3 font-medium text-warn">{children}</div>
) : null;

const MODE = {
  demo: ["Demo data", "bg-warn/15 text-warn border-warn"],
  ai: ["AI generated", "bg-good/15 text-good border-good"],
  partial: ["Partial result: AI part unavailable", "bg-warn/15 text-warn border-warn"],
};
export function ResultMeta({ result }) {
  if (!result) return null;
  const [label, cls] = MODE[result.mode] || MODE.ai;
  return (
    <div className="mb-3 space-y-2">
      <span className={`inline-block rounded-full border-2 px-2.5 py-0.5 font-mono text-xs font-bold uppercase tracking-wide ${cls}`}>{label}</span>
      {arr(result.notes).map((n, i) => <p key={i} className="text-sm text-mute">{n}</p>)}
    </div>
  );
}

export const List = ({ items }) => (
  <ul className="mt-2 list-disc space-y-1 pl-5 marker:text-ink">{arr(items).map((t, i) => <li key={i}>{text(t)}</li>)}</ul>
);

/** Heading + bullet list; renders nothing when there are no items. */
export const Block = ({ title, items, note }) => arr(items).length ? (
  <div>
    <h3 className="inline-block rounded-md border-2 border-ink bg-tone px-2 py-0.5 font-mono text-xs font-bold uppercase tracking-wider text-on-tone">{title}</h3>
    {note && <p className="mt-2 mb-1 text-sm text-mute">{note}</p>}
    <List items={items} />
  </div>
) : null;

export const Empty = ({ children }) => (
  <div className="rounded-2xl border-2 border-dashed border-ink/50 bg-white/60 p-6 text-mute">{children}</div>
);

export const inr = (n) => (n === null || n === undefined || Number.isNaN(Number(n)) ? "n/a" : `₹ ${Number(n).toLocaleString("en-IN")}`);
export const label = (s) => text(s).replace(/_/g, " ");

/** Keeps one broken result from blanking the whole app. */
export class ErrorBoundary extends Component {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(err) { console.error("UI render error", err); }
  render() {
    return this.state.failed ? (
      <div role="alert" className="rounded-xl border-2 border-bad bg-bad/10 p-4 font-medium text-bad">
        Something went wrong while showing this result. Please try again.{" "}
        <button className="underline" onClick={() => this.setState({ failed: false })}>Dismiss</button>
      </div>
    ) : this.props.children;
  }
}