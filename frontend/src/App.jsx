import { useEffect, useState } from "react";
import { api } from "./api";
import { ErrorBoundary } from "./ui";
import Safety from "./features/Safety";
import Cracks from "./features/Cracks";
import Valuation from "./features/Valuation";
import Planning from "./features/Planning";
import Assistant from "./features/Assistant";
import Weather from "./features/Weather";
import Knowledge from "./features/Knowledge";

// Simple 24px line icons, one per tool.
const ICONS = {
  safety: "M5 15a7 7 0 0 1 14 0 M3 15h18 M12 8v4 M5 15v2h14v-2",
  cracks: "M4 4h16v16H4z M13 4l-3 5 4 3-3 4 1 4",
  valuation: "M5 21V5l7-3 7 3v16 M3 21h18 M9 9h2 M13 9h2 M9 13h2 M13 13h2 M10 21v-4h4v4",
  planning: "M3 3v18 M7 6h7 M10 12h9 M7 18h6",
  assistant: "M9 4h6v3H9z M7 5H5v16h14V5h-2 M9 12h6 M9 16h4",
  weather: "M7 18a4 4 0 0 1-.5-8A5.5 5.5 0 0 1 17 8.5 3.8 3.8 0 0 1 17 18z",
  knowledge: "M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z M4 21V5 M9 7h6",
};
const Icon = ({ id, cls = "h-5 w-5" }) => (
  <svg aria-hidden="true" viewBox="0 0 24 24" className={`${cls} shrink-0`} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d={ICONS[id]} />
  </svg>
);

// Brand mark: a black hard hat on a yellow square.
const Mark = () => (
  <svg aria-hidden="true" viewBox="0 0 32 32" className="h-10 w-10 shrink-0">
    <rect x="1" y="1" width="30" height="30" rx="7" fill="#facc15" stroke="#111" strokeWidth="2" />
    <path d="M7 20a9 9 0 0 1 18 0z" fill="#111" />
    <rect x="5" y="20" width="22" height="3.5" rx="1.5" fill="#111" />
    <rect x="14.5" y="8.5" width="3" height="6" rx="1" fill="#facc15" />
  </svg>
);

const FEATURES = [
  { id: "safety", tone: "#4ade80", name: "Site safety", about: "AI reviews a site photo for PPE and visible hazards. Not a certified inspection.", C: Safety },
  { id: "cracks", tone: "#fb923c", name: "Crack analysis", about: "AI describes a crack photo, possible causes and next steps. Not a structural diagnosis.", C: Cracks },
  { id: "valuation", tone: "#a78bfa", name: "Property valuation", about: "An indicative, AI-generated value range, rent and return. Not verified market data.", C: Valuation },
  { id: "planning", tone: "#22d3ee", name: "Project planning", about: "A computed timeline plus approximate AI-estimated materials and costs.", C: Planning },
  { id: "assistant", tone: "#f472b6", name: "Daily site assistant", about: "Ask site questions or turn today's activities into a daily plan.", C: Assistant },
  { id: "weather", tone: "#60a5fa", name: "Weather advisory", about: "Live forecast or your own conditions, with GO / CAUTION / NO-GO guidance.", C: Weather },
  { id: "knowledge", tone: "#facc15", name: "Knowledge assistant", about: "Materials, methods and terminology explained.", C: Knowledge },
];

export default function App() {
  const [active, setActive] = useState("safety");
  const [health, setHealth] = useState(null);
  useEffect(() => {
    api("/api/health").then((r) => setHealth(r.data.ai)).catch((e) => setHealth({ error: e.message }));
  }, []);
  const f = FEATURES.find((x) => x.id === active);
  const dot = !health ? "bg-mute" : health.error ? "bg-bad" : "bg-good";
  const statusText = !health ? "Checking backend…" : health.error ? health.error : health.detail;
  const idx = FEATURES.indexOf(f) + 1;

  return (
    <div className="min-h-screen pb-28" style={{ "--tone": f.tone }}>
      <header className="border-b-2 border-ink bg-[#facc15]">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-8">
          <div className="flex items-center gap-3">
            <Mark />
            <h1 className="text-xl leading-none sm:text-2xl">Construction Intelligence</h1>
          </div>
          <div role="status" className="flex max-w-full items-center gap-2 rounded-full border-2 border-ink bg-white px-3 py-1 font-mono text-xs font-bold">
            <span aria-hidden="true" className={`h-2.5 w-2.5 shrink-0 rounded-full border border-ink ${dot}`} />
            <span className="truncate">{statusText}</span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 pt-8 sm:px-8 sm:pt-10">
        <section className="rounded-2xl border-2 border-ink bg-tone p-6 shadow-[6px_6px_0_0_#111] sm:p-8">
          <div className="flex flex-wrap items-center gap-4">
            <span className="grid h-14 w-14 place-items-center rounded-xl border-2 border-ink bg-white text-ink"><Icon id={f.id} cls="h-7 w-7" /></span>
            <span className="rounded-full border-2 border-ink bg-white px-3 py-1 font-mono text-xs font-bold uppercase tracking-wider">
              Tool {String(idx).padStart(2, "0")} / {String(FEATURES.length).padStart(2, "0")}
            </span>
          </div>
          <h2 className="mt-5 text-5xl leading-none sm:text-6xl">{f.name}</h2>
          <p className="mt-3 max-w-prose font-medium">{f.about}</p>
        </section>

        <div className="mt-8 space-y-5">
          <ErrorBoundary key={f.id}><f.C /></ErrorBoundary>
        </div>
      </main>

      <nav aria-label="Features" className="fixed bottom-4 left-1/2 z-30 w-max max-w-[95vw] -translate-x-1/2 overflow-x-auto rounded-2xl border-2 border-ink bg-white p-1.5 shadow-[4px_4px_0_0_#111]">
        <div className="flex gap-1.5">
          {FEATURES.map((x) => {
            const on = x.id === active;
            return (
              <button key={x.id} onClick={() => setActive(x.id)} aria-current={on ? "page" : undefined} aria-label={x.name} title={x.name}
                style={{ "--tone": x.tone }}
                className={`flex h-11 items-center gap-2 whitespace-nowrap rounded-xl border-2 px-3 font-bold transition ${
                  on ? "border-ink bg-tone text-ink" : "border-transparent text-mute hover:border-ink hover:bg-tone hover:text-ink"}`}>
                <Icon id={x.id} />
                {on && <span className="hidden text-sm sm:inline">{x.name}</span>}
              </button>
            );
          })}
        </div>
      </nav>
    </div>
  );
}