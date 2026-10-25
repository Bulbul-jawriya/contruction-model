import { useRef, useState } from "react";
import { api } from "../api";
import { Button, ErrorBox, Input, ResultMeta } from "../ui";

export default function Chat({ endpoint, placeholder, suggestions = [] }) {
  const [msgs, setMsgs] = useState([]);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const end = useRef(null);

  const send = async (q) => {
    const message = (q ?? text).trim();
    if (message.length < 2 || loading) return;
    const history = msgs.map(({ role, content }) => ({ role, content }));
    setMsgs((m) => [...m, { role: "user", content: message }]);
    setText(""); setError(null); setLoading(true);
    try {
      const r = await api(endpoint, { json: { message, history } });
      setMsgs((m) => [...m, { role: "assistant", content: r.data.answer, meta: r }]);
    } catch (e) { setError(e.message); }
    setLoading(false);
    setTimeout(() => end.current?.scrollIntoView({ block: "end" }), 0);
  };

  return (
    <div className="overflow-hidden rounded-2xl border-2 border-ink bg-paper shadow-[5px_5px_0_0_#111]">
      <div className="max-h-[28rem] min-h-48 space-y-3 overflow-y-auto p-4" aria-live="polite">
        {msgs.length === 0 && (
          <div className="space-y-2 text-mute">
            <p>Ask a question to start. Try:</p>
            <div className="flex flex-wrap gap-2">
              {suggestions.map((s) => (
                <button key={s} onClick={() => send(s)} className="rounded-full border-2 border-ink bg-white px-3.5 py-1.5 text-left text-sm font-medium text-ink transition hover:bg-tone">{s}</button>))}
            </div>
          </div>)}
        {msgs.map((m, i) => (
          <div key={i} className={m.role === "user" ? "ml-auto max-w-[85%] rounded-2xl rounded-br-sm border-2 border-ink bg-tone px-3.5 py-2 font-medium text-on-tone" : "max-w-[85%] rounded-2xl rounded-bl-sm border-2 border-ink bg-white px-3.5 py-2"}>
            {m.meta && <ResultMeta result={m.meta} />}
            <p className="whitespace-pre-wrap">{m.content}</p>
          </div>))}
        {loading && <p className="text-mute">Thinking…</p>}
        <div ref={end} />
      </div>
      <div className="space-y-2 border-t-2 border-ink p-3">
        <ErrorBox message={error} />
        <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); send(); }}>
          <Input value={text} onChange={(e) => setText(e.target.value)} placeholder={placeholder} aria-label="Your question" />
          <Button type="submit" loading={loading}>Send</Button>
        </form>
      </div>
    </div>
  );
}