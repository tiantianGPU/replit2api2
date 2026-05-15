import { useState, useEffect, useCallback } from "react";

const BG = "hsl(222, 47%, 11%)";
const CARD = "hsl(222, 42%, 15%)";
const CARD2 = "hsl(222, 38%, 18%)";
const BORDER = "hsl(222, 30%, 24%)";
const FG = "hsl(215, 28%, 90%)";
const MUTED = "hsl(215, 16%, 60%)";
const BLUE = "hsl(210, 100%, 66%)";
const PURPLE = "hsl(270, 80%, 70%)";
const GREEN = "hsl(145, 65%, 55%)";
const ORANGE = "hsl(28, 95%, 65%)";

const OPENAI_MODELS = ["gpt-5.2", "gpt-5-mini", "gpt-5-nano", "o4-mini", "o3"];
const ANTHROPIC_MODELS = [
  "claude-opus-4-7",
  "claude-opus-4-6",
  "claude-opus-4-6-thinking",
  "claude-sonnet-4-6",
  "claude-haiku-4-5",
];

const BASE_URL = window.location.origin;

// PROXY_API_KEY is generated at build time by scripts/gen-proxy-key.mjs and
// baked into the bundle here. The same key is loaded by api-server at boot.
const PROXY_API_KEY =
  (import.meta.env.VITE_PROXY_API_KEY as string | undefined) ?? "";

function CopyButton({ text, style }: { text: string; style?: React.CSSProperties }) {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(async () => {
    try {
      if (navigator.clipboard) {
        await navigator.clipboard.writeText(text);
      } else {
        const el = document.createElement("textarea");
        el.value = text;
        document.body.appendChild(el);
        el.select();
        document.execCommand("copy");
        document.body.removeChild(el);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {}
  }, [text]);

  return (
    <button
      onClick={copy}
      style={{
        background: copied ? "hsl(145,65%,18%)" : CARD2,
        border: `1px solid ${copied ? "hsl(145,65%,35%)" : BORDER}`,
        color: copied ? GREEN : MUTED,
        borderRadius: 6,
        padding: "4px 10px",
        fontSize: 12,
        cursor: "pointer",
        transition: "all 0.15s",
        whiteSpace: "nowrap",
        ...style,
      }}
    >
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}

function MethodBadge({ method }: { method: "GET" | "POST" }) {
  return (
    <span
      style={{
        background: method === "GET" ? "hsl(145,65%,16%)" : "hsl(270,50%,20%)",
        color: method === "GET" ? GREEN : PURPLE,
        border: `1px solid ${method === "GET" ? "hsl(145,65%,30%)" : "hsl(270,50%,40%)"}`,
        borderRadius: 4,
        padding: "2px 8px",
        fontSize: 11,
        fontWeight: 700,
        fontFamily: "monospace",
        letterSpacing: "0.05em",
        flexShrink: 0,
      }}
    >
      {method}
    </span>
  );
}

function TypeBadge({ type }: { type: "openai" | "anthropic" | "both" }) {
  const colors = {
    openai: { bg: "hsl(210,80%,16%)", fg: BLUE, border: "hsl(210,80%,30%)", label: "OpenAI" },
    anthropic: { bg: "hsl(28,80%,16%)", fg: ORANGE, border: "hsl(28,80%,30%)", label: "Anthropic" },
    both: { bg: "hsl(222,30%,20%)", fg: MUTED, border: BORDER, label: "Both" },
  };
  const c = colors[type];
  return (
    <span
      style={{
        background: c.bg,
        color: c.fg,
        border: `1px solid ${c.border}`,
        borderRadius: 4,
        padding: "2px 8px",
        fontSize: 11,
        fontWeight: 600,
        flexShrink: 0,
      }}
    >
      {c.label}
    </span>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h2
      style={{
        fontSize: 18,
        fontWeight: 700,
        color: FG,
        marginBottom: 16,
        paddingBottom: 10,
        borderBottom: `1px solid ${BORDER}`,
      }}
    >
      {children}
    </h2>
  );
}

function Card({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div
      style={{
        background: CARD,
        border: `1px solid ${BORDER}`,
        borderRadius: 10,
        padding: 20,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

function StatusDot({ online }: { online: boolean | null }) {
  return (
    <span style={{ position: "relative", display: "inline-flex", alignItems: "center" }}>
      {online && (
        <span
          style={{
            position: "absolute",
            width: 10,
            height: 10,
            borderRadius: "50%",
            background: GREEN,
            opacity: 0.4,
            animation: "pulse 2s infinite",
          }}
        />
      )}
      <span
        style={{
          width: 10,
          height: 10,
          borderRadius: "50%",
          background: online === null ? MUTED : online ? GREEN : "hsl(0,70%,55%)",
          display: "block",
          position: "relative",
        }}
      />
    </span>
  );
}

const ENDPOINTS = [
  {
    method: "GET" as const,
    path: "/v1/models",
    type: "both" as const,
    description: "List all available models",
  },
  {
    method: "POST" as const,
    path: "/v1/chat/completions",
    type: "openai" as const,
    description: "OpenAI-compatible chat completions — supports all OpenAI and Claude models",
  },
  {
    method: "POST" as const,
    path: "/v1/messages",
    type: "anthropic" as const,
    description: "Anthropic Messages API native format — supports all Claude and OpenAI models",
  },
];

const STEPS = [
  {
    title: "Open CherryStudio Settings",
    desc: "Go to Settings → AI Provider and click Add New Provider.",
  },
  {
    title: "Choose Provider Type",
    desc: "Select OpenAI (for /v1/chat/completions) or Anthropic (for /v1/messages). Both formats are fully supported.",
  },
  {
    title: "Set Base URL & API Key",
    desc: `Set the Base URL to ${BASE_URL} and copy the API key shown above into the API Key field.`,
  },
  {
    title: "Select a Model & Chat",
    desc: "Pick any listed model — gpt-5.2, claude-sonnet-4-6, claude-opus-4-7, etc. — and start chatting.",
  },
];

const CURL_EXAMPLE = `curl ${BASE_URL}/v1/chat/completions \\
  -H "Authorization: Bearer ${PROXY_API_KEY || "$PROXY_API_KEY"}" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "claude-opus-4-7",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'`;

export default function App() {
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch("/api/healthz");
        setOnline(r.ok);
      } catch {
        setOnline(false);
      }
    };
    check();
    const t = setInterval(check, 15000);
    return () => clearInterval(t);
  }, []);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: BG,
        color: FG,
        fontFamily: "'Inter', system-ui, sans-serif",
        lineHeight: 1.6,
      }}
    >
      <style>{`
        @keyframes pulse {
          0%, 100% { transform: scale(1); opacity: 0.4; }
          50% { transform: scale(1.8); opacity: 0; }
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${BORDER}; border-radius: 3px; }
      `}</style>

      {/* Header */}
      <header
        style={{
          background: CARD,
          borderBottom: `1px solid ${BORDER}`,
          padding: "0 32px",
          height: 60,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          position: "sticky",
          top: 0,
          zIndex: 10,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 8,
              background: "linear-gradient(135deg, hsl(210,100%,50%), hsl(270,80%,60%))",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 16,
            }}
          >
            ⚡
          </div>
          <span style={{ fontWeight: 700, fontSize: 17, color: FG }}>AI Proxy API</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <StatusDot online={online} />
          <span style={{ fontSize: 13, color: online === null ? MUTED : online ? GREEN : "hsl(0,70%,55%)" }}>
            {online === null ? "Checking..." : online ? "Online" : "Offline"}
          </span>
        </div>
      </header>

      <main style={{ maxWidth: 860, margin: "0 auto", padding: "32px 24px", display: "flex", flexDirection: "column", gap: 32 }}>

        {/* Connection Details */}
        <section>
          <SectionTitle>Connection Details</SectionTitle>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <Card>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
                <div>
                  <div style={{ fontSize: 12, color: MUTED, marginBottom: 4 }}>Base URL</div>
                  <code style={{ color: BLUE, fontSize: 14 }}>{BASE_URL}</code>
                </div>
                <CopyButton text={BASE_URL} />
              </div>
            </Card>
            <Card>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div style={{ fontSize: 12, color: MUTED, marginBottom: 4 }}>
                    API Key (PROXY_API_KEY)
                  </div>
                  <code
                    style={{
                      color: PROXY_API_KEY ? ORANGE : MUTED,
                      fontSize: 13,
                      fontFamily: "Menlo, Monaco, 'Courier New', monospace",
                      wordBreak: "break-all",
                      display: "block",
                    }}
                  >
                    {PROXY_API_KEY || "(not generated — set PROXY_API_KEY secret or rebuild)"}
                  </code>
                </div>
                <CopyButton text={PROXY_API_KEY} />
              </div>
            </Card>
            <Card>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
                <div>
                  <div style={{ fontSize: 12, color: MUTED, marginBottom: 4 }}>Authorization Header</div>
                  <code style={{ color: BLUE, fontSize: 14 }}>
                    Authorization: Bearer {PROXY_API_KEY || "<your PROXY_API_KEY>"}
                  </code>
                </div>
                <CopyButton
                  text={`Authorization: Bearer ${PROXY_API_KEY || "<your PROXY_API_KEY>"}`}
                />
              </div>
            </Card>
          </div>
        </section>

        {/* Endpoints */}
        <section>
          <SectionTitle>API Endpoints</SectionTitle>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {ENDPOINTS.map((ep) => (
              <Card key={ep.path}>
                <div style={{ display: "flex", alignItems: "flex-start", gap: 12, flexWrap: "wrap" }}>
                  <MethodBadge method={ep.method} />
                  <div style={{ flex: 1, minWidth: 200 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                      <code style={{ color: FG, fontSize: 14, fontWeight: 600 }}>{ep.path}</code>
                      <TypeBadge type={ep.type} />
                    </div>
                    <div style={{ fontSize: 13, color: MUTED }}>{ep.description}</div>
                  </div>
                  <CopyButton text={`${BASE_URL}${ep.path}`} />
                </div>
              </Card>
            ))}
          </div>
        </section>

        {/* Models */}
        <section>
          <SectionTitle>Available Models</SectionTitle>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 10 }}>
            {OPENAI_MODELS.map((m) => (
              <Card key={m} style={{ padding: "12px 14px" }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: FG, marginBottom: 6 }}>{m}</div>
                <TypeBadge type="openai" />
              </Card>
            ))}
            {ANTHROPIC_MODELS.map((m) => (
              <Card key={m} style={{ padding: "12px 14px" }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: FG, marginBottom: 6 }}>{m}</div>
                <TypeBadge type="anthropic" />
              </Card>
            ))}
          </div>
        </section>

        {/* CherryStudio Setup */}
        <section>
          <SectionTitle>CherryStudio Setup Guide</SectionTitle>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {STEPS.map((step, i) => (
              <Card key={i} style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
                <div
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: "50%",
                    background: "linear-gradient(135deg, hsl(210,100%,50%), hsl(270,80%,60%))",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontWeight: 800,
                    fontSize: 14,
                    color: "#fff",
                    flexShrink: 0,
                    marginTop: 2,
                  }}
                >
                  {i + 1}
                </div>
                <div>
                  <div style={{ fontWeight: 700, fontSize: 15, color: FG, marginBottom: 4 }}>{step.title}</div>
                  <div style={{ fontSize: 13, color: MUTED }}>{step.desc}</div>
                </div>
              </Card>
            ))}
          </div>
        </section>

        {/* Quick Test */}
        <section>
          <SectionTitle>Quick Test (curl)</SectionTitle>
          <Card style={{ padding: 0 }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "12px 16px",
                borderBottom: `1px solid ${BORDER}`,
              }}
            >
              <span style={{ fontSize: 12, color: MUTED }}>shell</span>
              <CopyButton text={CURL_EXAMPLE} />
            </div>
            <pre
              style={{
                padding: 16,
                overflowX: "auto",
                fontSize: 13,
                lineHeight: 1.7,
                fontFamily: "Menlo, Monaco, 'Courier New', monospace",
              }}
            >
              {CURL_EXAMPLE.split("\n").map((line, i) => {
                if (line.includes("curl ")) return <div key={i}><span style={{ color: GREEN }}>curl</span><span style={{ color: BLUE }}> {line.replace("curl ", "")}</span></div>;
                if (line.includes("-H ")) return <div key={i}><span style={{ color: PURPLE }}>  -H </span><span style={{ color: ORANGE }}>{line.replace(/\s*-H\s*/, "")}</span></div>;
                if (line.includes("-d ")) return <div key={i}><span style={{ color: PURPLE }}>  -d </span><span style={{ color: FG }}>{line.replace(/\s*-d\s*/, "")}</span></div>;
                if (line.trim().startsWith('"model"')) return <div key={i}><span style={{ color: FG }}>    {line.trim()}</span></div>;
                if (line.trim().startsWith('"messages"') || line.trim().startsWith('"role"') || line.trim().startsWith('"content"')) return <div key={i}><span style={{ color: MUTED }}>    {line.trim()}</span></div>;
                return <div key={i}><span style={{ color: FG }}>{line}</span></div>;
              })}
            </pre>
          </Card>
        </section>

        {/* Footer */}
        <footer
          style={{
            textAlign: "center",
            padding: "16px 0",
            borderTop: `1px solid ${BORDER}`,
            fontSize: 12,
            color: MUTED,
          }}
        >
          Powered by <span style={{ color: BLUE }}>OpenAI</span> &amp; <span style={{ color: ORANGE }}>Anthropic</span> via Replit AI Integrations · Express · TypeScript
        </footer>
      </main>
    </div>
  );
}
