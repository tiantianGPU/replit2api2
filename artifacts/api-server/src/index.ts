import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { randomBytes } from "node:crypto";
import app from "./app";
import { logger } from "./lib/logger";

// Resolve PROXY_API_KEY with this priority:
//   1. artifacts/api-portal/.env.production -> VITE_PROXY_API_KEY
//      This is the value Vite bakes into the frontend bundle at build time
//      and what users copy/paste into their OpenAI clients. Reading it here
//      guarantees the backend bearer-check agrees with the displayed key.
//   2. process.env.PROXY_API_KEY (Replit Secret) — explicit operator override
//   3. .proxy-key file (cached from previous gen-proxy-key run)
//   4. fresh randomBytes — last-resort for `pnpm dev` paths before build runs
//
// Why VITE_PROXY_API_KEY beats env: a stale Replit Secret left over from a
// fork template or an unrelated AI Integrations setup can silently desync the
// backend from the bundle, producing 401s on every chat. The bundle is the
// source-of-truth users actually see.
function readVitePortalKey(): { key: string; source: string } | null {
  let dir = process.cwd();
  for (let i = 0; i < 5; i++) {
    const f = resolve(dir, "artifacts/api-portal/.env.production");
    if (existsSync(f)) {
      try {
        const text = readFileSync(f, "utf8");
        const m = text.match(/^\s*VITE_PROXY_API_KEY\s*=\s*([A-Za-z0-9._-]+)\s*$/m);
        if (m && m[1]) {
          return { key: m[1].trim(), source: f };
        }
      } catch (err) {
        logger.warn({ err, file: f }, "failed to read .env.production");
      }
    }
    const parent = resolve(dir, "..");
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

function ensureProxyApiKey() {
  const vite = readVitePortalKey();
  if (vite) {
    process.env.PROXY_API_KEY = vite.key;
    logger.info({
      keyPreview: `${vite.key.slice(0, 8)}...${vite.key.slice(-4)}`,
      source: vite.source,
    }, "PROXY_API_KEY loaded from api-portal/.env.production (matches frontend bundle)");
    return;
  }
  const envKey = (process.env.PROXY_API_KEY || "").trim();
  if (envKey) {
    process.env.PROXY_API_KEY = envKey;
    logger.info({ keyPreview: `${envKey.slice(0, 8)}...${envKey.slice(-4)}` },
                "PROXY_API_KEY loaded from environment");
    return;
  }
  let dir = process.cwd();
  for (let i = 0; i < 5; i++) {
    const f = resolve(dir, ".proxy-key");
    if (existsSync(f)) {
      try {
        const k = readFileSync(f, "utf8").trim();
        if (k) {
          process.env.PROXY_API_KEY = k;
          logger.info({
            keyPreview: `${k.slice(0, 8)}...${k.slice(-4)}`,
            source: f,
          }, "PROXY_API_KEY loaded from cached .proxy-key");
          return;
        }
      } catch (err) {
        logger.warn({ err, file: f }, "failed to read .proxy-key");
      }
    }
    const parent = resolve(dir, "..");
    if (parent === dir) break;
    dir = parent;
  }
  const fresh = randomBytes(32).toString("hex");
  process.env.PROXY_API_KEY = fresh;
  try {
    writeFileSync(resolve(process.cwd(), ".proxy-key"), fresh + "\n", "utf8");
  } catch (err) {
    logger.warn({ err }, "failed to persist runtime-generated .proxy-key");
  }
  logger.warn({ keyPreview: `${fresh.slice(0, 8)}...${fresh.slice(-4)}` },
              "PROXY_API_KEY not in env or .proxy-key; generated fresh at runtime");
}

ensureProxyApiKey();

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error(
    "PORT environment variable is required but was not provided.",
  );
}

const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

app.listen(port, (err) => {
  if (err) {
    logger.error({ err }, "Error listening on port");
    process.exit(1);
  }

  logger.info({ port }, "Server listening");
});
