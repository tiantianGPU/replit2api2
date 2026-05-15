import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { randomBytes } from "node:crypto";
import app from "./app";
import { logger } from "./lib/logger";

// Resolve PROXY_API_KEY: prefer env (e.g. Replit Secret); fall back to the
// .proxy-key file written at build time by scripts/gen-proxy-key.mjs; final
// fallback generates a fresh key at runtime (covers `pnpm dev` paths).
function ensureProxyApiKey() {
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
