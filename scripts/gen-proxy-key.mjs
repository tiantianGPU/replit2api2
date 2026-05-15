#!/usr/bin/env node
/**
 * Generate (or reuse) a single PROXY_API_KEY that is shared between
 * the api-server runtime and the api-portal Vite build.
 *
 * Resolution order:
 *   1. process.env.PROXY_API_KEY (e.g. Replit Secret) — used verbatim
 *   2. <repo>/.proxy-key — reused from a previous build
 *   3. crypto.randomBytes(32).toString("hex") — fresh
 *
 * Side effects:
 *   - <repo>/.proxy-key gets the chosen key (so api-server boots can find it)
 *   - <repo>/artifacts/api-portal/.env.production gets VITE_PROXY_API_KEY=<key>
 *     so Vite bakes it into the bundle at build time
 *
 * Run this BEFORE `pnpm -r run build`.
 */
import { randomBytes } from "node:crypto";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const KEY_FILE = resolve(ROOT, ".proxy-key");
const PORTAL_ENV = resolve(ROOT, "artifacts/api-portal/.env.production");

let key = "";
let source = "";

const envKey = (process.env.PROXY_API_KEY || "").trim();
if (envKey) {
  key = envKey;
  source = "env PROXY_API_KEY";
} else if (existsSync(KEY_FILE)) {
  const cached = readFileSync(KEY_FILE, "utf8").trim();
  if (cached) {
    key = cached;
    source = "cached .proxy-key";
  }
}

if (!key) {
  key = randomBytes(32).toString("hex");
  source = "fresh randomBytes(32)";
}

writeFileSync(KEY_FILE, key + "\n", { encoding: "utf8", mode: 0o600 });
mkdirSync(dirname(PORTAL_ENV), { recursive: true });
writeFileSync(PORTAL_ENV, `VITE_PROXY_API_KEY=${key}\n`, "utf8");

const preview = `${key.slice(0, 8)}...${key.slice(-4)}`;
console.log(`[gen-proxy-key] source=${source} key=${preview} (len=${key.length})`);
console.log(`[gen-proxy-key] wrote ${KEY_FILE}`);
console.log(`[gen-proxy-key] wrote ${PORTAL_ENV}`);
