import express, { type Express } from "express";
import cors from "cors";
import pinoHttp from "pino-http";
import { existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import router from "./routes";
import proxyRouter from "./routes/proxy";
import { logger } from "./lib/logger";

const app: Express = express();

app.use(
  pinoHttp({
    logger,
    serializers: {
      req(req) {
        return {
          id: req.id,
          method: req.method,
          url: req.url?.split("?")[0],
        };
      },
      res(res) {
        return {
          statusCode: res.statusCode,
        };
      },
    },
  }),
);
app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

app.use("/api", router);
// Also mount /healthz at root so generic uptime checks (e.g. Replit's
// platform health probe, the replit2api-manage harvester) hit the JSON
// response instead of falling through to the SPA fallback below.
app.use(router);
app.use("/v1", proxyRouter);

// Serve the api-portal SPA static bundle, if present. The build pipeline
// emits it to <repo>/artifacts/api-portal/dist/public via Vite. We try a few
// relative locations so this works from `dist/index.mjs` (bundled output)
// as well as `tsx src/index.ts` (dev).
const moduleFile = fileURLToPath(import.meta.url);
const moduleDir = dirname(moduleFile);
const PORTAL_CANDIDATES = [
  resolve(moduleDir, "..", "..", "api-portal", "dist", "public"),
  resolve(moduleDir, "..", "api-portal", "dist", "public"),
  resolve(process.cwd(), "artifacts", "api-portal", "dist", "public"),
  resolve(process.cwd(), "..", "api-portal", "dist", "public"),
];
let portalDir: string | null = null;
for (const cand of PORTAL_CANDIDATES) {
  if (existsSync(cand)) {
    portalDir = cand;
    break;
  }
}

if (portalDir) {
  const resolvedPortalDir = portalDir;
  logger.info({ portalDir: resolvedPortalDir }, "Mounting api-portal SPA at /");
  app.use(express.static(resolvedPortalDir, { maxAge: "1h", index: "index.html" }));
  // SPA fallback: any non-API GET that didn't hit a static file returns the
  // portal's index.html so client-side routing works on hard refresh. The
  // explicit `/healthz` exclusion keeps that route serving the JSON response
  // produced by the health router below.
  app.use((req, res, next) => {
    if (req.method !== "GET" && req.method !== "HEAD") return next();
    if (
      req.path.startsWith("/api/") ||
      req.path.startsWith("/v1/") ||
      req.path === "/healthz" ||
      req.path === "/readyz" ||
      req.path === "/metrics"
    ) {
      return next();
    }
    if (res.headersSent) return next();
    res.sendFile(resolve(resolvedPortalDir, "index.html"));
  });
} else {
  logger.warn({ tried: PORTAL_CANDIDATES },
              "api-portal/dist/public not found; / will 404 (run `pnpm run build` first)");
}

export default app;
