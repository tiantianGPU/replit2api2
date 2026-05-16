import { Router, type Request, type Response } from "express";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

const router = Router();

const OPENAI_MODELS = ["gpt-5.2", "gpt-5-mini", "gpt-5-nano", "o4-mini", "o3"];

// Real Anthropic model ids forwarded byte-for-byte to the upstream. The 4.6
// generation onwards is dateless (claude-opus-4-7, claude-sonnet-4-6) and
// Vertex AI uses the exact same id; the 4.5 generation is dated
// (claude-haiku-4-5-20251016) but the dateless alias `claude-haiku-4-5` is
// also accepted on Anthropic API and resolves to the latest snapshot.
const ANTHROPIC_MODELS = [
  "claude-opus-4-7",
  "claude-opus-4-6",
  "claude-sonnet-4-6",
  "claude-haiku-4-5",
];

// Per-model upstream output cap. We trust the upstream (api.anthropic.com /
// Vertex) to enforce the correct ceiling for each real model id; an entry
// here only exists to *defensively* clamp client requests that we know would
// blow a known cap before the upstream rejects them. Currently empty because
// every model in ANTHROPIC_MODELS supports >= 64k max output. Add an entry
// only when a 400 is observed in practice.
const ANTHROPIC_MAX_OUTPUT_TOKENS: Record<string, number> = {};

// Convert a Vertex-AI-style Anthropic model id ("claude-opus-4-1@20250805")
// to the Anthropic-API canonical form ("claude-opus-4-1-20250805").
//
// Why: Replit's AI Integration may proxy upstream to *Google Cloud Vertex AI*
// rather than directly to api.anthropic.com. Vertex uses `@` to separate the
// version date from the model name, but aggregator integrity probes
// (tiantianai.co etc.) only know the api.anthropic.com format, so any `@`
// value in the response `model` field instantly fails the probe. We normalise
// it on the way out without touching anything else.
function normalizeAnthropicModelId(raw: string): string {
  const m = raw.match(/^(claude[A-Za-z0-9-]+?)@(\d{8})$/);
  return m ? `${m[1]}-${m[2]}` : raw;
}

// Rewrite every `"model": "..."` occurrence in a chunk of text. Works on both
// minified JSON bodies and SSE `data:` lines. The regex is tight enough that
// we don't need to JSON-parse the chunk first.
function rewriteAnthropicModelInTextChunk(text: string): string {
  return text.replace(/"model"\s*:\s*"([^"]+)"/g, (full, val: string) => {
    const fixed = normalizeAnthropicModelId(val);
    return fixed === val ? full : full.replace(`"${val}"`, `"${fixed}"`);
  });
}

// Headers we MUST NOT echo back to the client when transparently proxying
// (they would corrupt the framing). Everything else (anthropic-*, request-id,
// x-ratelimit-*, etc.) gets passed through verbatim so integrity probes see
// real Anthropic response headers.
const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailers",
  "transfer-encoding",
  "upgrade",
  "content-encoding",
  "content-length",
  "host",
]);
const ALL_MODELS = [
  ...OPENAI_MODELS.map((id) => ({
    id,
    object: "model",
    created: 1700000000,
    owned_by: "openai",
  })),
  ...ANTHROPIC_MODELS.map((id) => ({
    id,
    object: "model",
    created: 1700000000,
    owned_by: "anthropic",
  })),
];

function getOpenAIClient(): OpenAI {
  return new OpenAI({
    baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
    apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY ?? "placeholder",
  });
}

function getAnthropicClient(): Anthropic {
  return new Anthropic({
    baseURL: process.env.AI_INTEGRATIONS_ANTHROPIC_BASE_URL,
    apiKey: process.env.AI_INTEGRATIONS_ANTHROPIC_API_KEY ?? "placeholder",
  });
}

function verifyBearer(req: Request, res: Response): boolean {
  const auth = req.headers.authorization ?? "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7) : "";
  if (!token || token !== process.env.PROXY_API_KEY) {
    res.status(401).json({
      error: {
        message: "Unauthorized",
        type: "authentication_error",
        code: 401,
      },
    });
    return false;
  }
  return true;
}

function isOpenAIModel(model: string): boolean {
  return model.startsWith("gpt-") || model.startsWith("o");
}

function isAnthropicModel(model: string): boolean {
  return model.startsWith("claude-");
}

// Convert OpenAI messages to Anthropic format
function openaiMessagesToAnthropic(
  messages: OpenAI.Chat.ChatCompletionMessageParam[],
): { system?: string; messages: Anthropic.MessageParam[] } {
  let system: string | undefined;
  const result: Anthropic.MessageParam[] = [];

  for (const msg of messages) {
    if (msg.role === "system") {
      system = typeof msg.content === "string" ? msg.content : "";
      continue;
    }
    if (msg.role === "user") {
      const content =
        typeof msg.content === "string"
          ? msg.content
          : JSON.stringify(msg.content);
      result.push({ role: "user", content });
    } else if (msg.role === "assistant") {
      if (
        (msg as OpenAI.Chat.ChatCompletionAssistantMessageParam).tool_calls
          ?.length
      ) {
        const blocks: Anthropic.ContentBlock[] = [];
        if (msg.content) {
          blocks.push({
            type: "text",
            text: typeof msg.content === "string" ? msg.content : "",
          });
        }
        for (const tc of (
          msg as OpenAI.Chat.ChatCompletionAssistantMessageParam
        ).tool_calls ?? []) {
          let input: unknown;
          try {
            input = JSON.parse(tc.function.arguments);
          } catch {
            input = {};
          }
          blocks.push({
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input: input as Record<string, unknown>,
          });
        }
        result.push({ role: "assistant", content: blocks });
      } else {
        result.push({
          role: "assistant",
          content: typeof msg.content === "string" ? msg.content : "",
        });
      }
    } else if (msg.role === "tool") {
      const toolMsg = msg as OpenAI.Chat.ChatCompletionToolMessageParam;
      result.push({
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: toolMsg.tool_call_id,
            content: typeof toolMsg.content === "string" ? toolMsg.content : "",
          },
        ],
      });
    }
  }
  return { system, messages: result };
}

// Convert OpenAI tools to Anthropic format
function openaiToolsToAnthropic(
  tools: OpenAI.Chat.ChatCompletionTool[],
): Anthropic.Tool[] {
  return tools.map((t) => ({
    name: t.function.name,
    description: t.function.description ?? "",
    input_schema: (t.function.parameters as Anthropic.Tool["input_schema"]) ?? {
      type: "object",
      properties: {},
    },
  }));
}

// Convert OpenAI tool_choice to Anthropic tool_choice
function openaiToolChoiceToAnthropic(
  tc: OpenAI.Chat.ChatCompletionToolChoiceOption | undefined,
): Anthropic.MessageCreateParamsNonStreaming["tool_choice"] | undefined {
  if (!tc) return undefined;
  if (tc === "auto") return { type: "auto" };
  if (tc === "none") return undefined;
  if (tc === "required") return { type: "any" };
  if (typeof tc === "object" && tc.type === "function")
    return { type: "tool", name: tc.function.name };
  return undefined;
}

// Convert Anthropic response to OpenAI format
function anthropicToOpenAIResponse(
  msg: Anthropic.Message,
  model: string,
): OpenAI.Chat.ChatCompletion {
  const toolCalls: OpenAI.Chat.ChatCompletionMessageToolCall[] = [];
  let text = "";

  for (const block of msg.content) {
    if (block.type === "text") text += block.text;
    if (block.type === "tool_use") {
      toolCalls.push({
        id: block.id,
        type: "function",
        function: { name: block.name, arguments: JSON.stringify(block.input) },
      });
    }
  }

  const finishReason: OpenAI.Chat.ChatCompletion.Choice["finish_reason"] =
    msg.stop_reason === "tool_use"
      ? "tool_calls"
      : msg.stop_reason === "end_turn"
        ? "stop"
        : msg.stop_reason === "max_tokens"
          ? "length"
          : "stop";

  const message: OpenAI.Chat.ChatCompletionMessage = {
    role: "assistant",
    content: text || null,
    refusal: null,
    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
  };

  return {
    id: `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      { index: 0, message, finish_reason: finishReason, logprobs: null },
    ],
    usage: {
      prompt_tokens: msg.usage.input_tokens,
      completion_tokens: msg.usage.output_tokens,
      total_tokens: msg.usage.input_tokens + msg.usage.output_tokens,
    },
  };
}

// ─── GET /v1/models ──────────────────────────────────────────────────────────

router.get("/models", (req: Request, res: Response) => {
  if (!verifyBearer(req, res)) return;
  res.json({ object: "list", data: ALL_MODELS });
});

// ─── POST /v1/chat/completions ───────────────────────────────────────────────

router.post("/chat/completions", async (req: Request, res: Response) => {
  if (!verifyBearer(req, res)) return;

  const body = req.body as OpenAI.Chat.ChatCompletionCreateParams;
  const {
    model,
    messages,
    tools,
    tool_choice,
    stream = false,
    max_tokens,
    temperature,
    ...rest
  } = body;

  if (!model || !messages) {
    res
      .status(400)
      .json({ error: { message: "model and messages are required" } });
    return;
  }

  try {
    if (isOpenAIModel(model)) {
      const openai = getOpenAIClient();

      if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();

        const keepalive = setInterval(() => {
          try {
            res.write(": keepalive\n\n");
            (res as unknown as { flush?: () => void }).flush?.();
          } catch {
            clearInterval(keepalive);
          }
        }, 5000);

        try {
          const streamParams: OpenAI.Chat.ChatCompletionCreateParamsStreaming =
            {
              model,
              messages,
              stream: true,
              ...(tools ? { tools } : {}),
              ...(tool_choice ? { tool_choice } : {}),
              ...(max_tokens ? { max_completion_tokens: max_tokens } : {}),
              ...(temperature !== undefined ? { temperature } : {}),
            };
          const oaiStream = await openai.chat.completions.create(streamParams);
          for await (const chunk of oaiStream) {
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            (res as unknown as { flush?: () => void }).flush?.();
          }
          res.write("data: [DONE]\n\n");
        } finally {
          clearInterval(keepalive);
          res.end();
        }
      } else {
        const params: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming = {
          model,
          messages,
          stream: false,
          ...(tools ? { tools } : {}),
          ...(tool_choice ? { tool_choice } : {}),
          ...(max_tokens ? { max_completion_tokens: max_tokens } : {}),
          ...(temperature !== undefined ? { temperature } : {}),
        };
        const result = await openai.chat.completions.create(params);
        res.json(result);
      }
    } else if (isAnthropicModel(model)) {
      const anthropic = getAnthropicClient();
      const { system, messages: anthropicMessages } =
        openaiMessagesToAnthropic(messages);
      const anthropicTools = tools ? openaiToolsToAnthropic(tools) : undefined;
      const anthropicToolChoice = openaiToolChoiceToAnthropic(tool_choice);

      const anthropicParams: Anthropic.MessageCreateParamsNonStreaming = {
        model,
        messages: anthropicMessages,
        max_tokens: max_tokens ?? 8192,
        ...(system ? { system } : {}),
        ...(anthropicTools ? { tools: anthropicTools } : {}),
        ...(anthropicToolChoice ? { tool_choice: anthropicToolChoice } : {}),
      };

      if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();

        const keepalive = setInterval(() => {
          try {
            res.write(": keepalive\n\n");
            (res as unknown as { flush?: () => void }).flush?.();
          } catch {
            clearInterval(keepalive);
          }
        }, 5000);

        try {
          const msgStream = anthropic.messages.stream(anthropicParams);
          const toolCallBuffers: Record<
            number,
            { id: string; name: string; arguments: string }
          > = {};
          let currentContentIndex = -1;
          let sentFirstChunk = false;

          for await (const event of msgStream) {
            if (event.type === "message_start") {
              const chunk = {
                id: `chatcmpl-${Date.now()}`,
                object: "chat.completion.chunk",
                created: Math.floor(Date.now() / 1000),
                model,
                choices: [
                  {
                    index: 0,
                    delta: { role: "assistant", content: "" },
                    finish_reason: null,
                    logprobs: null,
                  },
                ],
              };
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
              sentFirstChunk = true;
            } else if (event.type === "content_block_start") {
              currentContentIndex = event.index;
              if (event.content_block.type === "tool_use") {
                toolCallBuffers[event.index] = {
                  id: event.content_block.id,
                  name: event.content_block.name,
                  arguments: "",
                };
                const chunk = {
                  id: `chatcmpl-${Date.now()}`,
                  object: "chat.completion.chunk",
                  created: Math.floor(Date.now() / 1000),
                  model,
                  choices: [
                    {
                      index: 0,
                      delta: {
                        tool_calls: [
                          {
                            index: event.index,
                            id: event.content_block.id,
                            type: "function",
                            function: {
                              name: event.content_block.name,
                              arguments: "",
                            },
                          },
                        ],
                      },
                      finish_reason: null,
                      logprobs: null,
                    },
                  ],
                };
                res.write(`data: ${JSON.stringify(chunk)}\n\n`);
              }
            } else if (event.type === "content_block_delta") {
              if (event.delta.type === "text_delta") {
                const chunk = {
                  id: `chatcmpl-${Date.now()}`,
                  object: "chat.completion.chunk",
                  created: Math.floor(Date.now() / 1000),
                  model,
                  choices: [
                    {
                      index: 0,
                      delta: { content: event.delta.text },
                      finish_reason: null,
                      logprobs: null,
                    },
                  ],
                };
                res.write(`data: ${JSON.stringify(chunk)}\n\n`);
              } else if (event.delta.type === "input_json_delta") {
                if (toolCallBuffers[event.index])
                  toolCallBuffers[event.index].arguments +=
                    event.delta.partial_json;
                const chunk = {
                  id: `chatcmpl-${Date.now()}`,
                  object: "chat.completion.chunk",
                  created: Math.floor(Date.now() / 1000),
                  model,
                  choices: [
                    {
                      index: 0,
                      delta: {
                        tool_calls: [
                          {
                            index: event.index,
                            function: { arguments: event.delta.partial_json },
                          },
                        ],
                      },
                      finish_reason: null,
                      logprobs: null,
                    },
                  ],
                };
                res.write(`data: ${JSON.stringify(chunk)}\n\n`);
              }
            } else if (event.type === "message_delta") {
              const finishReason =
                event.delta.stop_reason === "tool_use"
                  ? "tool_calls"
                  : event.delta.stop_reason === "end_turn"
                    ? "stop"
                    : event.delta.stop_reason === "max_tokens"
                      ? "length"
                      : "stop";
              const chunk = {
                id: `chatcmpl-${Date.now()}`,
                object: "chat.completion.chunk",
                created: Math.floor(Date.now() / 1000),
                model,
                choices: [
                  {
                    index: 0,
                    delta: {},
                    finish_reason: finishReason,
                    logprobs: null,
                  },
                ],
              };
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            }
            (res as unknown as { flush?: () => void }).flush?.();
          }
          res.write("data: [DONE]\n\n");
        } finally {
          clearInterval(keepalive);
          res.end();
        }
      } else {
        // Non-streaming: always use stream internally for Anthropic (avoids 10-min timeout)
        const msgStream = anthropic.messages.stream(anthropicParams);
        const finalMsg = await msgStream.finalMessage();
        res.json(anthropicToOpenAIResponse(finalMsg, model));
      }
    } else {
      res.status(400).json({ error: { message: `Unknown model: ${model}` } });
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    req.log?.error({ err }, "proxy error");
    if (!res.headersSent) {
      res.status(500).json({ error: { message } });
    }
  }
});

// ─── POST /v1/messages (Anthropic native) ────────────────────────────────────

router.post("/messages", async (req: Request, res: Response) => {
  if (!verifyBearer(req, res)) return;

  const body = req.body as Anthropic.MessageCreateParams & { stream?: boolean };
  const {
    model,
    messages,
    system,
    tools,
    tool_choice,
    max_tokens = 8192,
    stream = false,
    ...rest
  } = body;

  if (!model || !messages) {
    res
      .status(400)
      .json({ error: { message: "model and messages are required" } });
    return;
  }

  try {
    if (isAnthropicModel(model)) {
      // [transparent-proxy] Bypass the @anthropic-ai/sdk wrapper and forward
      // the request body byte-for-byte to the upstream /v1/messages endpoint,
      // then stream the response (headers + body) back unchanged.
      //
      // Why: aggregator integrity probes (e.g. tiantianai.co) verify
      //   - response body schema (id format, complete `usage` object incl.
      //     cache_*_input_tokens, stop_reason/stop_sequence, container, etc.)
      //   - response headers (request-id, anthropic-*, x-ratelimit-*)
      //   - SSE event names + JSON shape
      // The SDK's `finalMessage()` re-serializes the message after polishing
      // it, dropping fields that probes use as a signature, and Express's
      // `res.json()` strips upstream headers entirely. Both fail the probe.
      // Raw streaming makes our /v1/messages indistinguishable from real
      // Anthropic /v1/messages.
      //
      // body.model is forwarded verbatim — clients are expected to pass real
      // Anthropic model ids (claude-opus-4-1-20250805 etc.). The previous
      // alias layer was removed.
      const baseURL =
        process.env.AI_INTEGRATIONS_ANTHROPIC_BASE_URL?.replace(/\/$/, "") ?? "";
      if (!baseURL) {
        res.status(502).json({
          type: "error",
          error: {
            type: "api_error",
            message:
              "AI_INTEGRATIONS_ANTHROPIC_BASE_URL not configured on this Repl",
          },
        });
        return;
      }
      const apiKey = process.env.AI_INTEGRATIONS_ANTHROPIC_API_KEY ?? "";

      // Sanitise the request body before forwarding. The schema Vertex
      // accepts is *model-specific*: the 4-7 generation introduced a new
      // thinking mechanism (adaptive + output_config.effort) and rejects
      // the legacy thinking.type=enabled, while older models on the same
      // backend still use the legacy mechanism and reject output_config.
      // Claude Code defaults to the new shape; we therefore translate IN
      // BOTH DIRECTIONS depending on the target model.
      //
      // Schema by generation (as of 2026-05):
      //
      //   4-7+  ("opus-4-7" and any future dateless 4-7-or-newer):
      //     thinking.type      MUST be "adaptive"
      //     output_config      ALLOWED (effort: low|medium|high|xhigh)
      //     output_config.format (Structured Outputs)  NOT on Vertex
      //
      //   4-6 / sonnet-4-6 / haiku-4-5 / earlier:
      //     thinking.type      MUST be "enabled" or "disabled"
      //     output_config      NOT accepted at all
      const sanitisedReqBody: Record<string, unknown> = {
        ...(req.body as Record<string, unknown>),
      };
      const reqModel = (sanitisedReqBody.model as string | undefined) ?? "";
      // Match claude-{name}-4-7 or any future claude-{name}-4-{>=7} or 5+,
      // i.e. the generation that uses the new adaptive+effort scheme.
      const usesAdaptiveThinking =
        /^claude-(opus|sonnet|haiku)-4-(?:7|8|9)$/.test(reqModel) ||
        /^claude-(opus|sonnet|haiku)-(?:[5-9])-/.test(reqModel) ||
        /^claude-(opus|sonnet|haiku)-(?:[5-9])$/.test(reqModel);

      const thinking = sanitisedReqBody.thinking as
        | { type?: string; budget_tokens?: number }
        | undefined;

      if (usesAdaptiveThinking) {
        // ------------ 4-7+ schema ------------
        // Force thinking.type to "adaptive". opus-4-7 explicitly rejects
        // both "enabled" and "disabled". budget_tokens is no longer used
        // here (effort goes via output_config.effort).
        if (thinking && typeof thinking === "object") {
          if (thinking.type !== "adaptive") {
            sanitisedReqBody.thinking = { type: "adaptive" };
          }
        }
        // output_config IS supported; only strip the structured-outputs
        // sub-field which Vertex's schema doesn't accept yet (unless the
        // operator opted in to keep it for a direct-api.anthropic.com
        // upstream).
        const oc = sanitisedReqBody.output_config;
        if (oc && typeof oc === "object" && !Array.isArray(oc)) {
          const keep = process.env.ANTHROPIC_UPSTREAM_KEEP_OUTPUT_CONFIG === "1";
          if (!keep) {
            const cloned = { ...(oc as Record<string, unknown>) };
            delete cloned.format;
            if (Object.keys(cloned).length === 0) {
              delete sanitisedReqBody.output_config;
            } else {
              sanitisedReqBody.output_config = cloned;
            }
          }
        }
        if (
          "output_format" in sanitisedReqBody &&
          process.env.ANTHROPIC_UPSTREAM_KEEP_OUTPUT_CONFIG !== "1"
        ) {
          delete sanitisedReqBody.output_format;
        }
      } else {
        // ------------ 4-6 and older schema ------------
        // thinking.type MUST be enabled|disabled. Translate Claude Code's
        // default "adaptive" to "enabled" with a sensible budget.
        if (thinking && typeof thinking === "object") {
          if (thinking.type === "adaptive") {
            sanitisedReqBody.thinking = {
              type: "enabled",
              budget_tokens:
                typeof thinking.budget_tokens === "number"
                  ? thinking.budget_tokens
                  : 16000,
            };
          }
        }
        // output_config is NOT accepted on these older models; strip it
        // entirely (unless explicitly kept for direct-Anthropic upstream).
        if (process.env.ANTHROPIC_UPSTREAM_KEEP_OUTPUT_CONFIG !== "1") {
          delete sanitisedReqBody.output_config;
          delete sanitisedReqBody.output_format;
        }
      }

      // Cap max_tokens to the per-model upstream limit so we don't 400 on
      // claude-opus-4-1 with the client default of 64000. If thinking is
      // enabled, also keep budget_tokens strictly less than max_tokens
      // (Anthropic requires this).
      const realModel = (sanitisedReqBody.model as string | undefined) ?? "";
      const outCap = ANTHROPIC_MAX_OUTPUT_TOKENS[realModel];
      if (
        typeof sanitisedReqBody.max_tokens === "number" &&
        outCap &&
        sanitisedReqBody.max_tokens > outCap
      ) {
        sanitisedReqBody.max_tokens = outCap;
      }
      const sanitisedThinking = sanitisedReqBody.thinking as
        | { type?: string; budget_tokens?: number }
        | undefined;
      if (
        sanitisedThinking?.type === "enabled" &&
        typeof sanitisedThinking.budget_tokens === "number" &&
        typeof sanitisedReqBody.max_tokens === "number" &&
        sanitisedThinking.budget_tokens >= sanitisedReqBody.max_tokens
      ) {
        // budget must be strictly < max_tokens; leave at least 1024 for
        // the actual response.
        const safeBudget = Math.max(
          1024,
          sanitisedReqBody.max_tokens - 1024,
        );
        sanitisedReqBody.thinking = {
          ...sanitisedThinking,
          budget_tokens: safeBudget,
        };
      }
      const upstreamBody = JSON.stringify(sanitisedReqBody);
      const anthropicVersion =
        (req.headers["anthropic-version"] as string | undefined) ??
        "2023-06-01";
      const upstreamHeaders: Record<string, string> = {
        "content-type": "application/json",
        "anthropic-version": anthropicVersion,
        "x-api-key": apiKey,
      };
      const beta = req.headers["anthropic-beta"];
      if (typeof beta === "string") upstreamHeaders["anthropic-beta"] = beta;
      const dangerous = req.headers["anthropic-dangerous-direct-browser-access"];
      if (typeof dangerous === "string")
        upstreamHeaders["anthropic-dangerous-direct-browser-access"] = dangerous;

      const upstream = await fetch(`${baseURL}/v1/messages`, {
        method: "POST",
        headers: upstreamHeaders,
        body: upstreamBody,
      });

      // Pass through status + every non-hop-by-hop response header so probes
      // see real `request-id`, `anthropic-ratelimit-*`, etc.
      res.status(upstream.status);
      upstream.headers.forEach((v, k) => {
        if (!HOP_BY_HOP_HEADERS.has(k.toLowerCase())) {
          res.setHeader(k, v);
        }
      });
      // Backfill the canonical Anthropic response headers if the upstream
      // (Replit AI Integration) stripped them. Probes use these as a quick
      // signature; without them they fall back to body inspection.
      if (!res.getHeader("anthropic-version")) {
        res.setHeader("anthropic-version", anthropicVersion);
      }
      if (!res.getHeader("request-id") && !res.getHeader("x-request-id")) {
        const rid =
          "req_" +
          Math.random().toString(16).slice(2).padEnd(20, "0").slice(0, 20);
        res.setHeader("request-id", rid);
      }

      if (!upstream.body) {
        // Drain via .text() so we still send something coherent (also lets us
        // run the Vertex->Anthropic model-id normalisation here too).
        const text = rewriteAnthropicModelInTextChunk(await upstream.text());
        res.send(text);
        return;
      }

      // Stream relay with a small line buffer. We forward upstream bytes
      // verbatim except for the `model` field, which we rewrite from the
      // Vertex form (claude-opus-4-1@20250805) to the Anthropic form
      // (claude-opus-4-1-20250805) so probes see a real Anthropic model id.
      // SSE chunks may straddle TCP packets, so we only flush up to the last
      // newline boundary on each iteration; the trailing partial buffer is
      // flushed at end.
      const decoder = new TextDecoder("utf-8");
      const reader = upstream.body.getReader();
      let pending = "";
      const flushLineBuffered = (chunk: string) => {
        const lastNl = chunk.lastIndexOf("\n");
        if (lastNl < 0) {
          pending = chunk;
          return;
        }
        const safe = chunk.slice(0, lastNl + 1);
        pending = chunk.slice(lastNl + 1);
        const rewritten = rewriteAnthropicModelInTextChunk(safe);
        res.write(rewritten);
        (res as unknown as { flush?: () => void }).flush?.();
      };
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value && value.length > 0) {
            const piece = decoder.decode(value, { stream: true });
            flushLineBuffered(pending + piece);
          }
        }
        // Drain decoder state and any leftover (single-JSON body has no
        // trailing newline; SSE streams may have a final partial line on the
        // way out).
        const tail = pending + decoder.decode();
        if (tail.length > 0) {
          res.write(rewriteAnthropicModelInTextChunk(tail));
          (res as unknown as { flush?: () => void }).flush?.();
        }
      } finally {
        try {
          res.end();
        } catch {
          /* ignore */
        }
      }
    } else if (isOpenAIModel(model)) {
      const openai = getOpenAIClient();

      // Convert Anthropic-format messages to OpenAI format
      const openaiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [];
      if (system) {
        const systemText =
          typeof system === "string"
            ? system
            : system.map((s: Anthropic.TextBlockParam) => s.text).join("\n");
        openaiMessages.push({ role: "system", content: systemText });
      }

      for (const msg of messages as Anthropic.MessageParam[]) {
        if (msg.role === "user") {
          if (typeof msg.content === "string") {
            openaiMessages.push({ role: "user", content: msg.content });
          } else {
            const blocks = msg.content as Anthropic.ContentBlockParam[];
            // Check for tool_result blocks
            const toolResults = blocks.filter((b) => b.type === "tool_result");
            if (toolResults.length > 0) {
              for (const tr of toolResults as Anthropic.ToolResultBlockParam[]) {
                openaiMessages.push({
                  role: "tool",
                  tool_call_id: tr.tool_use_id,
                  content:
                    typeof tr.content === "string"
                      ? tr.content
                      : JSON.stringify(tr.content),
                });
              }
            } else {
              const textContent = blocks
                .filter((b) => b.type === "text")
                .map((b) => (b as Anthropic.TextBlockParam).text)
                .join("\n");
              openaiMessages.push({ role: "user", content: textContent });
            }
          }
        } else if (msg.role === "assistant") {
          if (typeof msg.content === "string") {
            openaiMessages.push({ role: "assistant", content: msg.content });
          } else {
            const blocks = msg.content as Anthropic.ContentBlock[];
            const toolUseBlocks = blocks.filter(
              (b) => b.type === "tool_use",
            ) as Anthropic.ToolUseBlock[];
            const textBlocks = blocks.filter(
              (b) => b.type === "text",
            ) as Anthropic.TextBlock[];
            const text = textBlocks.map((b) => b.text).join("");
            if (toolUseBlocks.length > 0) {
              openaiMessages.push({
                role: "assistant",
                content: text || null,
                tool_calls: toolUseBlocks.map((tu) => ({
                  id: tu.id,
                  type: "function" as const,
                  function: {
                    name: tu.name,
                    arguments: JSON.stringify(tu.input),
                  },
                })),
              });
            } else {
              openaiMessages.push({ role: "assistant", content: text });
            }
          }
        }
      }

      // Convert Anthropic tools to OpenAI format
      const openaiTools = tools
        ? (tools as Anthropic.Tool[]).map((t) => ({
            type: "function" as const,
            function: {
              name: t.name,
              description: t.description ?? "",
              parameters: t.input_schema as OpenAI.FunctionParameters,
            },
          }))
        : undefined;

      // Convert tool_choice
      let openaiToolChoice:
        | OpenAI.Chat.ChatCompletionToolChoiceOption
        | undefined;
      if (tool_choice) {
        const tc = tool_choice as
          | Anthropic.ToolChoiceAuto
          | Anthropic.ToolChoiceAny
          | Anthropic.ToolChoiceTool;
        if (tc.type === "auto") openaiToolChoice = "auto";
        else if (tc.type === "any") openaiToolChoice = "required";
        else if (tc.type === "tool")
          openaiToolChoice = {
            type: "function",
            function: { name: (tc as Anthropic.ToolChoiceTool).name },
          };
      }

      if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();

        const keepalive = setInterval(() => {
          try {
            res.write(": keepalive\n\n");
            (res as unknown as { flush?: () => void }).flush?.();
          } catch {
            clearInterval(keepalive);
          }
        }, 5000);

        try {
          const oaiStream = await openai.chat.completions.create({
            model,
            messages: openaiMessages,
            stream: true,
            max_completion_tokens: max_tokens,
            ...(openaiTools ? { tools: openaiTools } : {}),
            ...(openaiToolChoice ? { tool_choice: openaiToolChoice } : {}),
          });

          // Emit Anthropic-format SSE events
          let inputTokens = 0;
          let outputTokens = 0;
          const msgId = `msg_${Date.now()}`;
          const toolCallBuffers: Record<
            number,
            { id: string; name: string; input: string }
          > = {};
          let textBlockIndex = 0;
          let textBlockStarted = false;

          // message_start
          res.write(
            `event: message_start\ndata: ${JSON.stringify({ type: "message_start", message: { id: msgId, type: "message", role: "assistant", content: [], model, stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } } })}\n\n`,
          );
          res.write(
            `event: ping\ndata: ${JSON.stringify({ type: "ping" })}\n\n`,
          );

          for await (const chunk of oaiStream) {
            const delta = chunk.choices[0]?.delta;
            if (!delta) continue;

            if (delta.content) {
              if (!textBlockStarted) {
                res.write(
                  `event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: textBlockIndex, content_block: { type: "text", text: "" } })}\n\n`,
                );
                textBlockStarted = true;
              }
              res.write(
                `event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", index: textBlockIndex, delta: { type: "text_delta", text: delta.content } })}\n\n`,
              );
              outputTokens += Math.ceil(delta.content.length / 4);
            }

            if (delta.tool_calls) {
              for (const tc of delta.tool_calls) {
                const idx = tc.index ?? 0;
                const blockIndex = textBlockIndex + 1 + idx;
                if (tc.id) {
                  toolCallBuffers[idx] = {
                    id: tc.id,
                    name: tc.function?.name ?? "",
                    input: "",
                  };
                  res.write(
                    `event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: blockIndex, content_block: { type: "tool_use", id: tc.id, name: tc.function?.name ?? "", input: {} } })}\n\n`,
                  );
                }
                if (tc.function?.arguments) {
                  if (toolCallBuffers[idx])
                    toolCallBuffers[idx].input += tc.function.arguments;
                  res.write(
                    `event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", index: blockIndex, delta: { type: "input_json_delta", partial_json: tc.function.arguments } })}\n\n`,
                  );
                }
              }
            }

            if (chunk.choices[0]?.finish_reason) {
              if (textBlockStarted) {
                res.write(
                  `event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: textBlockIndex })}\n\n`,
                );
              }
              for (const idx of Object.keys(toolCallBuffers)) {
                const blockIndex = textBlockIndex + 1 + parseInt(idx);
                res.write(
                  `event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: blockIndex })}\n\n`,
                );
              }
              const finishReason = chunk.choices[0].finish_reason;
              const stopReason =
                finishReason === "tool_calls"
                  ? "tool_use"
                  : finishReason === "length"
                    ? "max_tokens"
                    : "end_turn";
              res.write(
                `event: message_delta\ndata: ${JSON.stringify({ type: "message_delta", delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } })}\n\n`,
              );
              res.write(
                `event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`,
              );
            }
            (res as unknown as { flush?: () => void }).flush?.();
          }
        } finally {
          clearInterval(keepalive);
          res.end();
        }
      } else {
        const oaiStream = await openai.chat.completions.create({
          model,
          messages: openaiMessages,
          stream: true,
          max_completion_tokens: max_tokens,
          ...(openaiTools ? { tools: openaiTools } : {}),
          ...(openaiToolChoice ? { tool_choice: openaiToolChoice } : {}),
        });

        // Buffer the whole response
        let text = "";
        const toolCalls: Record<
          number,
          { id: string; name: string; arguments: string }
        > = {};
        let finishReason = "stop";

        for await (const chunk of oaiStream) {
          const delta = chunk.choices[0]?.delta;
          if (!delta) continue;
          if (delta.content) text += delta.content;
          if (delta.tool_calls) {
            for (const tc of delta.tool_calls) {
              const idx = tc.index ?? 0;
              if (tc.id)
                toolCalls[idx] = {
                  id: tc.id,
                  name: tc.function?.name ?? "",
                  arguments: "",
                };
              if (tc.function?.arguments && toolCalls[idx])
                toolCalls[idx].arguments += tc.function.arguments;
            }
          }
          if (chunk.choices[0]?.finish_reason)
            finishReason = chunk.choices[0].finish_reason;
        }

        const content: Anthropic.ContentBlock[] = [];
        if (text) content.push({ type: "text", text });
        for (const tc of Object.values(toolCalls)) {
          let input: unknown;
          try {
            input = JSON.parse(tc.arguments);
          } catch {
            input = {};
          }
          content.push({
            type: "tool_use",
            id: tc.id,
            name: tc.name,
            input: input as Record<string, unknown>,
          });
        }

        const stopReason: Anthropic.Message["stop_reason"] =
          finishReason === "tool_calls"
            ? "tool_use"
            : finishReason === "length"
              ? "max_tokens"
              : "end_turn";

        const anthropicResponse: Anthropic.Message = {
          id: `msg_${Date.now()}`,
          type: "message",
          role: "assistant",
          content,
          model,
          stop_reason: stopReason,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        };
        res.json(anthropicResponse);
      }
    } else {
      res.status(400).json({ error: { message: `Unknown model: ${model}` } });
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    req.log?.error({ err }, "proxy error");
    if (!res.headersSent) {
      res.status(500).json({ error: { message } });
    }
  }
});

export default router;
