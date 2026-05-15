import { Router, type Request, type Response } from "express";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

const router = Router();

const OPENAI_MODELS = ["gpt-5.2", "gpt-5-mini", "gpt-5-nano", "o4-mini", "o3"];
const ANTHROPIC_MODELS = [
  "claude-opus-4-7",
  "claude-opus-4-6",
  "claude-opus-4-6-thinking",
  "claude-sonnet-4-6",
  "claude-haiku-4-5",
];
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
      const anthropic = getAnthropicClient();

      const params: Anthropic.MessageCreateParamsNonStreaming = {
        model,
        messages,
        max_tokens,
        ...(system ? { system } : {}),
        ...(tools ? { tools } : {}),
        ...(tool_choice ? { tool_choice } : {}),
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
          const msgStream = anthropic.messages.stream(params);
          for await (const event of msgStream) {
            res.write(
              `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`,
            );
            (res as unknown as { flush?: () => void }).flush?.();
          }
        } finally {
          clearInterval(keepalive);
          res.end();
        }
      } else {
        const msgStream = anthropic.messages.stream(params);
        const finalMsg = await msgStream.finalMessage();
        res.json(finalMsg);
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
