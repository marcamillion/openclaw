import type { StreamFn } from "@mariozechner/pi-agent-core";
import {
  createAssistantMessageEventStream,
  streamSimple,
  type AssistantMessage,
  type Usage,
} from "@mariozechner/pi-ai";
import type { ProviderWrapStreamFnContext } from "openclaw/plugin-sdk/plugin-entry";
import {
  applyAnthropicPayloadPolicyToParams,
  composeProviderStreamWrappers,
  resolveAnthropicPayloadPolicy,
  streamWithPayloadPatch,
} from "openclaw/plugin-sdk/provider-stream-shared";
import { createSubsystemLogger } from "openclaw/plugin-sdk/runtime-env";
import { normalizeOptionalString, readStringValue } from "openclaw/plugin-sdk/text-runtime";

const log = createSubsystemLogger("anthropic-stream");

const ANTHROPIC_CONTEXT_1M_BETA = "context-1m-2025-08-07";
const ANTHROPIC_1M_MODEL_PREFIXES = ["claude-opus-4", "claude-sonnet-4"] as const;
const PI_AI_DEFAULT_ANTHROPIC_BETAS = [
  "fine-grained-tool-streaming-2025-05-14",
  "interleaved-thinking-2025-05-14",
] as const;
const PI_AI_OAUTH_ANTHROPIC_BETAS = [
  "claude-code-20250219",
  "oauth-2025-04-20",
  ...PI_AI_DEFAULT_ANTHROPIC_BETAS,
] as const;

type AnthropicServiceTier = "auto" | "standard_only";

function isAnthropic1MModel(modelId: string): boolean {
  const normalized = modelId.trim().toLowerCase();
  return ANTHROPIC_1M_MODEL_PREFIXES.some((prefix) => normalized.startsWith(prefix));
}

function parseHeaderList(value: unknown): string[] {
  if (typeof value !== "string") {
    return [];
  }
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function mergeAnthropicBetaHeader(
  headers: Record<string, string> | undefined,
  betas: string[],
): Record<string, string> {
  const merged = { ...headers };
  const existingKey = Object.keys(merged).find((key) => key.toLowerCase() === "anthropic-beta");
  const existing = existingKey ? parseHeaderList(merged[existingKey]) : [];
  const values = Array.from(new Set([...existing, ...betas]));
  const key = existingKey ?? "anthropic-beta";
  merged[key] = values.join(",");
  return merged;
}

function isAnthropicOAuthApiKey(apiKey: unknown): boolean {
  return typeof apiKey === "string" && apiKey.includes("sk-ant-oat");
}

function resolveAnthropicFastServiceTier(enabled: boolean): AnthropicServiceTier {
  return enabled ? "auto" : "standard_only";
}

function normalizeFastMode(raw?: string | boolean | null): boolean | undefined {
  if (typeof raw === "boolean") {
    return raw;
  }
  if (!raw) {
    return undefined;
  }
  const key = raw.toLowerCase();
  if (["off", "false", "no", "0", "disable", "disabled", "normal"].includes(key)) {
    return false;
  }
  if (["on", "true", "yes", "1", "enable", "enabled", "fast"].includes(key)) {
    return true;
  }
  return undefined;
}

function normalizeAnthropicServiceTier(value: unknown): AnthropicServiceTier | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const normalized = normalizeOptionalString(value)?.toLowerCase();
  if (normalized === "auto" || normalized === "standard_only") {
    return normalized;
  }
  return undefined;
}

export function resolveAnthropicBetas(
  extraParams: Record<string, unknown> | undefined,
  modelId: string,
): string[] | undefined {
  const betas = new Set<string>();
  const configured = extraParams?.anthropicBeta;
  if (typeof configured === "string" && configured.trim()) {
    betas.add(configured.trim());
  } else if (Array.isArray(configured)) {
    for (const beta of configured) {
      if (typeof beta === "string" && beta.trim()) {
        betas.add(beta.trim());
      }
    }
  }

  if (extraParams?.context1m === true) {
    if (isAnthropic1MModel(modelId)) {
      betas.add(ANTHROPIC_CONTEXT_1M_BETA);
    } else if (modelId.trim().toLowerCase() !== "claude-haiku-5-5") {
      log.warn(`ignoring context1m for non-opus/sonnet model: anthropic/${modelId}`);
    }
  }

  return betas.size > 0 ? [...betas] : undefined;
}

export function createAnthropicBetaHeadersWrapper(
  baseStreamFn: StreamFn | undefined,
  betas: string[],
): StreamFn {
  const underlying = baseStreamFn ?? streamSimple;
  return (model, context, options) => {
    const isOauth = isAnthropicOAuthApiKey(options?.apiKey);
    const requestedContext1m = betas.includes(ANTHROPIC_CONTEXT_1M_BETA);
    const effectiveBetas =
      isOauth && requestedContext1m
        ? betas.filter((beta) => beta !== ANTHROPIC_CONTEXT_1M_BETA)
        : betas;
    if (isOauth && requestedContext1m) {
      log.warn(
        `ignoring context1m for Anthropic Claude CLI or legacy token auth on ${model.provider}/${model.id}; falling back to the standard context window because Anthropic rejects context-1m beta with non-API-key auth`,
      );
    }

    const piAiBetas = isOauth
      ? (PI_AI_OAUTH_ANTHROPIC_BETAS as readonly string[])
      : (PI_AI_DEFAULT_ANTHROPIC_BETAS as readonly string[]);
    const allBetas = [...new Set([...piAiBetas, ...effectiveBetas])];
    return underlying(model, context, {
      ...options,
      headers: mergeAnthropicBetaHeader(options?.headers, allBetas),
    });
  };
}

export function createAnthropicFastModeWrapper(
  baseStreamFn: StreamFn | undefined,
  enabled: boolean,
): StreamFn {
  const underlying = baseStreamFn ?? streamSimple;
  const serviceTier = resolveAnthropicFastServiceTier(enabled);
  return (model, context, options) => {
    if (isAnthropicOAuthApiKey(options?.apiKey)) {
      return underlying(model, context, options);
    }

    const payloadPolicy = resolveAnthropicPayloadPolicy({
      provider: readStringValue(model.provider),
      api: readStringValue(model.api),
      baseUrl: readStringValue(model.baseUrl),
      serviceTier,
    });
    if (!payloadPolicy.allowsServiceTier) {
      return underlying(model, context, options);
    }

    return streamWithPayloadPatch(underlying, model, context, options, (payloadObj) =>
      applyAnthropicPayloadPolicyToParams(payloadObj, payloadPolicy),
    );
  };
}

export function createAnthropicServiceTierWrapper(
  baseStreamFn: StreamFn | undefined,
  serviceTier: AnthropicServiceTier,
): StreamFn {
  const underlying = baseStreamFn ?? streamSimple;
  return (model, context, options) => {
    if (isAnthropicOAuthApiKey(options?.apiKey)) {
      return underlying(model, context, options);
    }

    const payloadPolicy = resolveAnthropicPayloadPolicy({
      provider: readStringValue(model.provider),
      api: readStringValue(model.api),
      baseUrl: readStringValue(model.baseUrl),
      serviceTier,
    });
    if (!payloadPolicy.allowsServiceTier) {
      return underlying(model, context, options);
    }

    return streamWithPayloadPatch(underlying, model, context, options, (payloadObj) =>
      applyAnthropicPayloadPolicyToParams(payloadObj, payloadPolicy),
    );
  };
}

export function resolveAnthropicFastMode(
  extraParams: Record<string, unknown> | undefined,
): boolean | undefined {
  return normalizeFastMode(
    (extraParams?.fastMode ?? extraParams?.fast_mode) as string | boolean | null | undefined,
  );
}

export function resolveAnthropicServiceTier(
  extraParams: Record<string, unknown> | undefined,
): AnthropicServiceTier | undefined {
  const raw = extraParams?.serviceTier ?? extraParams?.service_tier;
  const normalized = normalizeAnthropicServiceTier(raw);
  if (raw !== undefined && normalized === undefined) {
    const rawSummary = typeof raw === "string" ? raw : typeof raw;
    log.warn(`ignoring invalid Anthropic service tier param: ${rawSummary}`);
  }
  return normalized;
}

// The catalog SDK predates Haiku 5.5. Apply its native payload and usage contract here.
export function priceHaikuUsage(usage: Usage, cacheTtl: "5m" | "1h" = "5m"): void {
  const longPrompt = usage.input + usage.cacheRead + usage.cacheWrite > 100_000;
  const input = longPrompt ? 0.5 : 0.1;
  const output = longPrompt ? 2.5 : 0.5;
  const cacheRead = longPrompt ? 0.05 : 0.01;
  const cacheWrite = input * (cacheTtl === "1h" ? 2 : 1.25);
  usage.cost = {
    input: (usage.input * input) / 1_000_000,
    output: (usage.output * output) / 1_000_000,
    cacheRead: (usage.cacheRead * cacheRead) / 1_000_000,
    cacheWrite: (usage.cacheWrite * cacheWrite) / 1_000_000,
    total: 0,
  };
  usage.cost.total =
    usage.cost.input + usage.cost.output + usage.cost.cacheRead + usage.cost.cacheWrite;
}

function createHaikuStreamWrapper(baseStreamFn: StreamFn | undefined): StreamFn {
  const underlying = baseStreamFn ?? streamSimple;
  return (model, context, options) => {
    if (model.id.trim().toLowerCase() !== "claude-haiku-5-5" || model.api !== "anthropic-messages") {
      return underlying(model, context, options);
    }
    let cacheTtl: "5m" | "1h" = "5m";
    const source = streamWithPayloadPatch(underlying, model, context, options, (payload) => {
      delete payload.temperature;
      delete payload.top_p;
      delete payload.top_k;
      payload.thinking = { type: options?.reasoning ? "adaptive" : "disabled" };
      if (options?.reasoning) {
        payload.output_config = {
          ...(payload.output_config as Record<string, unknown> | undefined),
          effort: options.reasoning === "minimal" ? "low" : options.reasoning === "xhigh" ? "high" : options.reasoning,
        };
      }
      // The older SDK adds a thinking budget to max_tokens. Adaptive uses the caller's ceiling.
      if (options?.maxTokens !== undefined) {
        payload.max_tokens = options.maxTokens;
      }
      const visit = (value: unknown): void => {
        if (!value || typeof value !== "object") {
          return;
        }
        const object = value as Record<string, unknown>;
        const control = object.cache_control as Record<string, unknown> | undefined;
        if (control?.ttl === "1h") {
          cacheTtl = "1h";
        }
        for (const nested of Object.values(object)) {
          visit(nested);
        }
      };
      visit(payload);
    });
    const output = createAssistantMessageEventStream();
    void (async () => {
      try {
        const stream = await source;
        for await (const event of stream) {
          const message =
            event.type === "done"
              ? event.message
              : event.type === "error"
                ? event.error
                : event.partial;
          priceHaikuUsage(message.usage, cacheTtl);
          output.push(event);
        }
        const final = await stream.result();
        priceHaikuUsage(final.usage, cacheTtl);
        output.end(final);
      } catch (error) {
        const failed: AssistantMessage = {
          role: "assistant",
          content: [],
          api: model.api,
          provider: model.provider,
          model: model.id,
          timestamp: Date.now(),
          stopReason: "error",
          errorMessage: String(error),
          usage: {
            input: 0,
            output: 0,
            cacheRead: 0,
            cacheWrite: 0,
            totalTokens: 0,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
          },
        };
        output.push({ type: "error", reason: "error", error: failed });
        output.end(failed);
      }
    })();
    return output;
  };
}

export function wrapAnthropicProviderStream(
  ctx: ProviderWrapStreamFnContext,
): StreamFn | undefined {
  const anthropicBetas = resolveAnthropicBetas(ctx.extraParams, ctx.modelId);
  const serviceTier = resolveAnthropicServiceTier(ctx.extraParams);
  const fastMode = resolveAnthropicFastMode(ctx.extraParams);
  return composeProviderStreamWrappers(
    ctx.streamFn,
    ctx.modelId.trim().toLowerCase() === "claude-haiku-5-5" ? createHaikuStreamWrapper : undefined,
    anthropicBetas?.length
      ? (streamFn) => createAnthropicBetaHeadersWrapper(streamFn, anthropicBetas)
      : undefined,
    serviceTier
      ? (streamFn) => createAnthropicServiceTierWrapper(streamFn, serviceTier)
      : undefined,
    fastMode !== undefined
      ? (streamFn) => createAnthropicFastModeWrapper(streamFn, fastMode)
      : undefined,
  );
}

export const __testing = { log };
