import type { StreamFn } from "@mariozechner/pi-agent-core";
import {
  createAssistantMessageEventStream,
  type AssistantMessage,
  type Usage,
} from "@mariozechner/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  __testing,
  priceHaikuUsage,
  createAnthropicBetaHeadersWrapper,
  createAnthropicFastModeWrapper,
  createAnthropicServiceTierWrapper,
  wrapAnthropicProviderStream,
} from "./stream-wrappers.js";

const CONTEXT_1M_BETA = "context-1m-2025-08-07";
const OAUTH_BETA = "oauth-2025-04-20";

function runWrapper(apiKey: string | undefined): Record<string, string> | undefined {
  const captured: { headers?: Record<string, string> } = {};
  const base: StreamFn = (_model, _context, options) => {
    captured.headers = options?.headers;
    return {} as never;
  };
  const wrapper = createAnthropicBetaHeadersWrapper(base, [CONTEXT_1M_BETA]);
  wrapper(
    { provider: "anthropic", id: "claude-opus-4-6" } as never,
    {} as never,
    { apiKey } as never,
  );
  return captured.headers;
}

describe("anthropic stream wrappers", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("adapts Haiku 5.5 payloads without losing an explicit output limit or tool choice", () => {
    const captured: { payload?: Record<string, unknown> } = {};
    const base: StreamFn = (model, _context, options) => {
      const payload = {
        temperature: 0,
        top_p: 0.9,
        top_k: 5,
        max_tokens: 8192,
        thinking: { type: "enabled", budget_tokens: 4096 },
        tool_choice: { type: "any" },
      };
      options?.onPayload?.(payload as never, model as never);
      captured.payload = payload;
      return createAssistantMessageEventStream();
    };
    const wrapped = wrapAnthropicProviderStream({
      streamFn: base,
      modelId: "claude-haiku-5-5",
    } as never);
    wrapped?.(
      { id: "claude-haiku-5-5", provider: "anthropic", api: "anthropic-messages" } as never,
      {} as never,
      { maxTokens: 2048, reasoning: "medium" } as never,
    );
    expect(captured.payload).toMatchObject({
      max_tokens: 2048,
      thinking: { type: "adaptive" },
      output_config: { effort: "medium" },
      tool_choice: { type: "any" },
    });
    for (const key of ["temperature", "top_p", "top_k"]) {
      expect(captured.payload).not.toHaveProperty(key);
    }
    expect(captured.payload?.thinking).not.toHaveProperty("budget_tokens");
  });

  it.each(["claude-haiku-5-5", " Claude-Haiku-5-5 "])("normalizes %s registration and clamps xhigh effort", (modelId) => {
    let payload: Record<string, unknown> = {};
    const base: StreamFn = (model, _context, options) => {
      payload = { temperature: 0, top_p: 0.9, top_k: 5 };
      options?.onPayload?.(payload as never, model as never);
      const stream = createAssistantMessageEventStream();
      stream.end();
      return stream;
    };
    const wrapped = wrapAnthropicProviderStream({ streamFn: base, modelId } as never);
    wrapped?.({ id: modelId, provider: "anthropic", api: "anthropic-messages" } as never,
      {} as never, { reasoning: "xhigh" } as never);
    expect(payload).toMatchObject({ thinking: { type: "adaptive" }, output_config: { effort: "high" } });
    for (const key of ["temperature", "top_p", "top_k"]) {
      expect(payload).not.toHaveProperty(key);
    }
  });

  it("explicitly disables Haiku 5.5 thinking when the caller turns it off", () => {
    let payload: Record<string, unknown> = {};
    const base: StreamFn = (model, _context, options) => {
      options?.onPayload?.(payload as never, model as never);
      return createAssistantMessageEventStream();
    };
    const wrapped = wrapAnthropicProviderStream({
      streamFn: base,
      modelId: "claude-haiku-5-5",
    } as never);
    wrapped?.(
      { id: "claude-haiku-5-5", provider: "anthropic", api: "anthropic-messages" } as never,
      {} as never,
      {} as never,
    );
    expect(payload).toMatchObject({ thinking: { type: "disabled" } });
  });

  it.each([
    [100_000, 0, 0, "5m", 0.0105],
    [99_999, 1, 1, "5m", 0.052500175],
    [0, 100_001, 0, "5m", 0.00750005],
    [0, 0, 100_001, "5m", 0.065000625],
    [0, 0, 100_000, "1h", 0.0205],
    [0, 0, 100_001, "1h", 0.102501],
  ] as const)(
    "prices the entire prompt tier with cache classes (%i/%i/%i, %s)",
    (input, cacheRead, cacheWrite, ttl, expected) => {
      const usage = { input, output: 1000, cacheRead, cacheWrite } as Usage;
      priceHaikuUsage(usage, ttl);
      expect(usage.cost.total).toBeCloseTo(expected, 10);
    },
  );

  it("reports corrected costs through both streamed events and result()", async () => {
    const message: AssistantMessage = {
      role: "assistant",
      content: [],
      api: "anthropic-messages",
      provider: "anthropic",
      model: "claude-haiku-5-5",
      timestamp: 0,
      stopReason: "stop",
      usage: {
        input: 100001,
        output: 1000,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 101001,
        cost: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, total: 2 },
      },
    };
    const base: StreamFn = async () => {
      const stream = createAssistantMessageEventStream();
      stream.push({ type: "done", reason: "stop", message });
      stream.end(message);
      return stream;
    };
    const wrapped = wrapAnthropicProviderStream({
      streamFn: base,
      modelId: "claude-haiku-5-5",
    } as never)!;
    const stream = await wrapped(
      { id: "claude-haiku-5-5", provider: "anthropic", api: "anthropic-messages" } as never,
      {} as never,
    );
    for await (const event of stream) {
      if (event.type === "done") {
        expect(event.message.usage.cost.total).toBeCloseTo(0.0525005, 10);
      }
    }
    expect((await stream.result()).usage.cost.total).toBeCloseTo(0.0525005, 10);
  });

  it("strips context-1m for Claude CLI or legacy token auth and warns", () => {
    const warn = vi.spyOn(__testing.log, "warn").mockImplementation(() => undefined);
    const headers = runWrapper("sk-ant-oat01-123");
    expect(headers?.["anthropic-beta"]).toBeDefined();
    expect(headers?.["anthropic-beta"]).toContain(OAUTH_BETA);
    expect(headers?.["anthropic-beta"]).not.toContain(CONTEXT_1M_BETA);
    expect(warn).toHaveBeenCalledOnce();
  });

  it("keeps context-1m for API key auth", () => {
    const warn = vi.spyOn(__testing.log, "warn").mockImplementation(() => undefined);
    const headers = runWrapper("sk-ant-api-123");
    expect(headers?.["anthropic-beta"]).toBeDefined();
    expect(headers?.["anthropic-beta"]).toContain(CONTEXT_1M_BETA);
    expect(warn).not.toHaveBeenCalled();
  });

  it("skips service_tier for OAuth token in composed stream chain", () => {
    const captured: { headers?: Record<string, string>; payload?: Record<string, unknown> } = {};
    const base: StreamFn = (model, _context, options) => {
      captured.headers = options?.headers;
      const payload = {} as Record<string, unknown>;
      options?.onPayload?.(payload as never, model as never);
      captured.payload = payload;
      return {} as never;
    };

    const wrapped = wrapAnthropicProviderStream({
      streamFn: base,
      modelId: "claude-sonnet-4-6",
      extraParams: { context1m: true, serviceTier: "auto" },
    } as never);

    wrapped?.(
      { provider: "anthropic", api: "anthropic-messages", id: "claude-sonnet-4-6" } as never,
      {} as never,
      { apiKey: "sk-ant-oat01-oauth-token" } as never,
    );

    expect(captured.headers?.["anthropic-beta"]).toContain(OAUTH_BETA);
    expect(captured.headers?.["anthropic-beta"]).not.toContain(CONTEXT_1M_BETA);
    expect(captured.payload?.service_tier).toBeUndefined();
  });

  it("composes the anthropic provider stream chain from extra params", () => {
    const captured: { headers?: Record<string, string>; payload?: Record<string, unknown> } = {};
    const base: StreamFn = (model, _context, options) => {
      captured.headers = options?.headers;
      const payload = {} as Record<string, unknown>;
      options?.onPayload?.(payload as never, model as never);
      captured.payload = payload;
      return {} as never;
    };

    const wrapped = wrapAnthropicProviderStream({
      streamFn: base,
      modelId: "claude-sonnet-4-6",
      extraParams: { context1m: true, serviceTier: "auto" },
    } as never);

    wrapped?.(
      { provider: "anthropic", api: "anthropic-messages", id: "claude-sonnet-4-6" } as never,
      {} as never,
      { apiKey: "sk-ant-api-123" } as never,
    );

    expect(captured.headers?.["anthropic-beta"]).toContain(CONTEXT_1M_BETA);
    expect(captured.payload).toMatchObject({ service_tier: "auto" });
  });
});

describe("createAnthropicFastModeWrapper", () => {
  function runFastModeWrapper(params: {
    apiKey?: string;
    provider?: string;
    api?: string;
    baseUrl?: string;
    enabled?: boolean;
  }): Record<string, unknown> | undefined {
    const captured: { payload?: Record<string, unknown> } = {};
    const base: StreamFn = (_model, _context, options) => {
      if (options?.onPayload) {
        const payload: Record<string, unknown> = {};
        options.onPayload(payload, _model);
        captured.payload = payload;
      }
      return {} as never;
    };

    const wrapper = createAnthropicFastModeWrapper(base, params.enabled ?? true);
    wrapper(
      {
        provider: params.provider ?? "anthropic",
        api: params.api ?? "anthropic-messages",
        baseUrl: params.baseUrl,
        id: "claude-sonnet-4-6",
      } as never,
      {} as never,
      { apiKey: params.apiKey } as never,
    );
    return captured.payload;
  }

  it("does not inject service_tier for OAuth token", () => {
    const payload = runFastModeWrapper({ apiKey: "sk-ant-oat01-test-token" });
    expect(payload?.service_tier).toBeUndefined();
  });

  it("injects service_tier for regular API keys", () => {
    const payload = runFastModeWrapper({ apiKey: "sk-ant-api03-test-key" });
    expect(payload?.service_tier).toBe("auto");
  });

  it("injects service_tier=standard_only when disabled for API keys", () => {
    const payload = runFastModeWrapper({ apiKey: "sk-ant-api03-test-key", enabled: false });
    expect(payload?.service_tier).toBe("standard_only");
  });

  it("does not inject service_tier for non-anthropic provider", () => {
    const payload = runFastModeWrapper({
      apiKey: "sk-ant-api03-test-key",
      provider: "openai",
      api: "openai-completions",
    });
    expect(payload?.service_tier).toBeUndefined();
  });
});

describe("createAnthropicServiceTierWrapper", () => {
  function runServiceTierWrapper(params: {
    apiKey?: string;
    provider?: string;
    api?: string;
    serviceTier?: "auto" | "standard_only";
  }): Record<string, unknown> | undefined {
    const captured: { payload?: Record<string, unknown> } = {};
    const base: StreamFn = (_model, _context, options) => {
      if (options?.onPayload) {
        const payload: Record<string, unknown> = {};
        options.onPayload(payload, _model);
        captured.payload = payload;
      }
      return {} as never;
    };

    const wrapper = createAnthropicServiceTierWrapper(base, params.serviceTier ?? "auto");
    wrapper(
      {
        provider: params.provider ?? "anthropic",
        api: params.api ?? "anthropic-messages",
        id: "claude-sonnet-4-6",
      } as never,
      {} as never,
      { apiKey: params.apiKey } as never,
    );
    return captured.payload;
  }

  it("does not inject service_tier for OAuth token", () => {
    const payload = runServiceTierWrapper({ apiKey: "sk-ant-oat01-test-token" });
    expect(payload?.service_tier).toBeUndefined();
  });

  it("injects service_tier for regular API keys", () => {
    const payload = runServiceTierWrapper({ apiKey: "sk-ant-api03-test-key" });
    expect(payload?.service_tier).toBe("auto");
  });

  it("injects service_tier=standard_only for regular API keys", () => {
    const payload = runServiceTierWrapper({
      apiKey: "sk-ant-api03-test-key",
      serviceTier: "standard_only",
    });
    expect(payload?.service_tier).toBe("standard_only");
  });

  it("does not inject service_tier for non-anthropic provider", () => {
    const payload = runServiceTierWrapper({
      apiKey: "sk-ant-api03-test-key",
      provider: "openai",
      api: "openai-completions",
    });
    expect(payload?.service_tier).toBeUndefined();
  });
});
