import { object } from './config-file-api'
import type { ModelConfig, ModelKind } from './config-file-api'

const bases: Record<string, string> = {
  openai: 'https://api.openai.com/v1',
  volcengine: 'https://ark.cn-beijing.volces.com/api/v3',
  glm: 'https://api.z.ai/api/coding/paas/v4',
}
export class ModelTestError extends Error {
  constructor(
    public readonly code:
      | 'unsupported'
      | 'missing'
      | 'invalidUrl'
      | 'network'
      | 'http'
      | 'invalidResponse',
    public readonly detail = '',
  ) {
    super(code)
  }
}
export function supportsBrowserModelTest(kind: ModelKind, provider: unknown) {
  return (
    kind === 'vlm'
      ? ['openai', 'volcengine', 'azure', 'glm', 'litellm']
      : ['openai', 'azure']
  ).includes(String(provider || '').toLowerCase())
}

/** Direct provider request: never uses Studio's API client or its admin credentials. */
export async function testModelConnection({
  kind,
  config,
  fallback,
  thinking,
  signal,
}: {
  kind: ModelKind
  config: ModelConfig
  fallback: ModelConfig
  thinking?: boolean
  signal: AbortSignal
}) {
  const provider = String(config.provider || '').toLowerCase()
  if (!supportsBrowserModelTest(kind, provider))
    throw new ModelTestError('unsupported')
  const model = String(config.model || fallback.model || '').trim()
  const apiKey = String(config.api_key || fallback.api_key || '').trim()
  const base = String(
    config.api_base || fallback.api_base || bases[provider] || '',
  ).trim()
  const keyOptional =
    provider === 'litellm' ||
    (kind === 'embedding' &&
      provider === 'openai' &&
      Boolean(config.api_base || fallback.api_base))
  if (!model || !base || (!apiKey && !keyOptional))
    throw new ModelTestError('missing')
  let url: URL
  try {
    url = new URL(base)
    if (
      !['http:', 'https:'].includes(url.protocol) ||
      url.username ||
      url.password ||
      url.search ||
      url.hash
    )
      throw new Error()
  } catch {
    throw new ModelTestError('invalidUrl')
  }
  const path = kind === 'vlm' ? 'chat/completions' : 'embeddings'
  let endpoint = base.replace(/\/+$/, '')
  if (provider === 'azure') {
    if (!endpoint.endsWith('/openai')) endpoint += '/openai'
    endpoint += `/deployments/${encodeURIComponent(model)}/${path}`
    url = new URL(endpoint)
    url.searchParams.set(
      'api-version',
      String(
        config.api_version || fallback.api_version || '2025-01-01-preview',
      ),
    )
  } else url = new URL(`${endpoint}/${path}`)
  const headers = new Headers()
  const extraHeaders = object(config.extra_headers || fallback.extra_headers)
  for (const [key, value] of Object.entries(extraHeaders)) {
    if (typeof value === 'string') headers.set(key, value)
  }
  headers.set('Content-Type', 'application/json')
  if (apiKey)
    headers.set(
      provider === 'azure' ? 'api-key' : 'Authorization',
      provider === 'azure' ? apiKey : `Bearer ${apiKey}`,
    )
  const body: ModelConfig =
    kind === 'vlm'
      ? {
          model,
          messages: [{ role: 'user', content: 'Reply only OK.' }],
          ...object(config.extra_request_body || fallback.extra_request_body),
          stream: false,
        }
      : { model, input: 'connection test' }
  if (kind === 'vlm') {
    // Match OpenAIVLM._apply_completion_params; Volcengine and LiteLLM
    // use their own backend parameter handling.
    const reasoningModel =
      ['openai', 'azure', 'glm'].includes(provider) &&
      /^(?:gpt-(?:[5-9]|[1-9]\d)|o[134])/i.test(model)
    if (provider === 'volcengine')
      body.thinking = { type: thinking ? 'enabled' : 'disabled' }
    for (const key of ['max_tokens', 'reasoning_effort']) {
      const value = config[key] ?? fallback[key]
      if (value !== undefined && value !== null)
        body[
          key === 'max_tokens' && reasoningModel ? 'max_completion_tokens' : key
        ] = value
    }
  }
  const started = performance.now()
  let response: Response
  try {
    response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal,
      mode: 'cors',
      credentials: 'omit',
      redirect: 'error',
      referrerPolicy: 'no-referrer',
    })
  } catch (error) {
    if (signal.aborted) throw error
    throw new ModelTestError('network')
  }
  let result: ModelConfig
  try {
    result = object(await response.json())
  } catch {
    signal.throwIfAborted()
    if (!response.ok)
      throw new ModelTestError('http', `HTTP ${response.status}`)
    throw new ModelTestError('invalidResponse')
  }
  signal.throwIfAborted()
  if (!response.ok) {
    let message = String(object(result.error).message || '')
    const secrets = [
      apiKey,
      ...Object.values(extraHeaders).filter(
        (value): value is string => typeof value === 'string',
      ),
    ].filter(Boolean)
    for (const secret of secrets)
      message = message.replaceAll(secret, '[redacted]')
    throw new ModelTestError(
      'http',
      `HTTP ${response.status}${message ? `: ${message.slice(0, 400)}` : ''}`,
    )
  }
  if (kind === 'vlm') {
    const choice = Array.isArray(result.choices)
      ? object(result.choices[0])
      : {}
    const message = object(choice.message)
    const content = message.content
    const validContent =
      typeof content === 'string'
        ? Boolean(content.trim())
        : Array.isArray(content) &&
          content.some((part) => {
            const text = object(part).text
            return typeof text === 'string' && Boolean(text.trim())
          })
    const reasoning = message.reasoning_content
    if (!validContent && !(typeof reasoning === 'string' && reasoning.trim()))
      throw new ModelTestError('invalidResponse')
  } else {
    const first = Array.isArray(result.data) ? object(result.data[0]) : {}
    if (
      !Array.isArray(first.embedding) ||
      !first.embedding.length ||
      !first.embedding.every(
        (value) => typeof value === 'number' && Number.isFinite(value),
      )
    )
      throw new ModelTestError('invalidResponse')
  }
  return Math.round(performance.now() - started)
}
