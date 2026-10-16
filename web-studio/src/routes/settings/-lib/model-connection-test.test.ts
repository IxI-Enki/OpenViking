import { afterEach, expect, it, vi } from 'vitest'
import { testModelConnection } from './model-connection-test'
import type { ModelConfig } from './config-file-api'

afterEach(() => vi.unstubAllGlobals())
function test(
  config: ModelConfig = {},
  kind: 'vlm' | 'embedding' = 'vlm',
  fallback: ModelConfig = {},
) {
  return testModelConnection({
    kind,
    config: {
      provider: 'volcengine',
      model: 'draft-model',
      api_key: 'test-secret',
      api_base: 'https://model.example/api/plan/v3',
      ...config,
    },
    fallback,
    thinking: false,
    signal: new AbortController().signal,
  })
}
function mockResponse(body: unknown, status = 200) {
  const fetch = vi
    .fn()
    .mockResolvedValue(new Response(JSON.stringify(body), { status }))
  vi.stubGlobal('fetch', fetch)
  return fetch
}
it('tests the draft endpoint with model credentials and thinking, without Studio authentication', async () => {
  const fetch = mockResponse({ choices: [{ message: { content: 'OK' } }] })
  expect(await test()).toBeGreaterThanOrEqual(0)
  const [url, init] = fetch.mock.calls[0]
  expect(String(url)).toBe('https://model.example/api/plan/v3/chat/completions')
  expect(init.credentials).toBe('omit')
  expect(init.redirect).toBe('error')
  expect(init.headers.get('Authorization')).toBe('Bearer test-secret')
  expect(init.headers.has('X-API-Key')).toBe(false)
  expect(JSON.parse(init.body)).toMatchObject({
    model: 'draft-model',
    stream: false,
    thinking: { type: 'disabled' },
  })
})
it('uses provider defaults and an inherited model when the binding has no model override', async () => {
  const fetch = mockResponse({ choices: [{ message: { content: 'OK' } }] })
  await test({ provider: 'openai', model: null, api_base: null }, 'vlm', {
    model: 'inherited',
  })
  expect(String(fetch.mock.calls[0][0])).toBe(
    'https://api.openai.com/v1/chat/completions',
  )
  expect(JSON.parse(fetch.mock.calls[0][1].body).model).toBe('inherited')
})
it('constructs Azure deployments and API-key authentication', async () => {
  const fetch = mockResponse({ choices: [{ message: { content: 'OK' } }] })
  await test({
    provider: 'azure',
    model: 'deployment',
    api_base: 'https://azure.example',
    api_version: '2025-01-01-preview',
  })
  expect(String(fetch.mock.calls[0][0])).toBe(
    'https://azure.example/openai/deployments/deployment/chat/completions?api-version=2025-01-01-preview',
  )
  expect(fetch.mock.calls[0][1].headers.get('api-key')).toBe('test-secret')
  expect(fetch.mock.calls[0][1].headers.has('Authorization')).toBe(false)
})
it('sends OpenAI embedding requests and validates vector results', async () => {
  const fetch = mockResponse({ data: [{ embedding: [0.1, 0.2] }] })
  await test({ provider: 'openai' }, 'embedding')
  expect(String(fetch.mock.calls[0][0])).toBe(
    'https://model.example/api/plan/v3/embeddings',
  )
  expect(JSON.parse(fetch.mock.calls[0][1].body).input).toBe('connection test')
})
it.each([
  {},
  { choices: [{ message: { content: {} } }] },
  { choices: [{ message: { content: '' } }] },
])(
  'rejects successful HTTP responses without a real completion: %j',
  async (body) => {
    mockResponse(body)
    await expect(test()).rejects.toMatchObject({ code: 'invalidResponse' })
  },
)
it('redacts credentials in provider error messages', async () => {
  mockResponse({ error: { message: 'Bad test-secret and custom-secret' } }, 401)
  await expect(
    test({ extra_headers: { 'x-secret': 'custom-secret' } }),
  ).rejects.toMatchObject({
    code: 'http',
    detail: 'HTTP 401: Bad [redacted] and [redacted]',
  })
})
it('does not label a CORS/network error as invalid credentials', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockRejectedValue(new TypeError('Failed to fetch')),
  )
  await expect(test()).rejects.toMatchObject({ code: 'network' })
})
it.each([
  [{ provider: 'openai-codex' }, 'unsupported'],
  [{ api_key: null }, 'missing'],
  [{ api_base: 'https://user:password@example.com' }, 'invalidUrl'],
])(
  'rejects unsupported or incomplete configurations before sending credentials: %j',
  async (config, code) => {
    const fetch = mockResponse({})
    await expect(test(config)).rejects.toMatchObject({ code })
    expect(fetch).not.toHaveBeenCalled()
  },
)

it.each([{}, { 'x-model-key': 'custom-auth' }])(
  'tests keyless OpenAI embeddings at a custom base with extra headers %j',
  async (extra_headers) => {
    const fetch = mockResponse({ data: [{ embedding: [0.1, 0.2] }] })
    await test(
      { provider: 'openai', api_key: null, extra_headers },
      'embedding',
    )
    const [url, init] = fetch.mock.calls[0]
    expect(String(url)).toBe('https://model.example/api/plan/v3/embeddings')
    expect(init.headers.has('Authorization')).toBe(false)
    for (const [key, value] of Object.entries(extra_headers))
      expect(init.headers.get(key)).toBe(value)
  },
)
it('accepts a keyless embedding base inherited from the shared config', async () => {
  const fetch = mockResponse({ data: [{ embedding: [0.1] }] })
  await test(
    { provider: 'openai', api_key: null, api_base: null },
    'embedding',
    { api_base: 'http://localhost:8080/v1' },
  )
  expect(String(fetch.mock.calls[0][0])).toBe(
    'http://localhost:8080/v1/embeddings',
  )
})
it.each(['vlm', 'embedding'] as const)(
  'still requires a key for default OpenAI %s endpoints',
  async (kind) => {
    const fetch = mockResponse({})
    await expect(
      test({ provider: 'openai', api_key: null, api_base: null }, kind),
    ).rejects.toMatchObject({ code: 'missing' })
    expect(fetch).not.toHaveBeenCalled()
  },
)
it.each(['o1', 'o3', 'o4-mini', 'gpt-5', 'gpt-6-sol', 'gpt-10'])(
  'adapts inherited max_tokens for reasoning model %s',
  async (model) => {
    const fetch = mockResponse({ choices: [{ message: { content: 'OK' } }] })
    await test({ provider: 'openai', model }, 'vlm', { max_tokens: 4096 })
    const body = JSON.parse(fetch.mock.calls[0][1].body)
    expect(body.max_completion_tokens).toBe(4096)
    expect(body).not.toHaveProperty('max_tokens')
  },
)
it('uses the credential token limit over the shared limit for Azure reasoning deployments', async () => {
  const fetch = mockResponse({ choices: [{ message: { content: 'OK' } }] })
  await test({ provider: 'azure', model: 'o3', max_tokens: 512 }, 'vlm', {
    max_tokens: 4096,
  })
  expect(JSON.parse(fetch.mock.calls[0][1].body)).toMatchObject({
    max_completion_tokens: 512,
  })
})
it.each([
  ['openai', 'gpt-4o'],
  ['volcengine', 'doubao-seed-evolving'],
  ['litellm', 'gpt-5'],
])('preserves max_tokens for %s/%s', async (provider, model) => {
  const fetch = mockResponse({ choices: [{ message: { content: 'OK' } }] })
  await test({ provider, model, max_tokens: 512 })
  const body = JSON.parse(fetch.mock.calls[0][1].body)
  expect(body.max_tokens).toBe(512)
  expect(body).not.toHaveProperty('max_completion_tokens')
})
