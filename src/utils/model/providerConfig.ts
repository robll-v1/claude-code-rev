import { getDefaultVertexRegion, getAWSRegion, isEnvTruthy } from '../envUtils.js'
import { getGitHubCopilotApiUrl } from '../github/copilotApi.js'
import { getGitHubAuthToken } from '../github/ghAuthToken.js'
import { getInitialSettings } from '../settings/settings.js'
import type {
  ProviderConfig,
  ProviderModelConfig,
} from '../settings/types.js'

export type ProviderModelTransport = 'chat-completions' | 'responses'

export type ResolvedProviderModelConfig = {
  id: string
  label: string
  description?: string
  transport?: ProviderModelTransport
  supportsTools?: boolean
  supportsStreaming?: boolean
}

export type ConfiguredProviderType =
  | 'firstParty'
  | 'bedrock'
  | 'vertex'
  | 'foundry'
  | 'anthropic-compatible'
  | 'openai-compatible'
  | 'github-models'
  | 'github-copilot'

export type ConfiguredProvider = {
  type?: ConfiguredProviderType
  name?: string
  baseURL?: string
  apiKeyEnv?: string
  authTokenEnv?: string
  defaultModel?: string
  models?: ProviderModelConfig[]
  smallFastModel?: string
  region?: string
  projectId?: string
  resource?: string
}

type ActiveProviderConfig = {
  id: string
  type: ConfiguredProviderType
  name: string
  baseURL?: string
  apiKeyEnv?: string
  authTokenEnv?: string
  defaultModel?: string
  models?: ProviderModelConfig[]
  smallFastModel?: string
  region?: string
  projectId?: string
  resource?: string
  isCustom: boolean
}

export type ResolvedProviderConfig = ActiveProviderConfig

type BuiltinProviderId =
  | 'firstParty'
  | 'bedrock'
  | 'vertex'
  | 'foundry'
  | 'github-models'
  | 'github-copilot'

const BUILTIN_PROVIDER_NAMES: Record<BuiltinProviderId, string> = {
  firstParty: 'Anthropic',
  bedrock: 'Amazon Bedrock',
  vertex: 'Google Vertex AI',
  foundry: 'Azure AI Foundry',
  'github-models': 'GitHub Models',
  'github-copilot': 'GitHub Copilot',
}

const BUILTIN_PROVIDER_IDS = new Set<BuiltinProviderId>([
  'firstParty',
  'bedrock',
  'vertex',
  'foundry',
  'github-models',
  'github-copilot',
])

const BUILTIN_PROVIDER_ORDER: BuiltinProviderId[] = [
  'firstParty',
  'github-copilot',
  'github-models',
  'bedrock',
  'vertex',
  'foundry',
]

const BUILTIN_PROVIDER_CONFIGS: Record<
  'github-models' | 'github-copilot',
  Pick<
    ActiveProviderConfig,
    'type' | 'baseURL' | 'authTokenEnv' | 'defaultModel' | 'models' | 'smallFastModel'
  >
> = {
  'github-models': {
    type: 'github-models',
    baseURL: 'https://models.github.ai/inference',
    authTokenEnv: 'GITHUB_MODELS_TOKEN',
    defaultModel: 'openai/gpt-4.1',
    models: [
      {
        id: 'openai/gpt-4.1',
        label: 'GPT-4.1',
        description: 'GitHub Models chat/completions model',
        transport: 'chat-completions',
      },
      {
        id: 'openai/gpt-4.1-mini',
        label: 'GPT-4.1 mini',
        description: 'GitHub Models chat/completions model',
        transport: 'chat-completions',
      },
      {
        id: 'anthropic/claude-3.7-sonnet',
        label: 'Claude 3.7 Sonnet',
        description: 'GitHub Models chat/completions model',
        transport: 'chat-completions',
        supportsTools: true,
      },
      {
        id: 'meta/Llama-3.3-70B-Instruct',
        label: 'Llama 3.3 70B Instruct',
        description: 'GitHub Models chat/completions model',
        transport: 'chat-completions',
      },
    ],
  },
  'github-copilot': {
    type: 'github-copilot',
    baseURL: 'https://api.githubcopilot.com',
    authTokenEnv: 'GITHUB_COPILOT_TOKEN',
    defaultModel: 'claude-sonnet-4.6',
    smallFastModel: 'claude-haiku-4.5',
    models: [
      {
        id: 'claude-sonnet-4.6',
        label: 'Claude Sonnet 4.6',
        description: 'Validated Copilot Claude model',
        transport: 'chat-completions',
        supportsTools: true,
        supportsStreaming: true,
      },
      {
        id: 'claude-opus-4.6',
        label: 'Claude Opus 4.6',
        description: 'Validated Copilot Claude model',
        transport: 'chat-completions',
        supportsTools: true,
        supportsStreaming: true,
      },
      {
        id: 'claude-haiku-4.5',
        label: 'Claude Haiku 4.5',
        description: 'Validated Copilot Claude model',
        transport: 'chat-completions',
        supportsTools: true,
        supportsStreaming: true,
      },
      {
        id: 'claude-sonnet-4.5',
        label: 'Claude Sonnet 4.5',
        description: 'Validated Copilot Claude model',
        transport: 'chat-completions',
        supportsTools: true,
        supportsStreaming: true,
      },
      {
        id: 'claude-opus-4.5',
        label: 'Claude Opus 4.5',
        description: 'Validated Copilot Claude model',
        transport: 'chat-completions',
        supportsTools: true,
        supportsStreaming: true,
      },
      {
        id: 'claude-sonnet-4',
        label: 'Claude Sonnet 4',
        description: 'Validated Copilot Claude model',
        transport: 'chat-completions',
        supportsTools: true,
        supportsStreaming: true,
      },
    ],
  },
}

function getProviderModelId(model: ProviderModelConfig): string | undefined {
  if (typeof model === 'string') {
    return model
  }

  return model.id ?? model.model ?? model.value ?? model.name
}

function normalizeProviderModel(
  model: ProviderModelConfig,
): ResolvedProviderModelConfig | undefined {
  const id = getProviderModelId(model)
  if (!id) {
    return undefined
  }

  if (typeof model === 'string') {
    return {
      id,
      label: id,
    }
  }

  return {
    id,
    label: model.label ?? id,
    description: model.description,
    transport: model.transport,
    supportsTools: model.supportsTools,
    supportsStreaming: model.supportsStreaming,
  }
}

function getEnvSelectedProviderId(): 'bedrock' | 'vertex' | 'foundry' | undefined {
  return isEnvTruthy(process.env.CLAUDE_CODE_USE_BEDROCK)
    ? 'bedrock'
    : isEnvTruthy(process.env.CLAUDE_CODE_USE_VERTEX)
      ? 'vertex'
      : isEnvTruthy(process.env.CLAUDE_CODE_USE_FOUNDRY)
        ? 'foundry'
        : undefined
}

export function getEnvironmentProviderOverrideId():
  | 'bedrock'
  | 'vertex'
  | 'foundry'
  | undefined {
  return getEnvSelectedProviderId()
}

export function isBuiltinProviderId(id: string): id is BuiltinProviderId {
  return BUILTIN_PROVIDER_IDS.has(id as BuiltinProviderId)
}

export function resolveConfiguredProviderType(
  id: string,
  provider: ProviderConfig | undefined,
): ConfiguredProviderType {
  return provider?.type ?? (isBuiltinProviderId(id) ? id : 'anthropic-compatible')
}

export function isOpenAICompatibleProviderType(
  type: ConfiguredProviderType,
): boolean {
  return (
    type === 'openai-compatible' ||
    type === 'github-models' ||
    type === 'github-copilot'
  )
}

export function getActiveProviderType(): ConfiguredProviderType {
  return getActiveProviderConfig().type
}

function getBuiltinProviderConfig(id: BuiltinProviderId): ActiveProviderConfig {
  if (id === 'bedrock') {
    return {
      id,
      type: 'bedrock',
      name: BUILTIN_PROVIDER_NAMES.bedrock,
      region: getAWSRegion(),
      isCustom: false,
    }
  }

  if (id === 'vertex') {
    return {
      id,
      type: 'vertex',
      name: BUILTIN_PROVIDER_NAMES.vertex,
      baseURL: process.env.ANTHROPIC_VERTEX_BASE_URL,
      region: getDefaultVertexRegion(),
      projectId: process.env.ANTHROPIC_VERTEX_PROJECT_ID,
      isCustom: false,
    }
  }

  if (id === 'foundry') {
    return {
      id,
      type: 'foundry',
      name: BUILTIN_PROVIDER_NAMES.foundry,
      baseURL: process.env.ANTHROPIC_FOUNDRY_BASE_URL,
      apiKeyEnv: 'ANTHROPIC_FOUNDRY_API_KEY',
      resource: process.env.ANTHROPIC_FOUNDRY_RESOURCE,
      isCustom: false,
    }
  }

  if (id === 'github-models' || id === 'github-copilot') {
    const config = BUILTIN_PROVIDER_CONFIGS[id]
    return {
      id,
      name: BUILTIN_PROVIDER_NAMES[id],
      ...config,
      ...(id === 'github-copilot'
        ? {
            baseURL: getGitHubCopilotApiUrl() ?? config.baseURL,
          }
        : {}),
      isCustom: false,
    }
  }

  return {
    id,
    type: 'firstParty',
    name: BUILTIN_PROVIDER_NAMES.firstParty,
    baseURL: process.env.ANTHROPIC_BASE_URL,
    apiKeyEnv: 'ANTHROPIC_API_KEY',
    authTokenEnv: 'ANTHROPIC_AUTH_TOKEN',
    isCustom: false,
  }
}

function toConfiguredProvider(providerId: string, provider: ProviderConfig): ConfiguredProvider {
  return {
    ...provider,
    type: resolveConfiguredProviderType(providerId, provider),
  }
}

export function getConfiguredProviders(): Record<string, ConfiguredProvider> {
  const settings = getInitialSettings()
  return Object.fromEntries(
    Object.entries(settings.providers ?? {}).map(([id, provider]) => [
      id,
      toConfiguredProvider(id, provider),
    ]),
  )
}

export function getConfiguredProviderId(): string | undefined {
  const envProviderId = getEnvSelectedProviderId()
  if (envProviderId) {
    return envProviderId
  }

  return getInitialSettings().provider
}

export function getStoredProviderId(): string | undefined {
  return getInitialSettings().provider
}

export function normalizeProviderSettingValue(
  providerId: string,
): string | undefined {
  return providerId === 'firstParty' ? undefined : providerId
}

export function getProviderConfigById(
  providerId: string | undefined,
): ActiveProviderConfig {
  if (!providerId) {
    return getBuiltinProviderConfig('firstParty')
  }

  const configuredProviders = getConfiguredProviders()

  if (isBuiltinProviderId(providerId)) {
    const builtin = getBuiltinProviderConfig(providerId)
    const configured = configuredProviders[providerId]
    return configured
      ? {
          ...builtin,
          ...configured,
          id: providerId,
          type: resolveConfiguredProviderType(providerId, configured),
          name: configured.name ?? builtin.name,
          isCustom: false,
        }
      : builtin
  }

  const configured = configuredProviders[providerId]
  if (!configured) {
    return getBuiltinProviderConfig('firstParty')
  }

  return {
    id: providerId,
    type: resolveConfiguredProviderType(providerId, configured),
    name: configured.name ?? providerId,
    baseURL: configured.baseURL,
    apiKeyEnv: configured.apiKeyEnv,
    authTokenEnv: configured.authTokenEnv,
    defaultModel: configured.defaultModel,
    models: configured.models,
    smallFastModel: configured.smallFastModel,
    region: configured.region,
    projectId: configured.projectId,
    resource: configured.resource,
    isCustom: true,
  }
}

export function getAllProviderConfigs(): ActiveProviderConfig[] {
  const customProviderIds = Object.keys(getConfiguredProviders())
    .filter(providerId => !isBuiltinProviderId(providerId))
    .sort((a, b) => a.localeCompare(b))

  return [
    ...BUILTIN_PROVIDER_ORDER.map(providerId => getProviderConfigById(providerId)),
    ...customProviderIds.map(providerId => getProviderConfigById(providerId)),
  ]
}

export function getSuggestedModelForProvider(
  providerId: string,
  currentModel: string | null | undefined,
): string | null {
  const provider = getProviderConfigById(providerId)

  if (
    currentModel !== undefined &&
    currentModel !== null &&
    (!provider.models ||
      provider.models.some(model => getProviderModelId(model) === currentModel))
  ) {
    return currentModel
  }

  return provider.defaultModel ?? null
}

export function getActiveProviderConfig(): ActiveProviderConfig {
  const envProviderId = getEnvSelectedProviderId()
  if (envProviderId) {
    return getBuiltinProviderConfig(envProviderId)
  }

  return getProviderConfigById(getInitialSettings().provider)
}

export function getProviderDisplayName(): string {
  return getActiveProviderConfig().name
}

export function getConfiguredAnthropicBaseUrl(): string | undefined {
  const provider = getActiveProviderConfig()
  return provider.type === 'firstParty' ||
    provider.type === 'anthropic-compatible'
    ? provider.baseURL
    : undefined
}

export function getConfiguredCustomModels(): ProviderModelConfig[] | undefined {
  const configuredModels = getActiveProviderConfig().models
  if (configuredModels && configuredModels.length > 0) {
    return configuredModels
  }

  return process.env.ANTHROPIC_CUSTOM_MODEL_OPTION
    ? [process.env.ANTHROPIC_CUSTOM_MODEL_OPTION]
    : undefined
}

export function getProviderModels(
  providerId?: string,
): ResolvedProviderModelConfig[] {
  const models = (providerId
    ? getProviderConfigById(providerId)
    : getActiveProviderConfig()
  ).models

  if (!models?.length) {
    return []
  }

  return models.flatMap(model => {
    const normalized = normalizeProviderModel(model)
    return normalized ? [normalized] : []
  })
}

export function getProviderModelConfig(
  modelId: string,
  providerId?: string,
): ResolvedProviderModelConfig | undefined {
  return getProviderModels(providerId).find(model => model.id === modelId)
}

export function getProviderModelConfigFromList(
  models: ProviderModelConfig[] | undefined,
  modelId: string,
): ResolvedProviderModelConfig | undefined {
  return (models ?? []).flatMap(model => {
    const normalized = normalizeProviderModel(model)
    return normalized ? [normalized] : []
  }).find(model => model.id === modelId)
}

export function getProviderModelTransportFromList(
  models: ProviderModelConfig[] | undefined,
  modelId: string,
  providerType: ConfiguredProviderType,
): ProviderModelTransport {
  const configuredTransport = getProviderModelConfigFromList(models, modelId)?.transport
  if (configuredTransport) {
    return configuredTransport
  }

  if (
    providerType === 'github-copilot' &&
    /^(gpt-5|gpt-5\.|o1|o3|o4)/i.test(modelId)
  ) {
    return 'responses'
  }

  return 'chat-completions'
}

export function getProviderModelTransport(
  modelId: string,
  providerId?: string,
  providerType?: ConfiguredProviderType,
): ProviderModelTransport {
  const effectiveProvider = providerId
    ? getProviderConfigById(providerId)
    : getActiveProviderConfig()
  return getProviderModelTransportFromList(
    effectiveProvider.models,
    modelId,
    providerType ?? effectiveProvider.type,
  )
}

export function getConfiguredProviderApiKey(): string | undefined {
  const apiKeyEnv = getActiveProviderConfig().apiKeyEnv
  return apiKeyEnv ? process.env[apiKeyEnv] : undefined
}

export function getConfiguredGitHubProviderAuthToken(
  type: ConfiguredProviderType,
): string | undefined {
  if (type === 'github-models') {
    return getGitHubAuthToken('github-models')
  }

  if (type === 'github-copilot') {
    return getGitHubAuthToken('github-copilot')
  }

  return undefined
}

export function getConfiguredProviderAuthToken(): string | undefined {
  const provider = getActiveProviderConfig()
  const authTokenEnv = provider.authTokenEnv
  const configuredToken = authTokenEnv ? process.env[authTokenEnv] : undefined
  if (configuredToken) {
    return configuredToken
  }

  return getConfiguredGitHubProviderAuthToken(provider.type)
}

export function getConfiguredProviderSecretEnvNames(): string[] {
  const provider = getActiveProviderConfig()
  const secretEnvNames = [provider.apiKeyEnv, provider.authTokenEnv]

  if (provider.type === 'github-models' || provider.type === 'github-copilot') {
    secretEnvNames.push('GH_TOKEN', 'GITHUB_TOKEN')
  }

  return secretEnvNames.filter(
    (value): value is string => typeof value === 'string' && value.length > 0,
  )
}
