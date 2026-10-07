import axios, { type AxiosError, type InternalAxiosRequestConfig } from 'axios'
import { currentUser, logout, saveAccessToken, sessionRefreshToken, sessionToken } from './auth'

export const api = axios.create({
  baseURL: '/api/v1',
  timeout: 20000,
  withCredentials: true
})

let refreshPromise: Promise<string | null> | null = null

api.interceptors.request.use((config) => {
  const token = sessionToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const original = error.config as (InternalAxiosRequestConfig & { _retried?: boolean }) | undefined
    const status = error.response?.status
    const url = original?.url ?? ''

    if (status !== 401 || !original || original._retried || url.includes('/auth/login') || url.includes('/auth/refresh')) {
      if (status === 401) logout()
      return Promise.reject(error)
    }

    original._retried = true
    refreshPromise ??= refreshAccessToken().finally(() => {
      refreshPromise = null
    })
    const token = await refreshPromise
    if (!token) {
      logout()
      return Promise.reject(error)
    }
    original.headers.Authorization = `Bearer ${token}`
    return api(original)
  }
)

async function refreshAccessToken(): Promise<string | null> {
  try {
    const refreshToken = sessionRefreshToken()
    if (!refreshToken) return null
    const data = await unwrap<AuthTokenResponse>(api.post('/auth/refresh', { refreshToken }))
    const existing = currentUser()
    if (!existing) return data.accessToken
    saveAccessToken(data.accessToken, data.refreshToken)
    return data.accessToken
  } catch {
    return null
  }
}

export interface ApiResponse<T> {
  success: boolean
  message: string
  data: T
  timestamp: string
}

export interface AuthTokenResponse {
  accessToken: string
  refreshToken?: string
  expiresIn: number
  user: {
    userId: number
    username: string
    name: string
    role: 'CUSTOMER' | 'ADMIN'
  }
}

export async function unwrap<T>(promise: Promise<{ data: ApiResponse<T> }>): Promise<T> {
  const response = await promise
  if (!response.data.success) {
    throw new Error(response.data.message)
  }
  return response.data.data
}
