<template>
  <el-container class="shell">
    <el-aside
      v-if="user && !isLoginRoute && !effectiveAsideHidden"
      width="224px"
      class="aside"
    >
      <div class="brand">
        <span class="brand-mark">智能</span>
        <span class="brand-text">AI客服</span>
      </div>
      <el-menu
        router
        :default-active="$route.path"
        class="menu"
      >
        <el-menu-item v-if="user.role === 'customer'" index="/chat">
          <el-icon><ChatDotRound /></el-icon>
          <template #title>客服咨询</template>
        </el-menu-item>
        <el-menu-item v-if="user.role === 'admin'" index="/admin">
          <el-icon><DataBoard /></el-icon>
          <template #title>管理后台</template>
        </el-menu-item>
      </el-menu>
      <div class="health" :class="{ ok: healthOk }">
        <span class="health-dot" aria-hidden="true"></span>
        <span class="health-label">后端：{{ healthText }}</span>
      </div>
    </el-aside>
    <el-main class="main" :class="{ centered: !user || isLoginRoute, 'chat-main': isChatRoute }">
      <div v-if="user && !isLoginRoute" class="topbar">
        <el-button
          class="nav-toggle"
          plain
          :icon="effectiveAsideHidden ? Expand : Fold"
          @click="isAsideHidden = !isAsideHidden"
        >
          {{ effectiveAsideHidden ? '展开导航' : '收起导航' }}
        </el-button>
        <div class="topbar-actions">
          <div class="topbar-user">
            <strong>{{ user.name }}</strong>
            <span>{{ user.role === 'admin' ? '管理员账号' : '用户账号' }}</span>
          </div>
          <el-button type="danger" plain :icon="SwitchButton" @click="handleLogout">退出登录</el-button>
        </div>
      </div>
      <router-view />
    </el-main>
  </el-container>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ChatDotRound, DataBoard, Expand, Fold, SwitchButton } from '@element-plus/icons-vue'
import { api, unwrap } from './api'
import { currentUser, logout, sessionRefreshToken, type SessionUser } from './auth'

const route = useRoute()
const router = useRouter()
const user = ref<SessionUser | null>(currentUser())
const healthText = ref('检查中')
const healthOk = ref(false)
const isAsideHidden = ref(false)
const isCompactViewport = ref(false)
const isLoginRoute = computed(() => route.path === '/user/login' || route.path === '/admin/login')
const isChatRoute = computed(() => route.path === '/chat')
const effectiveAsideHidden = computed(() => isAsideHidden.value && !isCompactViewport.value)

watch(() => route.fullPath, () => {
  user.value = currentUser()
})

onMounted(async () => {
  updateViewportMode()
  window.addEventListener('resize', updateViewportMode)
  await checkHealth()
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', updateViewportMode)
})

function updateViewportMode() {
  isCompactViewport.value = window.innerWidth <= 920
}

async function checkHealth() {
  try {
    const data = await unwrap<Record<string, unknown>>(api.get('/health'))
    healthOk.value = data.status === 'UP'
    healthText.value = healthOk.value ? '已连接' : '异常'
  } catch {
    healthText.value = '未连接'
  }
}

async function handleLogout() {
  const role = user.value?.role
  try {
    await api.post('/auth/logout', { refreshToken: sessionRefreshToken() })
  } catch {
    // Local cleanup still matters if the token has already expired.
  }
  logout()
  user.value = null
  router.push(role === 'admin' ? '/admin/login' : '/user/login')
}
</script>
