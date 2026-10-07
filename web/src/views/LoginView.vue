<template>
  <section class="login-page">
    <div class="login-panel">
      <div class="login-copy">
        <p class="eyebrow">{{ isAdmin ? '管理后台' : '用户客服端' }}</p>
        <h1>{{ isAdmin ? '管理员登录' : '用户登录' }}</h1>
        <p class="hint">
          {{ isAdmin ? '进入文档管理、工单处理和客服运营后台。' : '登录后开始咨询商品、发货、退款和售后问题。' }}
        </p>
      </div>

      <el-form label-position="top" @submit.prevent>
        <el-form-item label="账号">
          <el-input v-model="username" autocomplete="username" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="password" type="password" show-password autocomplete="current-password" @keyup.enter="submit" />
        </el-form-item>
        <el-button type="primary" size="large" class="login-button" :icon="Right" :loading="loading" @click="submit">登录</el-button>
      </el-form>

      <div class="demo-account">
        <div class="demo-account-header">
          <span>演示账号</span>
          <el-button size="small" text type="primary" @click="fillDemoAccount">填入</el-button>
        </div>
        <div class="demo-account-row">
          <span>账号</span>
          <strong>{{ demoAccount.username }}</strong>
        </div>
        <div class="demo-account-row">
          <span>密码</span>
          <strong>{{ demoAccount.password }}</strong>
        </div>
      </div>
      <router-link class="switch-link" :to="isAdmin ? '/user/login' : '/admin/login'">
        {{ isAdmin ? '去用户客服端登录' : '去管理后台登录' }}
      </router-link>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Right } from '@element-plus/icons-vue'
import { api, unwrap, type AuthTokenResponse } from '../api'
import { saveSession, type UserRole } from '../auth'

const route = useRoute()
const router = useRouter()
const username = ref('')
const password = ref('')
const loading = ref(false)

const role = computed(() => (route.meta.loginRole as UserRole) ?? 'customer')
const isAdmin = computed(() => role.value === 'admin')
const demoAccount = computed(() => isAdmin.value
  ? { username: 'admin', password: 'local_admin_demo_only' }
  : { username: 'user', password: 'local_customer_demo_only' })

function fillDemoAccount() {
  username.value = demoAccount.value.username
  password.value = demoAccount.value.password
}

async function submit() {
  loading.value = true
  try {
    const data = await unwrap<AuthTokenResponse>(api.post('/auth/login', {
      username: username.value,
      password: password.value
    }))
    const actualRole: UserRole = data.user.role === 'ADMIN' ? 'admin' : 'customer'
    saveSession({
      token: data.accessToken,
      refreshToken: data.refreshToken,
      userId: data.user.userId,
      username: data.user.username,
      name: data.user.name,
      role: actualRole
    })
    router.push(actualRole === 'admin' ? '/admin' : '/chat')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '登录失败')
  } finally {
    loading.value = false
  }
}
</script>
