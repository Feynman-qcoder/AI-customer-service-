import { createRouter, createWebHistory } from 'vue-router'
import { currentUser, type UserRole } from './auth'

const CustomerChat = () => import('./views/CustomerChat.vue')
const AdminPanel = () => import('./views/AdminPanel.vue')
const LoginView = () => import('./views/LoginView.vue')

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/user/login' },
    { path: '/user/login', component: LoginView, meta: { loginRole: 'customer' } },
    { path: '/admin/login', component: LoginView, meta: { loginRole: 'admin' } },
    { path: '/chat', component: CustomerChat, meta: { role: 'customer' } },
    { path: '/admin', component: AdminPanel, meta: { role: 'admin' } }
  ]
})

router.beforeEach((to) => {
  const requiredRole = to.meta.role as UserRole | undefined
  if (!requiredRole) return true

  const user = currentUser()
  if (user?.role === requiredRole) return true

  return requiredRole === 'admin' ? '/admin/login' : '/user/login'
})

export default router
