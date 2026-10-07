<template>
  <section class="chat-page">
    <div class="page-head">
      <div>
        <h1 class="page-title">客服咨询</h1>
        <p>可咨询发货、物流、退款、退换货、商品售后和账号问题。涉及订单时，系统会优先查询订单数据。</p>
      </div>
      <el-button
        type="warning"
        plain
        :icon="Service"
        :disabled="!conversationId || sending"
        @click="ticketVisible = true"
      >
        转人工
      </el-button>
    </div>

    <div class="chat-layout">
      <div class="panel conversation">
        <div class="history-toolbar">
          <div class="history-picker">
            <span class="history-label">历史会话</span>
            <el-select
              :model-value="conversationId"
              aria-label="历史会话"
              placeholder="选择历史会话"
              :loading="historyLoading"
              :disabled="sending || historyLoading || switchingConversation"
              @change="switchConversation"
            >
              <el-option
                v-for="conversation in conversationHistory"
                :key="conversation.id"
                :value="conversation.id"
                :label="conversation.title"
              >
                <div class="history-option">
                  <span>{{ conversation.title }}</span>
                  <small>{{ formatDate(conversation.lastMessageAt ?? conversation.createdAt) }} · {{ conversation.messageCount }} 条消息</small>
                </div>
              </el-option>
            </el-select>
            <small v-if="currentConversationSummary" class="history-meta">
              {{ formatDate(currentConversationSummary.lastMessageAt ?? currentConversationSummary.createdAt) }} ·
              {{ currentConversationSummary.messageCount }} 条消息
            </small>
            <small v-else-if="historyLoadFailed" class="history-error">历史会话加载失败，请刷新后重试。</small>
          </div>
          <el-button
            type="primary"
            plain
            :disabled="sending || historyLoading || switchingConversation"
            @click="createNewConversation"
          >
            新对话
          </el-button>
        </div>

        <div ref="messageListRef" class="conversation-scroll" aria-live="polite">
          <div class="welcome">您好，请直接描述问题。可以说“最近订单”“第二个订单”或带上商品名，我会先帮您查订单和物流。</div>
          <div class="quick">
            <el-button
              v-for="item in quickQuestions"
              :key="item"
              size="small"
              round
              :disabled="!conversationId || conversationBusy"
              @click="ask(item)"
            >
              {{ item }}
            </el-button>
          </div>

          <div class="messages">
            <div v-if="!messages.length" class="empty-chat">
              <strong>可以这样问：</strong>
              <span>“我刚买的杯子什么时候发货？”</span>
              <span>“第二个订单物流到哪了？”</span>
              <span>“这个拆封后还能退吗？”</span>
            </div>
            <div v-for="message in messages" :key="message.id" class="message" :class="message.role.toLowerCase()">
              <div class="role">{{ message.role === 'USER' ? '我' : '客服' }}</div>
              <div class="bubble">
                <div>{{ message.content }}</div>
                <div v-if="message.sources?.length" class="message-sources">
                  <button
                    v-for="(source, index) in message.sources"
                    :key="`${message.id}-${source.documentId}-${source.fileName}-${index}`"
                    type="button"
                    class="inline-source-link"
                    @click="openSource(source)"
                  >
                    引用：{{ source.fileName }}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="input-row">
          <el-input
            v-model="question"
            aria-label="输入客服问题"
            placeholder="例如：我刚买的杯子什么时候发货？"
            :disabled="!conversationId || conversationBusy"
            @keyup.enter="ask(question)"
          />
          <el-button
            type="primary"
            :icon="Promotion"
            :loading="sending"
            :disabled="!conversationId || conversationBusy || !question.trim()"
            @click="ask(question)"
          >
            发送
          </el-button>
          <el-button :icon="Delete" :disabled="!conversationId || conversationBusy" @click="clearConversation">清空</el-button>
        </div>
      </div>

      <aside class="panel side">
        <div class="side-scroll">
          <div class="side-switch" role="tablist" aria-label="右侧业务面板">
            <button
              class="switch-button"
              :class="{ active: sideTab === 'products' }"
              type="button"
              role="tab"
              :aria-selected="sideTab === 'products'"
              @click="sideTab = 'products'"
            >
              下单商品
              <span>{{ products.length }}</span>
            </button>
            <button
              class="switch-button"
              :class="{ active: sideTab === 'orders' }"
              type="button"
              role="tab"
              :aria-selected="sideTab === 'orders'"
              @click="sideTab = 'orders'"
            >
              已下单
              <span>{{ orders.length }}</span>
            </button>
          </div>

          <template v-if="sideTab === 'products'">
            <div class="side-section-head">
              <div>
                <div class="side-title">可下单商品</div>
                <p>选择商品后可模拟创建订单。</p>
              </div>
              <el-button size="small" plain @click="loadProducts">刷新</el-button>
            </div>
            <div v-if="!products.length" class="empty-source">暂无商品。</div>
            <div v-for="product in products" :key="product.id" class="product-item business-card">
              <div class="ticket-row">
                <strong>{{ product.productName }}</strong>
                <el-tag size="small" :type="product.saleStatus === 'ON_SALE' ? 'success' : 'info'">{{ productStatusLabel(product.saleStatus) }}</el-tag>
              </div>
              <p>¥{{ product.price }}，库存 {{ product.stockQuantity }}</p>
              <small>{{ product.dispatchRule }}</small>
              <el-button class="mini-action" size="small" type="primary" :icon="ShoppingCart" @click="openOrderDialog(product)">
                模拟下单
              </el-button>
            </div>
          </template>

          <template v-else>
            <div class="side-section-head">
              <div>
                <div class="side-title">已下单商品</div>
                <p>{{ orderSummary }}</p>
              </div>
              <el-button size="small" plain @click="loadOrders">刷新</el-button>
            </div>
            <div class="order-tools">
              <el-button size="small" plain @click="ask('查询我已经下单的商品')">问客服列出</el-button>
              <el-button size="small" plain @click="ask('最近订单物流到哪里了')">查最近物流</el-button>
            </div>
            <div v-if="!orders.length" class="empty-source">暂无订单，可以切到“下单商品”先模拟下单。</div>
            <div v-for="(order, index) in orders" :key="order.id" class="order-item business-card">
              <div class="ticket-row">
                <strong>{{ index + 1 }}. {{ order.product.productName }}</strong>
                <el-tag size="small" :type="orderStatusType(order.status)">{{ orderStatusLabel(order.status) }}</el-tag>
              </div>
              <p>{{ order.orderNo }}</p>
              <small>数量：{{ order.quantity }}，预计发货：{{ formatDate(order.expectedShipAt) }}</small>
              <small v-if="order.shipmentEvents.length">最新物流：{{ order.shipmentEvents[0].eventNote }}</small>
              <div class="order-actions">
                <el-button size="small" type="primary" plain @click="ask(`订单 ${order.orderNo} 物流到哪里了`)">查物流</el-button>
                <el-button size="small" plain @click="ask(`订单 ${order.orderNo} 能不能退货`)">问售后</el-button>
              </div>
            </div>

            <div class="side-title ticket-title">我的工单</div>
            <div v-if="!tickets.length" class="empty-source">暂无工单。需要人工处理时可以点击“转人工”。</div>
            <div v-for="item in tickets" :key="item.id" class="ticket-item">
              <div class="ticket-row">
                <strong>{{ item.ticketNo }}</strong>
                <el-tag size="small" :type="statusType(item.status)">{{ statusLabel(item.status) }}</el-tag>
              </div>
              <p>{{ item.description }}</p>
              <small v-if="item.handlingNote">处理备注：{{ item.handlingNote }}</small>
            </div>
          </template>

          <div class="side-title source-title">引用资料</div>
          <div v-if="!lastSources.length" class="empty-source">客服回答引用知识库时会显示来源。</div>
          <button
            v-for="(source, index) in lastSources"
            :key="`${source.documentId}-${source.fileName}-${index}`"
            class="source-link"
            type="button"
            @click="openSource(source)"
          >
            <span>{{ source.fileName }}</span>
            <small>查看知识库片段</small>
          </button>

          <div class="side-title source-title">知识库资料</div>
          <div v-if="!knowledgeDocuments.length" class="empty-source">暂无可查看的知识库资料。</div>
          <button
            v-for="doc in knowledgeDocuments"
            :key="doc.id"
            class="source-link"
            type="button"
            @click="openDocumentSource(doc)"
          >
            <span>{{ doc.originalName }}</span>
            <small>{{ doc.fileType }}，{{ doc.chunkCount }} 个片段，{{ documentStatusLabel(doc.status) }}</small>
          </button>
        </div>
      </aside>
    </div>

    <el-dialog v-model="orderVisible" title="模拟下单" width="520px">
      <el-form label-width="92px">
        <el-form-item label="商品">
          <strong>{{ selectedProduct?.productName }}</strong>
        </el-form-item>
        <el-form-item label="数量">
          <el-input-number v-model="orderForm.quantity" :min="1" :max="20" />
        </el-form-item>
        <el-form-item label="收货人">
          <el-input v-model="orderForm.receiverName" />
        </el-form-item>
        <el-form-item label="手机号">
          <el-input v-model="orderForm.receiverPhone" />
        </el-form-item>
        <el-form-item label="地址">
          <el-input v-model="orderForm.receiverAddress" />
        </el-form-item>
        <el-form-item label="备注">
          <el-input v-model="orderForm.remark" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="orderVisible = false">取消</el-button>
        <el-button type="primary" :loading="orderSubmitting" @click="createOrder">确认下单</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="ticketVisible" title="创建人工工单" width="460px">
      <el-form label-width="90px">
        <el-form-item label="问题分类">
          <el-select v-model="ticket.category" aria-label="问题分类">
            <el-option label="售前" value="PRE_SALE" />
            <el-option label="物流" value="DELIVERY" />
            <el-option label="退货" value="RETURN" />
            <el-option label="退款" value="REFUND" />
            <el-option label="账号" value="ACCOUNT" />
            <el-option label="售后" value="AFTER_SALE" />
            <el-option label="其他" value="OTHER" />
          </el-select>
        </el-form-item>
        <el-form-item label="联系方式">
          <el-input v-model="ticket.contact" aria-label="联系方式" placeholder="手机号、邮箱或微信号" />
        </el-form-item>
        <el-form-item label="问题描述">
          <el-input v-model="ticket.description" aria-label="问题描述" type="textarea" :rows="4" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="ticketVisible = false">取消</el-button>
        <el-button type="primary" :loading="ticketSubmitting" @click="createTicket">提交</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="sourceVisible" title="知识库来源" width="560px">
      <div v-if="selectedSource" class="source-dialog">
        <div class="source-name">{{ selectedSource.fileName }}</div>
        <p>{{ selectedSource.snippet }}</p>
      </div>
      <template #footer>
        <el-button type="primary" @click="sourceVisible = false">知道了</el-button>
      </template>
    </el-dialog>
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Delete, Promotion, Service, ShoppingCart } from '@element-plus/icons-vue'
import { api, unwrap } from '../api'
import { currentUser } from '../auth'

interface MessageRow {
  id: number
  role: 'USER' | 'ASSISTANT' | 'SYSTEM'
  content: string
  sourcesJson?: string | null
  sources?: SourceReference[]
}

interface ConversationSummaryRow {
  id: number
  conversationNo: string
  title: string
  status: string
  createdAt: string
  updatedAt: string
  messageCount: number
  lastMessageAt?: string | null
}

interface PageResult<T> {
  page: number
  size: number
  total: number
  records: T[]
}

interface SourceReference {
  documentId: number
  fileName: string
  snippet: string
  score: number
}

interface KnowledgeDocumentRow {
  id: number
  originalName: string
  fileType: string
  status: string
  chunkCount: number
}

interface TicketRow {
  id: number
  ticketNo: string
  category: string
  status: string
  description: string
  handlingNote?: string
}

interface ProductRow {
  id: number
  productCode: string
  productName: string
  saleStatus: string
  price: string
  stockQuantity: number
  dispatchRule: string
}

interface ShipmentEvent {
  eventNote: string
  eventTime: string
  trackingNo?: string
}

interface OrderRow {
  id: number
  orderNo: string
  product: ProductRow
  quantity: number
  status: string
  expectedShipAt?: string
  shipmentEvents: ShipmentEvent[]
}

const quickQuestions = [
  '我刚买的杯子什么时候发货？',
  '第二个订单物流到哪了？',
  '这个拆封后还能退吗？',
  '暖风杯 H100 还有库存吗？',
  '收到破损商品怎么办？'
]
const question = ref('')
const conversationId = ref<number | null>(null)
const conversationHistory = ref<ConversationSummaryRow[]>([])
const messages = ref<MessageRow[]>([])
const tickets = ref<TicketRow[]>([])
const products = ref<ProductRow[]>([])
const orders = ref<OrderRow[]>([])
const knowledgeDocuments = ref<KnowledgeDocumentRow[]>([])
const sideTab = ref<'products' | 'orders'>('products')
const lastSources = ref<SourceReference[]>([])
const sending = ref(false)
const historyLoading = ref(false)
const historyLoadFailed = ref(false)
const switchingConversation = ref(false)
const ticketSubmitting = ref(false)
const orderSubmitting = ref(false)
const ticketVisible = ref(false)
const orderVisible = ref(false)
const selectedProduct = ref<ProductRow | null>(null)
const selectedSource = ref<SourceReference | null>(null)
const sourceVisible = ref(false)
const messageListRef = ref<HTMLElement | null>(null)
let messageRequestSequence = 0
const activeUserId = currentUser()?.userId ?? null
const conversationPreferenceKey = activeUserId
  ? `smart-customer-service-conversation-preference-${activeUserId}`
  : null
const ticket = reactive({ category: 'OTHER', contact: '', description: '' })
const orderForm = reactive({
  quantity: 1,
  receiverName: '张同学',
  receiverPhone: '13800000001',
  receiverAddress: '上海市浦东新区演示路 100 号',
  remark: ''
})

const orderSummary = computed(() => {
  if (!orders.value.length) return '还没有已下单商品。'
  const waiting = orders.value.filter((order) => order.status === 'WAITING_SHIPMENT' || order.status === 'PAID').length
  const shipping = orders.value.filter((order) => order.status === 'SHIPPED' || order.status === 'IN_TRANSIT').length
  return `${orders.value.length} 个订单，${waiting} 个待发货，${shipping} 个配送中。`
})

const currentConversationSummary = computed(() =>
  conversationHistory.value.find((conversation) => conversation.id === conversationId.value) ?? null
)

const conversationBusy = computed(() =>
  sending.value || historyLoading.value || switchingConversation.value || historyLoadFailed.value
)

onMounted(async () => {
  await initializeConversationHistory()
  await Promise.all([loadTickets(), loadOrders(), loadProducts(), loadKnowledgeDocuments()])
})

async function initializeConversationHistory() {
  historyLoading.value = true
  historyLoadFailed.value = false
  try {
    await refreshConversationHistory()
  } catch (error) {
    historyLoadFailed.value = true
    ElMessage.error(error instanceof Error ? error.message : '历史会话加载失败')
  } finally {
    historyLoading.value = false
  }

  if (historyLoadFailed.value) return
  if (!conversationHistory.value.length) {
    await createNewConversation()
    return
  }

  const preferredId = preferredConversationId()
  const selected = conversationHistory.value.find((conversation) => conversation.id === preferredId)
    ?? conversationHistory.value.find((conversation) => conversation.messageCount > 0)
    ?? conversationHistory.value[0]
  await selectConversation(selected.id)
}

async function refreshConversationHistory() {
  const data = await unwrap<PageResult<ConversationSummaryRow>>(
    api.get('/conversations', { params: { page: 1, size: 100 } })
  )
  conversationHistory.value = data.records
}

function preferredConversationId(): number | null {
  if (!conversationPreferenceKey) return null
  const raw = localStorage.getItem(conversationPreferenceKey)
  if (!raw || !/^\d+$/.test(raw)) return null
  const value = Number.parseInt(raw, 10)
  return Number.isSafeInteger(value) && value > 0 ? value : null
}

function saveConversationPreference(value: number) {
  if (!conversationPreferenceKey) return
  localStorage.setItem(conversationPreferenceKey, String(value))
}

function resetConversationTransientState() {
  messages.value = []
  lastSources.value = []
  selectedSource.value = null
  sourceVisible.value = false
}

async function selectConversation(targetId: number) {
  if (sending.value) return
  if (!conversationHistory.value.some((conversation) => conversation.id === targetId)) {
    ElMessage.error('会话不存在或不属于当前账号')
    return
  }

  conversationId.value = targetId
  saveConversationPreference(targetId)
  resetConversationTransientState()
  switchingConversation.value = true
  try {
    await reloadMessages()
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '会话消息加载失败')
  } finally {
    switchingConversation.value = false
  }
}

async function switchConversation(value: number | string) {
  if (sending.value) return
  const targetId = typeof value === 'number' ? value : Number.parseInt(value, 10)
  if (!Number.isSafeInteger(targetId) || targetId <= 0 || targetId === conversationId.value) return
  await selectConversation(targetId)
}

async function createNewConversation() {
  if (sending.value || historyLoading.value || switchingConversation.value) return
  historyLoading.value = true
  try {
    const created = await unwrap<{ id: number }>(api.post('/conversations', { title: '用户客服会话' }))
    await refreshConversationHistory()
    const confirmed = conversationHistory.value.find((conversation) => conversation.id === created.id)
    if (!confirmed) throw new Error('新会话未出现在当前账号的历史列表中')
    await selectConversation(confirmed.id)
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '新会话创建失败')
  } finally {
    historyLoading.value = false
  }
}

async function reloadMessages() {
  const targetId = conversationId.value
  if (!targetId) return
  const requestSequence = ++messageRequestSequence
  const rows = await unwrap<MessageRow[]>(api.get(`/conversations/${targetId}/messages`))
  if (requestSequence !== messageRequestSequence || conversationId.value !== targetId) return
  messages.value = rows.map((row) => ({ ...row, sources: parseMessageSources(row.sourcesJson) }))
  await nextTick()
  messageListRef.value?.scrollTo({ top: messageListRef.value.scrollHeight, behavior: 'smooth' })
}

async function ask(text: string) {
  const content = text.trim()
  if (!content || !conversationId.value || sending.value) return
  sending.value = true
  messageRequestSequence += 1
  try {
    const data = await unwrap<{
      conversationId: number
      sources: SourceReference[]
      confidenceLevel: string
      needHuman: boolean
    }>(api.post('/chat', { conversationId: conversationId.value, question: content }))
    conversationId.value = data.conversationId
    saveConversationPreference(data.conversationId)
    lastSources.value = data.sources
    question.value = ''
    ticket.description = content
    await reloadMessages()
    await Promise.all([loadOrders(), refreshConversationHistory()])
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '发送失败')
  } finally {
    sending.value = false
  }
}

async function clearConversation() {
  if (!conversationId.value) return
  messageRequestSequence += 1
  await unwrap(api.delete(`/conversations/${conversationId.value}/messages`))
  resetConversationTransientState()
  await refreshConversationHistory()
}

function openSource(source: SourceReference) {
  selectedSource.value = source
  sourceVisible.value = true
}

async function openDocumentSource(document: KnowledgeDocumentRow) {
  try {
    const detail = await unwrap<KnowledgeDocumentRow & { content: string }>(api.get(`/knowledge/documents/${document.id}`))
    selectedSource.value = {
      documentId: document.id,
      fileName: document.originalName,
      snippet: detail.content || '该文档暂未生成可展示片段。',
      score: 0
    }
    sourceVisible.value = true
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '知识库资料加载失败')
  }
}

function parseMessageSources(value?: string | null): SourceReference[] {
  if (!value) return []
  try {
    const parsed = JSON.parse(value) as { sources?: SourceReference[] }
    return Array.isArray(parsed.sources) ? parsed.sources : []
  } catch {
    return []
  }
}

function openOrderDialog(product: ProductRow) {
  selectedProduct.value = product
  orderForm.quantity = 1
  orderVisible.value = true
}

async function createOrder() {
  if (!selectedProduct.value) return
  orderSubmitting.value = true
  try {
    await unwrap(api.post('/orders', {
      productId: selectedProduct.value.id,
      ...orderForm
    }))
    ElMessage.success('订单已创建，可以直接询问发货和物流')
    orderVisible.value = false
    await Promise.all([loadOrders(), loadProducts()])
    sideTab.value = 'orders'
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '下单失败')
  } finally {
    orderSubmitting.value = false
  }
}

async function createTicket() {
  if (!conversationId.value) return
  if (!ticket.description.trim()) {
    ElMessage.warning('请先填写问题描述')
    return
  }
  ticketSubmitting.value = true
  try {
    await unwrap(api.post('/tickets', { conversationId: conversationId.value, ...ticket }))
    ElMessage.success('工单已创建')
    ticketVisible.value = false
    await loadTickets()
  } finally {
    ticketSubmitting.value = false
  }
}

async function loadTickets() {
  const data = await unwrap<{ records: TicketRow[] }>(api.get('/tickets'))
  tickets.value = data.records
}

async function loadOrders() {
  const data = await unwrap<{ records: OrderRow[] }>(api.get('/orders', { params: { page: 1, size: 50 } }))
  orders.value = data.records
}

async function loadProducts() {
  const data = await unwrap<{ records: ProductRow[] }>(api.get('/products', { params: { page: 1, size: 50 } }))
  products.value = data.records
}

async function loadKnowledgeDocuments() {
  knowledgeDocuments.value = await unwrap<KnowledgeDocumentRow[]>(api.get('/knowledge/documents'))
}

function statusLabel(status: string) {
  return {
    OPEN: '待处理',
    PROCESSING: '处理中',
    RESOLVED: '已解决',
    CLOSED: '已关闭'
  }[status] ?? status
}

function statusType(status: string) {
  if (status === 'RESOLVED') return 'success'
  if (status === 'PROCESSING') return 'warning'
  if (status === 'CLOSED') return 'info'
  return 'primary'
}

function productStatusLabel(status: string) {
  return {
    ON_SALE: '在售',
    OUT_OF_STOCK: '缺货',
    OFF_SHELF: '下架'
  }[status] ?? status
}

function documentStatusLabel(status: string) {
  return {
    READY: '已就绪',
    COMPLETED: '已完成',
    PROCESSING: '处理中',
    PENDING: '待处理',
    FAILED: '失败'
  }[status] ?? status
}

function orderStatusLabel(status: string) {
  return {
    PENDING_PAYMENT: '待付款',
    PAID: '已付款',
    WAITING_SHIPMENT: '待发货',
    SHIPPED: '已发货',
    IN_TRANSIT: '运输中',
    SIGNED: '已签收',
    REFUNDING: '退款中',
    REFUNDED: '已退款',
    CANCELLED: '已取消'
  }[status] ?? status
}

function orderStatusType(status: string) {
  if (status === 'SIGNED') return 'success'
  if (status === 'IN_TRANSIT' || status === 'SHIPPED') return 'warning'
  if (status === 'CANCELLED' || status === 'REFUNDED') return 'info'
  return 'primary'
}

function formatDate(value?: string) {
  return value ? value.replace('T', ' ').slice(0, 16) : '暂未同步'
}
</script>

<style scoped>
.chat-page {
  flex: 1 1 auto;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.page-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 14px;
  flex: 0 0 auto;
}

.page-head p {
  margin: 6px 0 0;
  color: #75685f;
}

.chat-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 280px;
  gap: 12px;
  flex: 1 1 auto;
  min-height: 0;
}

.welcome {
  margin-bottom: 10px;
  color: #6b5f58;
}

.quick {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 10px;
}

.conversation {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.history-toolbar {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
  flex: 0 0 auto;
}

.history-picker {
  min-width: 0;
  display: grid;
  grid-template-columns: auto minmax(220px, 380px);
  align-items: center;
  gap: 4px 10px;
}

.history-label {
  color: #6b5f58;
  font-size: 13px;
  font-weight: 600;
}

.history-picker .el-select {
  width: 100%;
}

.history-meta,
.history-error {
  grid-column: 2;
  color: #8b7d72;
  line-height: 1.4;
}

.history-error {
  color: #b6473a;
}

.history-option {
  min-width: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.history-option span {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}

.history-option small {
  flex: 0 0 auto;
  color: #8b7d72;
}

.conversation-scroll {
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
  border: 1px solid #efe8de;
  border-radius: 8px;
  padding: 14px;
  background: #fffaf5;
}

.messages {
  min-height: 100%;
}

.empty-chat {
  display: grid;
  gap: 8px;
  color: #75685f;
}

.message {
  margin-bottom: 14px;
}

.role {
  font-size: 12px;
  color: #8b7d72;
  margin-bottom: 4px;
}

.bubble {
  display: inline-block;
  max-width: min(84%, 820px);
  padding: 10px 12px;
  border-radius: 8px;
  background: #fff;
  border: 1px solid #eee2d6;
  line-height: 1.65;
  white-space: pre-wrap;
  text-align: left;
}

.message.user {
  text-align: right;
}

.message.user .bubble {
  background: #ffe9e0;
}

.input-row {
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 10px;
  margin-top: 12px;
  flex: 0 0 auto;
}

.side {
  min-height: 0;
  overflow: hidden;
  padding: 0;
}

.side-scroll {
  height: 100%;
  overflow-y: auto;
  overflow-x: clip;
  padding: 18px;
}

.side-scroll * {
  max-width: 100%;
}

.side-switch {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
  margin-bottom: 14px;
  padding: 3px;
  background: #f7eee7;
  border: 1px solid #eadbd0;
  border-radius: 8px;
}

.switch-button {
  border: 1px solid transparent;
  border-radius: 6px;
  padding: 7px 8px;
  background: transparent;
  color: #75685f;
  font-weight: 600;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.switch-button span {
  min-width: 20px;
  height: 20px;
  padding: 0 6px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #f1e6dc;
  color: #8a7669;
  font-size: 12px;
}

.switch-button.active {
  border-color: #eadbd0;
  background: #fffaf5;
  color: #5c463b;
  box-shadow: none;
}

.switch-button.active span {
  background: #f4ded4;
  color: #7a4d3d;
}

.side-section-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}

.side-section-head p {
  margin: 3px 0 0;
  color: #8b7d72;
  font-size: 13px;
  line-height: 1.5;
}

.side-title,
.metric {
  margin-bottom: 10px;
  font-weight: 600;
}

.source-title,
.ticket-title {
  margin-top: 18px;
}

.empty-source {
  color: #8b7d72;
  font-size: 14px;
  line-height: 1.6;
}

.source-item,
.ticket-item,
.order-item,
.product-item {
  padding: 12px 0;
  border-bottom: 1px solid #f0e5dc;
}

.source-link {
  width: 100%;
  border: 1px solid #eadbd0;
  border-radius: 8px;
  background: #fffdfb;
  color: #5c463b;
  padding: 10px 12px;
  margin-bottom: 8px;
  text-align: left;
  cursor: pointer;
  display: grid;
  gap: 4px;
}

.source-link:hover {
  border-color: #d9b9a8;
  background: #fff7f1;
}

.source-link span {
  font-weight: 600;
  overflow-wrap: anywhere;
}

.source-link small {
  color: #8b7d72;
}

.message-sources {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 10px;
}

.inline-source-link {
  border: 1px solid #eadbd0;
  border-radius: 999px;
  background: #fffdfb;
  color: #7a4a35;
  cursor: pointer;
  font-size: 12px;
  line-height: 1.3;
  padding: 5px 9px;
}

.inline-source-link:hover {
  border-color: #d8aa93;
  background: #fff7f1;
}

.business-card {
  padding: 12px;
  border: 1px solid #f0e5dc;
  border-radius: 8px;
  background: #fffdfb;
  margin-bottom: 10px;
}

.source-name {
  font-weight: 600;
  margin-bottom: 6px;
}

.source-item p,
.source-dialog p,
.ticket-item p,
.order-item p,
.product-item p {
  margin: 0;
  color: #6b5f58;
  line-height: 1.6;
}

.ticket-row {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  align-items: center;
  margin-bottom: 6px;
  min-width: 0;
}

.ticket-row strong {
  min-width: 0;
  overflow-wrap: anywhere;
}

.ticket-item small,
.order-item small,
.product-item small {
  display: block;
  color: #8b7d72;
  margin-top: 6px;
}

.mini-action {
  margin-top: 8px;
}

.order-tools {
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
  margin-bottom: 12px;
}

.order-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 10px;
}

@media (max-width: 640px) {
  .chat-page {
    height: auto;
    min-height: 0;
    overflow: visible;
  }

  .chat-layout {
    display: grid;
  }

  .messages,
  .side-scroll {
    max-height: 560px;
  }

  .page-head {
    display: grid;
  }

  .history-toolbar {
    display: grid;
  }

  .history-picker {
    grid-template-columns: 1fr;
  }

  .history-meta,
  .history-error {
    grid-column: 1;
  }

  .input-row {
    grid-template-columns: 1fr 1fr;
  }

  .input-row .el-input {
    grid-column: 1 / -1;
  }

  .bubble {
    max-width: 92%;
  }
}

@media (max-width: 420px) {
  .quick .el-button {
    width: 100%;
    justify-content: flex-start;
  }
}
</style>
