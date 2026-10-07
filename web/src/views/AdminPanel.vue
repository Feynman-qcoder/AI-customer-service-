<template>
  <section>
    <h1 class="page-title">管理后台</h1>
    <el-tabs>
      <el-tab-pane label="数据概览">
        <div class="dashboard-grid">
          <div class="panel metric-card">
            <span>今日咨询</span>
            <strong>{{ dashboard.todayConsultations }}</strong>
          </div>
          <div class="panel metric-card">
            <span>待发货订单</span>
            <strong>{{ dashboard.waitingShipmentOrders }}</strong>
          </div>
          <div class="panel metric-card">
            <span>配送中订单</span>
            <strong>{{ dashboard.shippingOrders }}</strong>
          </div>
          <div class="panel metric-card">
            <span>待处理工单</span>
            <strong>{{ dashboard.pendingTickets }}</strong>
          </div>
          <div class="panel metric-card">
            <span>商品数</span>
            <strong>{{ dashboard.productCount }}</strong>
          </div>
          <div class="panel metric-card">
            <span>可用知识文档</span>
            <strong>{{ dashboard.readyDocuments }}</strong>
          </div>
        </div>
        <div class="panel dashboard-panel">
          <div class="toolbar">
            <strong>运营提醒</strong>
            <el-button :icon="Refresh" @click="loadDashboard">刷新</el-button>
          </div>
          <div class="notice-list">
            <span>有 {{ dashboard.waitingShipmentOrders }} 个订单等待仓库发货。</span>
            <span>有 {{ dashboard.pendingTickets }} 个工单还需要客服处理。</span>
            <span v-if="dashboard.failedDocuments">有 {{ dashboard.failedDocuments }} 个知识库文档处理失败，请检查失败原因。</span>
            <span v-else>知识库暂无失败文档。</span>
          </div>
        </div>
      </el-tab-pane>

      <el-tab-pane label="模型参数">
        <div class="panel">
          <div class="config-grid">
            <div class="config-item">
              <div>
                <strong>Temperature</strong>
                <p>控制回答发散程度。客服场景建议保持在 0.2 到 0.4。</p>
              </div>
              <el-slider v-model="modelConfig.temperature" :min="0" :max="1" :step="0.1" show-input />
            </div>
            <div class="config-item">
              <div>
                <strong>TopK</strong>
                <p>每次从知识库取多少条候选片段。数值越大，资料越多但干扰也可能增加。</p>
              </div>
              <el-input-number v-model="modelConfig.topK" :min="1" :max="20" />
            </div>
            <div class="config-item">
              <div>
                <strong>最低相似度阈值</strong>
                <p>控制低相关资料是否拒答。调高更保守，调低更容易回答但误答风险更高。</p>
              </div>
              <el-slider v-model="modelConfig.minRetrievalScore" :min="0" :max="1" :step="0.05" show-input />
            </div>
            <div class="config-item">
              <div>
                <strong>Mock ChatModel</strong>
                <p>开启后不调用真实大模型，适合无 API Key、本地演示和自动化测试。</p>
              </div>
              <el-switch v-model="modelConfig.mockEnabled" active-text="开启" inactive-text="关闭" />
            </div>
          </div>
          <div class="toolbar config-actions">
            <el-button type="primary" :loading="modelConfigSaving" @click="saveModelConfig">保存参数</el-button>
            <el-button :icon="Refresh" @click="loadModelConfig">恢复当前配置</el-button>
            <span class="muted">保存后会影响后续新的客服问答。</span>
          </div>
        </div>
      </el-tab-pane>

      <el-tab-pane label="文档管理">
        <div class="panel">
          <div class="toolbar">
            <el-upload :http-request="uploadDocument" :show-file-list="false">
              <el-button type="primary" :icon="UploadFilled">上传文档</el-button>
            </el-upload>
            <el-input v-model="keyword" class="toolbar-input" placeholder="按名称搜索" clearable @keyup.enter="loadDocuments" />
            <el-button :icon="Search" @click="loadDocuments">搜索</el-button>
          </div>
          <el-table :data="documents" border stripe v-loading="documentLoading">
            <el-table-column prop="originalName" label="文档名" min-width="220" />
            <el-table-column prop="fileType" label="类型" width="90" />
            <el-table-column label="状态" width="120">
              <template #default="{ row }">
                <el-tag :type="statusType(row.status)" effect="light">{{ statusLabel(row.status) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="chunkCount" label="片段数" width="100" />
            <el-table-column prop="failureReason" label="失败原因" min-width="180" />
            <el-table-column label="操作" width="300">
              <template #default="{ row }">
                <el-button text type="primary" @click="download(row)">下载</el-button>
                <el-button
                  text
                  type="success"
                  :loading="processingDocumentId === row.id"
                  :disabled="row.status === 'PROCESSING'"
                  @click="processDocument(row.id)"
                >
                  {{ row.status === 'READY' ? '重新处理' : '处理' }}
                </el-button>
                <el-button v-if="row.status === 'FAILED'" text type="warning" @click="retry(row.id)">重试</el-button>
                <el-button text type="danger" @click="remove(row.id)">删除</el-button>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </el-tab-pane>

      <el-tab-pane label="商品信息">
        <div class="panel">
          <div class="toolbar">
            <el-input v-model="productKeyword" class="toolbar-input" placeholder="商品名或编码" clearable @keyup.enter="loadProducts" />
            <el-button :icon="Search" @click="loadProducts">搜索</el-button>
          </div>
          <el-table :data="products" border stripe v-loading="productLoading">
            <el-table-column prop="productCode" label="编码" width="110" />
            <el-table-column prop="productName" label="商品" min-width="170" />
            <el-table-column prop="category" label="分类" width="110" />
            <el-table-column prop="price" label="价格" width="100" />
            <el-table-column prop="stockQuantity" label="库存" width="90" />
            <el-table-column label="状态" width="110">
              <template #default="{ row }">
                <el-tag :type="row.saleStatus === 'ON_SALE' ? 'success' : 'info'">{{ productStatusLabel(row.saleStatus) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="dispatchRule" label="发货规则" min-width="240" />
            <el-table-column prop="afterSaleRule" label="售后规则" min-width="260" />
          </el-table>
        </div>
      </el-tab-pane>

      <el-tab-pane label="订单物流">
        <div class="panel">
          <div class="toolbar">
            <el-select v-model="orderStatus" clearable placeholder="订单状态" class="toolbar-select" @change="loadOrders">
              <el-option label="待发货" value="WAITING_SHIPMENT" />
              <el-option label="已发货" value="SHIPPED" />
              <el-option label="运输中" value="IN_TRANSIT" />
              <el-option label="已签收" value="SIGNED" />
              <el-option label="退款中" value="REFUNDING" />
            </el-select>
            <el-input v-model="orderKeyword" class="toolbar-input" placeholder="订单号" clearable @keyup.enter="loadOrders" />
            <el-button :icon="Refresh" @click="loadOrders">刷新</el-button>
          </div>
          <el-table :data="orders" border stripe v-loading="orderLoading">
            <el-table-column prop="orderNo" label="订单号" min-width="170" />
            <el-table-column label="商品" min-width="170">
              <template #default="{ row }">{{ row.product.productName }} × {{ row.quantity }}</template>
            </el-table-column>
            <el-table-column label="状态" width="110">
              <template #default="{ row }">
                <el-tag :type="orderStatusType(row.status)">{{ orderStatusLabel(row.status) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="预计发货" width="160">
              <template #default="{ row }">{{ formatDate(row.expectedShipAt) }}</template>
            </el-table-column>
            <el-table-column label="最新物流" min-width="240">
              <template #default="{ row }">
                <span v-if="row.shipmentEvents.length">{{ row.shipmentEvents[0].eventNote }}</span>
                <span v-else class="muted">暂无物流</span>
              </template>
            </el-table-column>
            <el-table-column label="更新物流" width="520">
              <template #default="{ row }">
                <div class="order-action">
                  <el-select v-model="row.nextStatus" placeholder="新状态">
                    <el-option label="已发货" value="SHIPPED" />
                    <el-option label="运输中" value="IN_TRANSIT" />
                    <el-option label="已签收" value="SIGNED" />
                    <el-option label="退款中" value="REFUNDING" />
                  </el-select>
                  <el-input v-model="row.trackingNo" placeholder="物流单号" />
                  <el-input v-model="row.eventNote" placeholder="物流说明" />
                  <el-button text type="primary" @click="updateOrder(row)">保存</el-button>
                </div>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </el-tab-pane>

      <el-tab-pane label="工单处理">
        <div class="panel">
          <div class="toolbar">
            <el-select v-model="ticketStatus" clearable placeholder="状态筛选" class="toolbar-select" @change="loadTickets">
              <el-option label="待处理" value="OPEN" />
              <el-option label="处理中" value="PROCESSING" />
              <el-option label="已解决" value="RESOLVED" />
              <el-option label="已关闭" value="CLOSED" />
            </el-select>
            <el-button :icon="Refresh" @click="loadTickets">刷新</el-button>
          </div>
          <el-table :data="tickets" border stripe v-loading="ticketLoading">
            <el-table-column prop="ticketNo" label="工单编号" min-width="180" />
            <el-table-column prop="category" label="分类" width="120" />
            <el-table-column label="状态" width="120">
              <template #default="{ row }">
                <el-tag :type="statusType(row.status)" effect="light">{{ statusLabel(row.status) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="description" label="描述" min-width="220" />
            <el-table-column label="处理" width="360">
              <template #default="{ row }">
                <div class="ticket-action">
                  <el-select v-model="row.nextStatus" placeholder="新状态" style="width: 120px">
                    <el-option label="处理中" value="PROCESSING" />
                    <el-option label="已解决" value="RESOLVED" />
                    <el-option label="已关闭" value="CLOSED" />
                  </el-select>
                  <el-input v-model="row.resolution" placeholder="处理结果/备注" />
                  <el-button text type="primary" @click="updateTicket(row)">保存</el-button>
                </div>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </el-tab-pane>

      <el-tab-pane label="审批申请">
        <div class="panel">
          <div class="toolbar">
            <el-select v-model="actionStatus" clearable placeholder="审批状态" class="toolbar-select" @change="loadAgentActions">
              <el-option label="历史记录（需复核）" value="LEGACY_REVIEW_REQUIRED" />
              <el-option label="待审批" value="PENDING" />
              <el-option label="审批中" value="APPROVING" />
              <el-option label="已批准" value="APPROVED" />
              <el-option label="驳回处理中" value="REJECTING" />
              <el-option label="已执行" value="EXECUTED" />
              <el-option label="已驳回" value="REJECTED" />
              <el-option label="执行中" value="EXECUTING" />
              <el-option label="业务事实已变化" value="STALE" />
              <el-option label="失败" value="FAILED" />
              <el-option label="失败（可重试）" value="FAILED_RETRYABLE" />
            </el-select>
            <el-button :icon="Refresh" @click="loadAgentActions">刷新</el-button>
          </div>
          <el-table :data="agentActions" border stripe v-loading="actionLoading">
            <el-table-column label="类型" width="120">
              <template #default="{ row }">{{ actionTypeLabel(row.actionType) }}</template>
            </el-table-column>
            <el-table-column label="风险" width="90">
              <template #default="{ row }">
                <el-tag :type="riskType(row.riskLevel)" effect="light">{{ row.riskLevel }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="状态" width="150">
              <template #default="{ row }">
                <el-tag :type="actionStatusType(row.status)" effect="light">{{ actionStatusLabel(row.status) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="恢复状态" width="165">
              <template #default="{ row }">
                <el-tag :type="resumeStatusType(row.resumeStatus)" effect="light">{{ resumeStatusLabel(row.resumeStatus) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="申请内容" min-width="260">
              <template #default="{ row }">{{ actionPayloadLabel(row) }}</template>
            </el-table-column>
            <el-table-column label="备注" min-width="220" show-overflow-tooltip>
              <template #default="{ row }">{{ row.approvalNote || '暂无' }}</template>
            </el-table-column>
            <el-table-column label="创建时间" width="165">
              <template #default="{ row }">{{ formatDate(row.createdAt) }}</template>
            </el-table-column>
            <el-table-column label="操作" width="180" fixed="right">
              <template #default="{ row }">
                <template v-if="row.decisionAvailable">
                  <el-button
                    text
                    type="primary"
                    :loading="decisionInFlightId === row.id"
                    :disabled="decisionInFlightId !== null"
                    @click="approveAction(row)"
                  >通过</el-button>
                  <el-button
                    text
                    type="danger"
                    :loading="decisionInFlightId === row.id"
                    :disabled="decisionInFlightId !== null"
                    @click="rejectAction(row)"
                  >驳回</el-button>
                </template>
                <span v-else class="muted">不可操作</span>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </el-tab-pane>

      <el-tab-pane label="Agent运行">
        <div class="panel">
          <div class="toolbar">
            <el-select v-model="agentRunStatus" clearable placeholder="运行状态" class="toolbar-select" @change="loadAgentRuns">
              <el-option label="运行中" value="RUNNING" />
              <el-option label="等待客户确认" value="WAITING_CUSTOMER_CONFIRMATION" />
              <el-option label="等待管理员审批" value="WAITING_ADMIN_APPROVAL" />
              <el-option label="待恢复" value="RESUME_PENDING" />
              <el-option label="执行中" value="EXECUTING" />
              <el-option label="已完成" value="COMPLETED" />
              <el-option label="已驳回" value="REJECTED" />
              <el-option label="失败" value="FAILED" />
              <el-option label="已取消" value="CANCELLED" />
            </el-select>
            <el-select v-model="agentRunIntent" clearable placeholder="意图类型" class="toolbar-select" @change="loadAgentRuns">
              <el-option label="订单查询" value="ORDER_QUERY" />
              <el-option label="物流查询" value="SHIPPING_QUERY" />
              <el-option label="商品咨询" value="PRODUCT_QUERY" />
              <el-option label="知识库问答" value="KNOWLEDGE_QUERY" />
              <el-option label="退款申请" value="REFUND_REQUEST" />
              <el-option label="取消订单" value="CANCEL_ORDER" />
            </el-select>
            <el-button :icon="Refresh" @click="loadAgentRuns">刷新</el-button>
          </div>
          <el-table :data="agentRuns" border stripe v-loading="agentRunLoading">
            <el-table-column prop="runId" label="运行ID" min-width="210" />
            <el-table-column label="意图" width="120">
              <template #default="{ row }">{{ intentLabel(row.intent) }}</template>
            </el-table-column>
            <el-table-column label="风险" width="95">
              <template #default="{ row }">
                <el-tag :type="riskType(row.riskLevel)" effect="light">{{ row.riskLevel || 'LOW' }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="状态" width="110">
              <template #default="{ row }">
                <el-tag :type="agentStatusType(row.status)" effect="light">{{ agentStatusLabel(row.status) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="工具" width="90">
              <template #default="{ row }">{{ row.toolCallCount }}</template>
            </el-table-column>
            <el-table-column label="待审批" width="95">
              <template #default="{ row }">
                <el-tag v-if="row.pendingActionCount" type="warning">{{ row.pendingActionCount }}</el-tag>
                <span v-else class="muted">0</span>
              </template>
            </el-table-column>
            <el-table-column label="开始时间" width="165">
              <template #default="{ row }">{{ formatDate(row.startedAt) }}</template>
            </el-table-column>
            <el-table-column prop="finalAnswer" label="最终回答" min-width="260" show-overflow-tooltip />
            <el-table-column label="操作" width="100" fixed="right">
              <template #default="{ row }">
                <el-button text type="primary" @click="openAgentRun(row.runId)">详情</el-button>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </el-tab-pane>
    </el-tabs>

    <el-drawer v-model="agentRunDetailVisible" title="Agent运行详情" size="48%">
      <template v-if="agentRunDetail">
        <div class="run-summary">
          <div>
            <span>运行ID</span>
            <strong>{{ agentRunDetail.run.runId }}</strong>
          </div>
          <div>
            <span>意图</span>
            <strong>{{ intentLabel(agentRunDetail.run.intent) }}</strong>
          </div>
          <div>
            <span>状态</span>
            <strong>{{ agentStatusLabel(agentRunDetail.run.status) }}</strong>
          </div>
          <div>
            <span>风险</span>
            <strong>{{ agentRunDetail.run.riskLevel || 'LOW' }}</strong>
          </div>
        </div>
        <div class="detail-block">
          <h3>最终回答</h3>
          <p>{{ agentRunDetail.run.finalAnswer || '暂无最终回答' }}</p>
        </div>
        <div class="detail-block">
          <h3>执行步骤</h3>
          <el-timeline v-if="agentRunDetail.steps.length">
            <el-timeline-item v-for="step in agentRunDetail.steps" :key="step.id" :timestamp="formatDate(step.createdAt)">
              <div class="tool-call">
                <div>
                  <strong>{{ step.nodeName }}</strong>
                  <el-tag size="small" :type="agentStatusType(step.status)">{{ agentStatusLabel(step.status) }}</el-tag>
                </div>
                <p v-if="step.inputSummary">输入：{{ step.inputSummary }}</p>
                <p v-if="step.outputSummary">输出：{{ step.outputSummary }}</p>
                <p v-if="step.errorSummary">异常：{{ step.errorSummary }}</p>
              </div>
            </el-timeline-item>
          </el-timeline>
          <el-empty v-else description="暂无执行步骤" />
        </div>
        <div class="detail-block">
          <h3>工具调用</h3>
          <el-timeline v-if="agentRunDetail.toolCalls.length">
            <el-timeline-item v-for="tool in agentRunDetail.toolCalls" :key="tool.id" :timestamp="formatDate(tool.createdAt)">
              <div class="tool-call">
                <div>
                  <strong>{{ tool.toolName }}</strong>
                  <el-tag size="small" :type="tool.success ? 'success' : 'danger'">{{ tool.success ? '成功' : '失败' }}</el-tag>
                </div>
                <code>{{ tool.redactedArgumentsJson }}</code>
                <p>{{ tool.resultSummary || '无返回摘要' }}</p>
              </div>
            </el-timeline-item>
          </el-timeline>
          <el-empty v-else description="暂无工具调用" />
        </div>
        <div class="detail-block">
          <h3>审批动作</h3>
          <el-table :data="agentRunDetail.actionRequests" border size="small">
            <el-table-column prop="actionType" label="动作" width="145" />
            <el-table-column prop="riskLevel" label="风险" width="90" />
            <el-table-column prop="status" label="状态" width="110" />
            <el-table-column prop="approvalNote" label="备注" min-width="180" show-overflow-tooltip />
          </el-table>
        </div>
      </template>
    </el-drawer>
  </section>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox, type UploadRequestOptions } from 'element-plus'
import { Refresh, Search, UploadFilled } from '@element-plus/icons-vue'
import { api, unwrap } from '../api'
import {
  actionStatusPresentation,
  resumeStatusPresentation,
  runStatusPresentation,
  type AgentActionRow
} from '../adminAgentStatus'

interface Dashboard {
  todayConsultations: number
  productCount: number
  waitingShipmentOrders: number
  shippingOrders: number
  signedOrders: number
  pendingTickets: number
  readyDocuments: number
  failedDocuments: number
}

interface ModelConfig {
  temperature: number
  topK: number
  minRetrievalScore: number
  mockEnabled: boolean
}

interface DocumentRow {
  id: number
  originalName: string
  fileType: string
  status: string
  chunkCount: number
  failureReason?: string
}

interface TicketRow {
  id: number
  ticketNo: string
  category: string
  status: string
  description: string
  lockVersion: number
  nextStatus?: string
  resolution?: string
}

interface ProductRow {
  id: number
  productCode: string
  productName: string
  category: string
  saleStatus: string
  price: string
  stockQuantity: number
  dispatchRule: string
  afterSaleRule: string
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
  nextStatus?: string
  trackingNo?: string
  eventNote?: string
}

interface AgentRunRow {
  id: number
  runId: string
  threadId: string
  conversationId: number
  userId: number
  status: string
  intent?: string
  riskLevel?: string
  startedAt: string
  completedAt?: string
  finalAnswer?: string
  errorType?: string
  requestId: string
  toolCallCount: number
  pendingActionCount: number
}

interface AgentToolCallRow {
  id: number
  runId: string
  toolName: string
  redactedArgumentsJson: string
  resultSummary?: string
  success: boolean
  retryCount: number
  durationMs: number
  createdAt: string
}

interface AgentStepRow {
  id: number
  runId: string
  nodeName: string
  inputSummary?: string
  outputSummary?: string
  status: string
  durationMs: number
  errorSummary?: string
  createdAt: string
}

interface AgentRunDetail {
  run: AgentRunRow
  steps: AgentStepRow[]
  toolCalls: AgentToolCallRow[]
  actionRequests: AgentActionRow[]
}

const dashboard = reactive<Dashboard>({
  todayConsultations: 0,
  productCount: 0,
  waitingShipmentOrders: 0,
  shippingOrders: 0,
  signedOrders: 0,
  pendingTickets: 0,
  readyDocuments: 0,
  failedDocuments: 0
})
const modelConfig = reactive<ModelConfig>({
  temperature: 0.2,
  topK: 5,
  minRetrievalScore: 0.35,
  mockEnabled: true
})
const keyword = ref('')
const productKeyword = ref('')
const orderKeyword = ref('')
const documents = ref<DocumentRow[]>([])
const tickets = ref<TicketRow[]>([])
const products = ref<ProductRow[]>([])
const orders = ref<OrderRow[]>([])
const agentRuns = ref<AgentRunRow[]>([])
const agentActions = ref<AgentActionRow[]>([])
const ticketStatus = ref('')
const orderStatus = ref('')
const agentRunStatus = ref('')
const agentRunIntent = ref('')
const actionStatus = ref('PENDING')
const documentLoading = ref(false)
const processingDocumentId = ref<number | null>(null)
const ticketLoading = ref(false)
const productLoading = ref(false)
const orderLoading = ref(false)
const agentRunLoading = ref(false)
const actionLoading = ref(false)
const decisionInFlightId = ref<number | null>(null)
const agentRunDetailVisible = ref(false)
const agentRunDetail = ref<AgentRunDetail | null>(null)
const modelConfigSaving = ref(false)

onMounted(async () => {
  await Promise.all([
    loadDashboard(),
    loadModelConfig(),
    loadDocuments(),
    loadTickets(),
    loadProducts(),
    loadOrders(),
    loadAgentRuns(),
    loadAgentActions()
  ])
})

onUnmounted(stopDocumentPolling)

async function loadDashboard() {
  const data = await unwrap<Dashboard>(api.get('/admin/dashboard'))
  Object.assign(dashboard, data)
}

async function loadModelConfig() {
  const data = await unwrap<ModelConfig>(api.get('/admin/model-config'))
  Object.assign(modelConfig, {
    temperature: Number(data.temperature),
    topK: data.topK,
    minRetrievalScore: Number(data.minRetrievalScore),
    mockEnabled: data.mockEnabled
  })
}

async function saveModelConfig() {
  modelConfigSaving.value = true
  try {
    const data = await unwrap<ModelConfig>(api.put('/admin/model-config', modelConfig))
    Object.assign(modelConfig, {
      temperature: Number(data.temperature),
      topK: data.topK,
      minRetrievalScore: Number(data.minRetrievalScore),
      mockEnabled: data.mockEnabled
    })
    ElMessage.success('模型参数已保存')
  } finally {
    modelConfigSaving.value = false
  }
}

async function fetchDocuments() {
  const data = await unwrap<{ records: DocumentRow[] }>(api.get('/admin/documents', { params: { keyword: keyword.value } }))
  documents.value = data.records
}

async function loadDocuments() {
  documentLoading.value = true
  try {
    await fetchDocuments()
  } finally {
    documentLoading.value = false
    documentPollRounds = 0
    scheduleDocumentPoll()
  }
}

// 只要列表里还有 PENDING / PROCESSING 的文档，就每 2 秒静默刷新一次，最多 150 轮（约 5 分钟）。
// 上限是必要的：本地若没启动 document-worker，文档会一直停在 PENDING，
// 无上限轮询就变成了对接口的持续空转。任何一次手动刷新（搜索 / 上传 / 处理 / 重试）都会重新计时。
const DOCUMENT_POLL_INTERVAL_MS = 2000
const DOCUMENT_POLL_MAX_ROUNDS = 150
let documentPollTimer: number | undefined
let documentPollRounds = 0

function clearDocumentPoll() {
  if (documentPollTimer !== undefined) {
    window.clearTimeout(documentPollTimer)
    documentPollTimer = undefined
  }
}

function stopDocumentPolling() {
  clearDocumentPoll()
  documentPollRounds = 0
}

function scheduleDocumentPoll() {
  clearDocumentPoll()
  const hasPending = documents.value.some((row) => row.status === 'PENDING' || row.status === 'PROCESSING')
  if (!hasPending || documentPollRounds >= DOCUMENT_POLL_MAX_ROUNDS) {
    return
  }
  documentPollTimer = window.setTimeout(() => {
    documentPollTimer = undefined
    documentPollRounds += 1
    void pollDocuments()
  }, DOCUMENT_POLL_INTERVAL_MS)
}

async function pollDocuments() {
  try {
    await fetchDocuments()
  } catch {
    // 轮询失败不打扰用户，下一轮自动重试
  } finally {
    scheduleDocumentPoll()
  }
}

async function uploadDocument(options: UploadRequestOptions) {
  const form = new FormData()
  form.append('file', options.file)
  try {
    await unwrap(api.post('/admin/documents', form, { headers: { 'Content-Type': 'multipart/form-data' } }))
    ElMessage.success('上传完成，点该行「处理」完成切片入库')
    await Promise.all([loadDocuments(), loadDashboard()])
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '上传失败')
  }
}

async function download(row: DocumentRow) {
  const response = await api.get(`/admin/documents/${row.id}/download`, { responseType: 'blob' })
  const url = URL.createObjectURL(response.data)
  const link = document.createElement('a')
  link.href = url
  link.download = row.originalName
  link.click()
  URL.revokeObjectURL(url)
}

async function processDocument(id: number) {
  processingDocumentId.value = id
  try {
    await unwrap(api.post(`/admin/documents/${id}/process`))
    ElMessage.success('文档处理完成')
    await Promise.all([loadDocuments(), loadDashboard()])
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '文档处理失败')
    await loadDocuments()
  } finally {
    processingDocumentId.value = null
  }
}

async function retry(id: number) {
  await unwrap(api.post(`/admin/documents/${id}/retry`))
  await Promise.all([loadDocuments(), loadDashboard()])
}

async function remove(id: number) {
  await unwrap(api.delete(`/admin/documents/${id}`))
  await Promise.all([loadDocuments(), loadDashboard()])
}

async function loadProducts() {
  productLoading.value = true
  try {
    const data = await unwrap<{ records: ProductRow[] }>(api.get('/admin/products', { params: { keyword: productKeyword.value } }))
    products.value = data.records
  } finally {
    productLoading.value = false
  }
}

async function loadOrders() {
  const params: Record<string, string> = {}
  if (orderStatus.value) params.status = orderStatus.value
  if (orderKeyword.value) params.keyword = orderKeyword.value
  orderLoading.value = true
  try {
    const data = await unwrap<{ records: OrderRow[] }>(api.get('/admin/orders', { params }))
    orders.value = data.records
  } finally {
    orderLoading.value = false
  }
}

async function updateOrder(row: OrderRow) {
  if (!row.nextStatus) return
  await unwrap(api.patch(`/admin/orders/${row.id}/status`, {
    status: row.nextStatus,
    carrier: row.nextStatus === 'SHIPPED' || row.nextStatus === 'IN_TRANSIT' || row.nextStatus === 'SIGNED' ? '演示快递' : undefined,
    trackingNo: row.trackingNo,
    location: '演示网点',
    eventNote: row.eventNote
  }))
  ElMessage.success('订单状态已更新')
  await Promise.all([loadOrders(), loadDashboard()])
}

async function loadTickets() {
  const params: Record<string, string> = {}
  if (ticketStatus.value) params.status = ticketStatus.value
  ticketLoading.value = true
  try {
    const data = await unwrap<{ records: TicketRow[] }>(api.get('/admin/tickets', { params }))
    tickets.value = data.records
  } finally {
    ticketLoading.value = false
  }
}

async function updateTicket(row: TicketRow) {
  if (!row.nextStatus) return
  await unwrap(api.patch(`/admin/tickets/${row.id}/status`, {
    status: row.nextStatus,
    handlingNote: row.resolution || '后台已处理',
    resolution: row.nextStatus === 'RESOLVED' ? row.resolution : undefined,
    lockVersion: row.lockVersion
  }))
  ElMessage.success('工单已更新')
  await Promise.all([loadTickets(), loadDashboard()])
}

async function loadAgentRuns() {
  const params: Record<string, string | number> = { page: 1, size: 50 }
  if (agentRunStatus.value) params.status = agentRunStatus.value
  if (agentRunIntent.value) params.intent = agentRunIntent.value
  agentRunLoading.value = true
  try {
    const data = await unwrap<{ records: AgentRunRow[] }>(api.get('/admin/agent/runs', { params }))
    agentRuns.value = data.records
  } finally {
    agentRunLoading.value = false
  }
}

async function loadAgentActions() {
  const params: Record<string, string> = {}
  if (actionStatus.value) params.status = actionStatus.value
  actionLoading.value = true
  try {
    agentActions.value = await unwrap<AgentActionRow[]>(api.get('/admin/agent/actions', { params }))
  } finally {
    actionLoading.value = false
  }
}

async function approveAction(row: AgentActionRow) {
  const note = await promptApprovalNote('确认通过该审批申请？', '审批通过后会更新对应订单状态。')
  if (note === null) return
  decisionInFlightId.value = row.id
  try {
    await unwrap(api.post(`/admin/agent/actions/${row.id}/approve`, {
      lockVersion: row.lockVersion,
      approvalNote: note || undefined
    }))
    ElMessage.success(row.actionType === 'REFUND' ? '审批决定已提交，退款处理中或售后审核中' : '审批决定已提交')
    await Promise.all([loadAgentActions(), loadAgentRuns(), loadOrders(), loadDashboard()])
    if (agentRunDetail.value?.run.runId === row.runId) {
      await openAgentRun(row.runId)
    }
  } finally {
    decisionInFlightId.value = null
  }
}

async function rejectAction(row: AgentActionRow) {
  const note = await promptApprovalNote('确认驳回该审批申请？', '请填写驳回原因，便于后续客服跟进。', true)
  if (note === null) return
  decisionInFlightId.value = row.id
  try {
    await unwrap(api.post(`/admin/agent/actions/${row.id}/reject`, {
      lockVersion: row.lockVersion,
      approvalNote: note || '后台驳回申请'
    }))
    ElMessage.success('审批决定已提交')
    await Promise.all([loadAgentActions(), loadAgentRuns(), loadDashboard()])
    if (agentRunDetail.value?.run.runId === row.runId) {
      await openAgentRun(row.runId)
    }
  } finally {
    decisionInFlightId.value = null
  }
}

async function promptApprovalNote(title: string, message: string, required = false): Promise<string | null> {
  try {
    const result = await ElMessageBox.prompt(message, title, {
      confirmButtonText: '确认',
      cancelButtonText: '取消',
      inputPlaceholder: required ? '请输入处理原因' : '可填写处理备注',
      inputValidator: (value) => (!required || Boolean(value.trim())) || '请填写处理原因'
    })
    return result.value.trim()
  } catch {
    return null
  }
}

async function openAgentRun(runId: string) {
  const data = await unwrap<AgentRunDetail>(api.get(`/admin/agent/runs/${runId}`))
  agentRunDetail.value = data
  agentRunDetailVisible.value = true
}

function statusLabel(status: string) {
  return {
    COMPLETED: '已完成',
    READY: '已就绪',
    PROCESSING: '处理中',
    OPEN: '待处理',
    PENDING: '待处理',
    FAILED: '失败',
    RESOLVED: '已解决',
    CLOSED: '已关闭'
  }[status] ?? status
}

function statusType(status: string) {
  if (status === 'COMPLETED' || status === 'READY' || status === 'RESOLVED') return 'success'
  if (status === 'PROCESSING') return 'warning'
  if (status === 'FAILED') return 'danger'
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

function orderStatusLabel(status: string) {
  return {
    PENDING_PAYMENT: '待付款',
    PAID: '已付款',
    WAITING_SHIPMENT: '待发货',
    SHIPPED: '已发货',
    IN_TRANSIT: '运输中',
    SIGNED: '已签收',
    REFUND_PENDING: '退款处理中',
    AFTER_SALE_REVIEWING: '售后审核中',
    REFUNDING: '退款中',
    REFUNDED: '已退款',
    CANCELLED: '已取消'
  }[status] ?? status
}

function orderStatusType(status: string) {
  if (status === 'SIGNED') return 'success'
  if (status === 'IN_TRANSIT' || status === 'SHIPPED') return 'warning'
  if (status === 'REFUND_PENDING' || status === 'AFTER_SALE_REVIEWING' || status === 'REFUNDING') return 'warning'
  if (status === 'CANCELLED' || status === 'REFUNDED') return 'info'
  return 'primary'
}

function actionTypeLabel(type: string) {
  return {
    REFUND: '退款申请',
    ORDER_CANCELLATION: '取消订单'
  }[type] ?? type
}

function actionStatusLabel(status: string) {
  return actionStatusPresentation(status).label
}

function actionStatusType(status: string) {
  return actionStatusPresentation(status).tag
}

function resumeStatusLabel(status: string) {
  return resumeStatusPresentation(status).label
}

function resumeStatusType(status: string) {
  return resumeStatusPresentation(status).tag
}

function actionPayloadLabel(row: AgentActionRow) {
  try {
    const payload = JSON.parse(row.actionPayloadJson) as { reason?: string }
    return payload.reason ? `原因：${payload.reason}` : row.actionPayloadJson
  } catch {
    return row.actionPayloadJson || '暂无'
  }
}

function agentStatusLabel(status: string) {
  return runStatusPresentation(status).label
}

function agentStatusType(status: string) {
  return runStatusPresentation(status).tag
}

function intentLabel(intent?: string) {
  if (!intent) return '未识别'
  return {
    ORDER_QUERY: '订单查询',
    SHIPPING_QUERY: '物流查询',
    PRODUCT_QUERY: '商品咨询',
    KNOWLEDGE_QUERY: '知识库问答',
    REFUND_REQUEST: '退款申请',
    CANCEL_ORDER: '取消订单',
    SMALL_TALK: '闲聊'
  }[intent] ?? intent
}

function riskType(risk?: string) {
  if (risk === 'HIGH') return 'danger'
  if (risk === 'MEDIUM') return 'warning'
  return 'success'
}

function formatDate(value?: string) {
  return value ? value.replace('T', ' ').slice(0, 16) : '暂未同步'
}
</script>

<style scoped>
.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin: 18px 0;
}

.metric-card {
  display: grid;
  gap: 8px;
}

.metric-card span {
  color: #75685f;
}

.metric-card strong {
  font-size: 30px;
  color: #c45656;
}

.dashboard-panel {
  margin-top: 14px;
}

.notice-list {
  display: grid;
  gap: 10px;
  color: #5f5048;
}

.config-grid {
  display: grid;
  gap: 18px;
}

.config-item {
  display: grid;
  grid-template-columns: minmax(220px, 320px) minmax(260px, 1fr);
  gap: 20px;
  align-items: center;
  padding: 16px 0;
  border-bottom: 1px solid #f0e5dc;
}

.config-item p {
  margin: 6px 0 0;
  color: #75685f;
  line-height: 1.6;
}

.config-actions {
  margin-top: 18px;
}

.ticket-action {
  display: grid;
  grid-template-columns: 120px minmax(120px, 1fr) auto;
  gap: 8px;
  align-items: center;
}

.order-action {
  display: grid;
  grid-template-columns: 110px 120px minmax(120px, 1fr) auto;
  gap: 8px;
  align-items: center;
}

.run-summary {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 20px;
}

.run-summary div {
  display: grid;
  gap: 6px;
  padding: 14px;
  border: 1px solid #f0e5dc;
  border-radius: 8px;
  background: #fffaf6;
}

.run-summary span {
  color: #8b7d72;
  font-size: 13px;
}

.run-summary strong {
  overflow-wrap: anywhere;
  color: #3f342d;
}

.detail-block {
  margin-top: 20px;
}

.detail-block h3 {
  margin: 0 0 12px;
  font-size: 16px;
  color: #3f342d;
}

.detail-block p {
  margin: 0;
  color: #5f5048;
  line-height: 1.7;
  white-space: pre-wrap;
}

.tool-call {
  display: grid;
  gap: 8px;
}

.tool-call > div {
  display: flex;
  align-items: center;
  gap: 8px;
}

.tool-call code {
  display: block;
  padding: 10px;
  border-radius: 6px;
  background: #f7f1eb;
  color: #5f5048;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.muted {
  color: #8b7d72;
}

@media (max-width: 900px) {
  .dashboard-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .order-action,
  .ticket-action,
  .run-summary,
  .config-item {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 520px) {
  .dashboard-grid {
    grid-template-columns: 1fr;
  }
}
</style>
