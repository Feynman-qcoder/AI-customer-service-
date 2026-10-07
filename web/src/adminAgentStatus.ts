export type AdminStatusTag = 'primary' | 'success' | 'warning' | 'info' | 'danger'

export interface AdminStatusPresentation {
  label: string
  tag: AdminStatusTag
  known: boolean
}

export interface AgentActionRow {
  id: number
  runId: string
  actionType: string
  targetOrderId?: number
  actionPayloadJson: string
  riskLevel: string
  status: string
  idempotencyKey: string
  lockVersion: number
  createdBy: number
  approvedBy?: number
  approvalNote?: string
  logicalActionId?: string
  confirmationMode: string
  customerConfirmedActorId?: number
  customerConfirmedAt?: string
  customerConfirmationChallengeDigest?: string
  adminDecision?: string
  adminDecidedActorId?: number
  adminDecidedAt?: string
  adminReasonCode?: string
  resumeStatus: string
  decisionAvailable: boolean
  executionResultCode?: string
  executionErrorType?: string
  executionErrorSummary?: string
  legacyOriginalStatus?: string
  createdAt: string
  approvedAt?: string
  executedAt?: string
}

const UNKNOWN_STATUS: AdminStatusPresentation = Object.freeze({
  label: '未知状态（不可操作）',
  tag: 'info',
  known: false
})

const ACTION_STATUS: Readonly<Record<string, AdminStatusPresentation>> = Object.freeze({
  LEGACY_REVIEW_REQUIRED: { label: '历史记录（需复核）', tag: 'warning', known: true },
  PENDING: { label: '待审批', tag: 'warning', known: true },
  APPROVING: { label: '审批中', tag: 'warning', known: true },
  APPROVED: { label: '已批准', tag: 'primary', known: true },
  REJECTING: { label: '驳回处理中', tag: 'warning', known: true },
  REJECTED: { label: '已驳回', tag: 'danger', known: true },
  EXECUTING: { label: '执行中', tag: 'warning', known: true },
  EXECUTED: { label: '已执行', tag: 'success', known: true },
  STALE: { label: '业务事实已变化', tag: 'info', known: true },
  FAILED: { label: '失败', tag: 'danger', known: true },
  FAILED_RETRYABLE: { label: '失败（可重试）', tag: 'danger', known: true }
})

const RESUME_STATUS: Readonly<Record<string, AdminStatusPresentation>> = Object.freeze({
  NOT_APPLICABLE: { label: '不适用', tag: 'info', known: true },
  WAITING_ADMIN_DECISION: { label: '等待管理员决定', tag: 'warning', known: true },
  RESUME_PENDING: { label: '待恢复', tag: 'warning', known: true },
  RESUMED: { label: '已恢复', tag: 'primary', known: true },
  COMPLETED: { label: '已完成', tag: 'success', known: true },
  FAILED_RETRYABLE: { label: '恢复失败（可重试）', tag: 'danger', known: true },
  LEGACY_BLOCKED: { label: '历史流程已阻断', tag: 'danger', known: true }
})

const RUN_STATUS: Readonly<Record<string, AdminStatusPresentation>> = Object.freeze({
  RUNNING: { label: '运行中', tag: 'primary', known: true },
  WAITING: { label: '等待中', tag: 'warning', known: true },
  WAITING_CUSTOMER_CONFIRMATION: { label: '等待客户确认', tag: 'warning', known: true },
  WAITING_ADMIN_APPROVAL: { label: '等待管理员审批', tag: 'warning', known: true },
  RESUME_PENDING: { label: '待恢复', tag: 'warning', known: true },
  EXECUTING: { label: '执行中', tag: 'warning', known: true },
  COMPLETED: { label: '已完成', tag: 'success', known: true },
  REJECTED: { label: '已驳回', tag: 'danger', known: true },
  FAILED: { label: '失败', tag: 'danger', known: true },
  CANCELLED: { label: '已取消', tag: 'info', known: true }
})

export function actionStatusPresentation(status: string): AdminStatusPresentation {
  return ACTION_STATUS[status] ?? UNKNOWN_STATUS
}

export function resumeStatusPresentation(status: string): AdminStatusPresentation {
  return RESUME_STATUS[status] ?? UNKNOWN_STATUS
}

export function runStatusPresentation(status: string): AdminStatusPresentation {
  return RUN_STATUS[status] ?? UNKNOWN_STATUS
}
