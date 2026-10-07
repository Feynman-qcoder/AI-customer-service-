import re
from dataclasses import dataclass

from app.core.config import settings


@dataclass(frozen=True)
class RetrievalScoringConfig:
    rrf_k: int = 60
    support_min_threshold: float = 0.08
    support_threshold_relaxation: float = 0.27
    support_terms: tuple[str, ...] = (
        "退款",
        "退货",
        "运费",
        "邮费",
        "破损",
        "损坏",
        "坏了",
        "质量",
        "拆封",
        "售后",
        "凭证",
        "物流时效",
        "偏远地区",
        "配送",
        "送达",
    )

    def threshold_for_query(self, query: str, configured_min_score: float | None = None) -> float:
        base_score = configured_min_score if configured_min_score is not None else settings.rag_min_retrieval_score
        if any(term in query for term in self.support_terms):
            return max(self.support_min_threshold, base_score - self.support_threshold_relaxation)
        return base_score


retrieval_scoring_config = RetrievalScoringConfig()


_RETRIEVAL_DOMAIN_TERMS: tuple[str, ...] = (
    "退款",
    "退货",
    "换货",
    "发货",
    "物流",
    "物流时效",
    "偏远地区",
    "配送",
    "送达",
    "快递",
    "订单",
    "拆封",
    "售后",
    "质量",
    "库存",
    "支付",
    "运费",
    "邮费",
    "承担",
    "包邮",
    "破损",
    "损坏",
    "包装",
    "凭证",
    "照片",
    "视频",
    "二次销售",
    "影响二次销售",
)


def extract_retrieval_keywords(query: str) -> list[str]:
    """Extract deterministic domain keywords for every database-backed recall path."""

    words = [word for word in re.split(r"[\s,，。？?、；;：:]+", query) if len(word) >= 2]
    terms = [term for term in _RETRIEVAL_DOMAIN_TERMS if term in query]
    upper_query = query.upper()
    for product_code in ("H100", "C20", "P9"):
        if product_code in upper_query and product_code not in terms:
            terms.append(product_code)
    for word in words:
        if word not in terms:
            terms.append(word)
    return terms[:8]
