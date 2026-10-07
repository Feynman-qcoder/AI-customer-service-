from pathlib import Path
from types import SimpleNamespace

import pytest
from docx import Document

from app.embeddings.mock_embedding import MockEmbeddingClient
from app.rag.chunker import chunk_text
from app.rag.document_parser import extract_text
from app.repositories.agent_workflow_repository import _extract_keywords as repository_extract_keywords
from app.runtime.model_config import EffectiveModelRuntimeConfig
from app.schemas.retrieval import RetrievalCandidate, RetrievalQueryContext
from app.services.knowledge_service import (
    KnowledgeService,
    _condition_matches,
    _extract_after_sale_type,
    _threshold_for_query,
    dedupe_candidates,
    heuristic_rerank,
    rrf_fuse,
)
from app.services.knowledge_service import (
    _extract_keywords as service_extract_keywords,
)


def test_text_parser_decodes_markdown_content() -> None:
    text = extract_text("policy.md", "# 退换货政策\n商品未影响二次销售时可以提交退货申请。".encode())

    assert "退换货政策" in text
    assert "二次销售" in text


def test_docx_parser_extracts_paragraphs(tmp_path: Path) -> None:
    path = tmp_path / "policy.docx"
    document = Document()
    document.add_paragraph("退款会按原支付路径退回。")
    document.add_paragraph("质量问题可以申请换货。")
    document.save(str(path))

    text = extract_text(path.name, path.read_bytes())

    assert "原支付路径" in text
    assert "质量问题" in text


def test_chunk_text_splits_long_policy_with_overlap() -> None:
    text = "发货规则：" + "下单后 24 小时内发货。" * 80

    chunks = chunk_text(text, max_chars=180, overlap=30)

    assert len(chunks) > 1
    assert chunks[0].index == 0
    assert all(chunk.char_count <= 180 for chunk in chunks)
    assert all(chunk.content_hash for chunk in chunks)


@pytest.mark.asyncio
async def test_mock_embedding_is_deterministic_and_normalized() -> None:
    client = MockEmbeddingClient(32)

    first = await client.embed("商品发货时间")
    second = await client.embed("商品发货时间")

    assert first == second
    assert len(first) == 32
    assert abs(sum(value * value for value in first) - 1.0) < 0.000001


def test_rrf_fusion_deduplicates_same_chunk() -> None:
    keyword = RetrievalCandidate(
        candidate_id="chunk:1",
        source_type="keyword",
        content="拆封后未影响二次销售可以退货",
        document_id="1",
        chunk_id="1",
        rule_id=None,
        metadata={"matched_terms": ["拆封", "退货"]},
        original_score=0.7,
    )
    dense = keyword.model_copy(update={"source_type": "dense_vector", "original_score": 0.9})
    other = RetrievalCandidate(
        candidate_id="chunk:2",
        source_type="keyword",
        content="发货规则",
        document_id="1",
        chunk_id="2",
        rule_id=None,
        metadata={},
        original_score=0.4,
    )

    fused = rrf_fuse([[keyword, other], [dense]])

    assert [item.candidate_id for item in fused].count("chunk:1") == 1
    assert fused[0].candidate_id == "chunk:1"
    assert fused[0].fused_score is not None


def test_heuristic_rerank_promotes_structured_rule_for_order_context() -> None:
    doc = RetrievalCandidate(
        candidate_id="chunk:1",
        source_type="keyword",
        content="通用文档：拆封可能需要人工判断",
        document_id="1",
        chunk_id="1",
        rule_id=None,
        metadata={},
        original_score=0.8,
        fused_score=0.01,
    )
    rule = RetrievalCandidate(
        candidate_id="rule:1",
        source_type="structured_rule",
        content="结构化规则：签收七天内，未影响二次销售可退货",
        document_id=None,
        chunk_id=None,
        rule_id="1",
        metadata={"rule_version": "AS-2026-07"},
        original_score=0.6,
        fused_score=0.01,
    )

    reranked = heuristic_rerank(
        "这个拆封后还能退吗",
        [doc, rule],
        RetrievalQueryContext(has_specific_order=True, order_status="SIGNED", after_sale_type="RETURN"),
    )

    assert reranked[0].source_type == "structured_rule"
    assert reranked[0].decision_reason == "命中当前订单状态适用的结构化业务规则"


def test_heuristic_rerank_promotes_explicit_product_document_above_generic_ties() -> None:
    generic = RetrievalCandidate(
        candidate_id="chunk:generic",
        source_type="keyword",
        content="通用规则：商品拆封后可按退货条件判断",
        document_id="1",
        chunk_id="1",
        rule_id=None,
        metadata={"file_name": "退换货政策.md"},
        original_score=0.65,
        fused_score=1 / 61,
    )
    product = RetrievalCandidate(
        candidate_id="chunk:p9",
        source_type="keyword",
        content="P9 未清洗、未明显使用且包装完整时可提交退货申请",
        document_id="2",
        chunk_id="2",
        rule_id=None,
        metadata={"file_name": "商品资料-云感靠枕P9.md"},
        original_score=0.65,
        fused_score=1 / 61,
    )

    reranked = heuristic_rerank("P9 拆封后还能退货吗？", [generic, product])

    assert reranked[0].candidate_id == "chunk:p9"


def test_structured_condition_filters_inapplicable_order_status() -> None:
    condition = SimpleNamespace(
        product_category=None,
        order_status="SIGNED",
        payment_status=None,
        shipment_status=None,
        signed_within_days=7,
        after_sale_type="RETURN",
    )

    matched, _ = _condition_matches(
        condition,  # type: ignore[arg-type]
        RetrievalQueryContext(has_specific_order=True, order_status="WAITING_SHIPMENT", after_sale_type="RETURN"),
    )

    assert matched is False


def test_structured_condition_allows_general_rule_without_specific_order() -> None:
    condition = SimpleNamespace(
        product_category=None,
        order_status="SIGNED",
        payment_status=None,
        shipment_status=None,
        signed_within_days=7,
        after_sale_type="RETURN",
    )

    matched, reasons = _condition_matches(
        condition,  # type: ignore[arg-type]
        RetrievalQueryContext(has_specific_order=False, after_sale_type="RETURN"),
    )

    assert matched is True
    assert "order_status=SIGNED" in reasons


def test_freight_question_routes_to_freight_rule_type() -> None:
    assert _extract_after_sale_type("退货包运费吗") == "RETURN_FREIGHT"
    assert _extract_after_sale_type("退货邮费谁承担") == "RETURN_FREIGHT"


def test_dedupe_candidates_by_candidate_id() -> None:
    first = RetrievalCandidate(
        candidate_id="rule:1",
        source_type="structured_rule",
        content="新规则",
        document_id=None,
        chunk_id=None,
        rule_id="1",
        metadata={},
        original_score=1,
    )
    duplicate = first.model_copy(update={"content": "旧规则"})

    assert dedupe_candidates([first, duplicate]) == [first]


@pytest.mark.asyncio
async def test_dense_vector_recall_returns_empty_when_qdrant_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken_search(_query_vector: list[float], _limit: int) -> list[object]:
        raise RuntimeError("qdrant down")

    from app.repositories.qdrant_store import qdrant_store

    monkeypatch.setattr(qdrant_store, "search", broken_search)

    assert await KnowledgeService().dense_vector_recall("拆封还能退吗", 3) == []


@pytest.mark.asyncio
async def test_retrieve_continues_when_keyword_recall_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    service = KnowledgeService()
    vector_candidate = RetrievalCandidate(
        candidate_id="chunk:9",
        source_type="dense_vector",
        content="语义命中：商品外包装损坏",
        document_id="1",
        chunk_id="9",
        rule_id=None,
        metadata={},
        original_score=0.7,
    )
    rule_candidate = RetrievalCandidate(
        candidate_id="rule:9",
        source_type="structured_rule",
        content="结构化规则：破损商品保留凭证后申请售后",
        document_id=None,
        chunk_id=None,
        rule_id="9",
        metadata={"rule_version": "AS-2026-07"},
        original_score=0.8,
    )

    async def broken_keyword(*_args: object, **_kwargs: object) -> list[RetrievalCandidate]:
        raise RuntimeError("keyword down")

    async def fake_dense(*_args: object, **_kwargs: object) -> list[RetrievalCandidate]:
        return [vector_candidate]

    async def fake_rule(*_args: object, **_kwargs: object) -> list[RetrievalCandidate]:
        return [rule_candidate]

    async def empty_cache(*_args: object, **_kwargs: object) -> None:
        return None

    async def noop_cache(*_args: object, **_kwargs: object) -> None:
        return None

    from app.services.redis_runtime_service import redis_runtime_service

    monkeypatch.setattr(service, "keyword_recall", broken_keyword)
    monkeypatch.setattr(service, "dense_vector_recall", fake_dense)
    monkeypatch.setattr(service, "structured_rule_recall", fake_rule)
    monkeypatch.setattr(redis_runtime_service, "get_json", empty_cache)
    monkeypatch.setattr(redis_runtime_service, "set_json", noop_cache)

    result = await service.retrieve_with_diagnostics(object(), "收到商品坏了怎么办", limit=3)  # type: ignore[arg-type]
    candidates = result.candidates

    assert {candidate.source_type for candidate in candidates} == {"dense_vector", "structured_rule"}
    assert any(
        diagnostic.channel == "keyword" and diagnostic.status == "FAILED"
        for diagnostic in result.diagnostics
    )


@pytest.mark.asyncio
async def test_retrieve_returns_empty_when_all_channels_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    service = KnowledgeService()
    weak = RetrievalCandidate(
        candidate_id="chunk:weak",
        source_type="dense_vector",
        content="完全无关内容",
        document_id="1",
        chunk_id="1",
        rule_id=None,
        metadata={},
        original_score=0.01,
    )

    async def empty_keyword(*_args: object, **_kwargs: object) -> list[RetrievalCandidate]:
        return []

    async def weak_dense(*_args: object, **_kwargs: object) -> list[RetrievalCandidate]:
        return [weak]

    async def empty_rule(*_args: object, **_kwargs: object) -> list[RetrievalCandidate]:
        return []

    monkeypatch.setattr(service, "keyword_recall", empty_keyword)
    monkeypatch.setattr(service, "dense_vector_recall", weak_dense)
    monkeypatch.setattr(service, "structured_rule_recall", empty_rule)

    candidates = await service.retrieve(object(), "天上的云是什么味道", limit=3)  # type: ignore[arg-type]

    assert candidates == []


def test_support_question_threshold_is_relaxed_but_bounded() -> None:
    assert _threshold_for_query("商品破损怎么售后", configured_min_score=0.35) == pytest.approx(0.08)
    assert _threshold_for_query("商品破损怎么售后", configured_min_score=0.60) == pytest.approx(0.33)
    assert _threshold_for_query("商品库存还有多少", configured_min_score=0.35) == pytest.approx(0.35)


@pytest.mark.parametrize(
    ("query", "expected_terms"),
    [
        ("偏远地区物流时效是多少？", {"偏远地区", "物流时效", "物流"}),
        ("配送多久可以送达？", {"配送", "送达"}),
    ],
)
def test_logistics_keyword_extraction_is_precise_and_shared(
    query: str,
    expected_terms: set[str],
) -> None:
    service_terms = service_extract_keywords(query)
    repository_terms = repository_extract_keywords(query)

    assert expected_terms <= set(service_terms)
    assert repository_terms == service_terms


def test_logistics_support_threshold_is_relaxed_but_bounded() -> None:
    assert _threshold_for_query("偏远地区配送多久送达", configured_min_score=0.35) == pytest.approx(0.08)
    assert _threshold_for_query("物流时效是多少", configured_min_score=0.60) == pytest.approx(0.33)


@pytest.mark.asyncio
async def test_unique_logistics_keyword_candidate_survives_retrieval_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyCache:
        async def get_json(self, *_args: object, **_kwargs: object) -> None:
            return None

        async def set_json(self, *_args: object, **_kwargs: object) -> None:
            return None

    service = KnowledgeService(cache=EmptyCache())  # type: ignore[arg-type]
    candidate = RetrievalCandidate(
        candidate_id="chunk:logistics",
        source_type="keyword",
        content="发货后普通地区通常 2 到 5 天送达，偏远地区可能延迟。",
        document_id="31",
        chunk_id="31",
        rule_id=None,
        metadata={"file_name": "发货与物流规则.md", "matched_terms": ["物流时效", "偏远地区"]},
        original_score=0.65,
    )

    async def keyword_loader(_query: str, _limit: int) -> list[RetrievalCandidate]:
        return [candidate]

    async def rule_loader(
        _query: str,
        _limit: int,
        _context: RetrievalQueryContext,
    ) -> list[RetrievalCandidate]:
        return []

    async def empty_dense(_query: str, _limit: int) -> list[RetrievalCandidate]:
        return []

    monkeypatch.setattr(service, "dense_vector_recall", empty_dense)
    result = await service.retrieve_with_ports(
        "偏远地区物流时效是多少？",
        runtime=EffectiveModelRuntimeConfig(
            temperature=0.2,
            top_k=5,
            min_retrieval_score=0.35,
            mock_enabled=True,
        ),
        keyword_loader=keyword_loader,
        rule_loader=rule_loader,
    )

    assert [item.metadata.get("file_name") for item in result.candidates] == ["发货与物流规则.md"]
