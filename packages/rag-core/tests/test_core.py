from pathlib import Path

from hbc_rag.pipeline import _extract_json
from hbc_rag.policies import CoarseHop, FineHop
from hbc_rag.store.pg import vec_literal


def test_fine_hop_grows_one_chunk_per_hop():
    policy = FineHop()
    assert [policy.retrieval_k(i) for i in range(4)] == [1, 2, 3, 4]


def test_coarse_hop_is_bounded_by_top_k():
    policy = CoarseHop()
    assert policy.retrieval_k(0) == 4
    assert policy.retrieval_k(1) == policy.top_k


def test_extract_json_handles_fenced_output():
    text = '```json\n{"action": "answer", "answer": "Gamma"}\n```'
    assert _extract_json(text)["answer"] == "Gamma"


def test_extract_json_falls_back_to_plain_text():
    parsed = _extract_json("Beta")
    assert parsed["action"] == "answer"
    assert parsed["answer"] == "Beta"


def test_vec_literal_shape():
    assert vec_literal([1.0, 0.5]) == "[1.000000,0.500000]"


def test_schema_is_namespaced():
    schema = (Path(__file__).parents[1] / "src" / "hbc_rag" / "store" / "schema.sql").read_text()
    assert "create schema if not exists hbc_rag" in schema
    assert "set search_path to hbc_rag" in schema
