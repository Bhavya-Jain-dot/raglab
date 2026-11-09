from app.engine import chunks, tokens
from app.security import hash_password, verify_password
from app.evaluation import evaluate_result
from app.pricing import estimate_cost
from app.retrieval import rank
from app.main import normalize_pipeline_config
import uuid

def test_chunking_preserves_content_and_overlap():
    source=" ".join(f"word{i}" for i in range(120))
    parts=chunks(source,50,10)
    assert len(parts)>1
    assert parts[0].split()[-10:]==parts[1].split()[:10]
    assert "word119" in parts[-1]

def test_chunking_bounds_unsafe_values():
    assert chunks("short text",500,900)==["short text"]
    assert chunks("",100,0)==[]

def test_tokenizer_is_case_insensitive():
    assert tokens("Policy 123, POLICY!")=={"policy","123"}

def test_password_hashing_uses_salt_and_verifies():
    first=hash_password("correct horse battery staple")
    second=hash_password("correct horse battery staple")
    assert first!=second
    assert verify_password("correct horse battery staple",first)
    assert not verify_password("wrong password",first)

def test_workspace_boundary_is_owner_scoped(client):
    def register():
        response=client.post("/api/auth/register",json={"email":f"{uuid.uuid4().hex}@example.test","password":"correct horse battery staple","name":"Tester"})
        assert response.status_code==200
        return response.json()
    owner=register(); other=register()
    project=client.post("/api/projects",headers={"Authorization":f"Bearer {owner['access_token']}"},json={"name":"Private project"}).json()
    denied=client.get(f"/api/projects/{project['id']}/summary",headers={"Authorization":f"Bearer {other['access_token']}"})
    assert denied.status_code==404

def test_project_authorization_rejects_a_user_from_another_workspace():
    import pytest
    from fastapi import HTTPException
    from app.database import Base, engine, SessionLocal
    from app.main import project_or_404
    from app.models import User, Workspace, WorkspaceMember, Project
    Base.metadata.create_all(bind=engine)
    db=SessionLocal()
    try:
        owner=User(email=f"{uuid.uuid4().hex}@example.test",name="Owner",password_hash="test")
        outsider=User(email=f"{uuid.uuid4().hex}@example.test",name="Outsider",password_hash="test")
        db.add_all([owner,outsider]);db.flush()
        workspace=Workspace(name="Private",owner_id=owner.id);db.add(workspace);db.flush()
        db.add(WorkspaceMember(workspace_id=workspace.id,user_id=owner.id,role="owner"))
        project=Project(owner_id=owner.id,workspace_id=workspace.id,name="Secret",description="",domain="test");db.add(project);db.commit()
        with pytest.raises(HTTPException) as denied:
            project_or_404(project.id,outsider,db)
        assert denied.value.status_code==404
    finally:
        db.rollback();db.close()

def test_refresh_token_rotation_rejects_reuse(client):
    data=client.post("/api/auth/register",json={"email":f"{uuid.uuid4().hex}@example.test","password":"correct horse battery staple","name":"Tester"}).json()
    rotated=client.post("/api/auth/refresh",json={"refresh_token":data["refresh_token"]})
    assert rotated.status_code==200
    reused=client.post("/api/auth/refresh",json={"refresh_token":data["refresh_token"]})
    assert reused.status_code==401

def test_retrieval_metrics_handle_known_relevance_and_missing_labels():
    rows=[{"document_id":"doc-b","text":"unrelated content"},{"document_id":"doc-a","text":"refund policy allows returns"}]
    scores,details=evaluate_result("refund policy", "Returns are allowed", rows, "Returns are allowed", ["doc-a"], ["refund policy allows returns"])
    assert scores["precision_at_k"]==0.5
    assert scores["recall_at_k"]==1.0
    assert scores["mrr"]==0.5
    assert details["faithfulness_overlap_proxy"]["method"]=="token support proxy"
    query_only,query_details=evaluate_result("where is the office", "The office is downtown", rows, None, [], [])
    assert query_only["answer_correctness_token_f1"] is None
    assert query_details["answer_correctness_token_f1"]["status"]=="not_computed"

def test_hybrid_retrieval_uses_both_lexical_and_vector_signals():
    candidates=[
        {"id":1,"text":"refund rules", "embedding":[1.0,0.0]},
        {"id":2,"text":"payment procedure unrelated terms", "embedding":[0.99,0.01]},
    ]
    ranked=rank(candidates,"refund policy",2,"hybrid",[1.0,0.0])
    assert len(ranked)==2
    assert all("fusion_score" in row for row in ranked)

def test_pipeline_validation_rejects_vector_search_without_embeddings():
    import pytest
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        normalize_pipeline_config({"retrieval_strategy":"hybrid","embedding_provider":"lexical"})
    assert error.value.status_code==422

def test_pricing_remains_unknown_when_no_rate_is_configured(monkeypatch):
    monkeypatch.setenv("RAGLAB_PRICING_JSON","{}")
    row=estimate_cost("openai","some-model",{"prompt_tokens":100,"completion_tokens":20})
    assert row["estimated_usd"] is None
    assert row["status"]=="unpriced"
