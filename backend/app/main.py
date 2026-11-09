from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
import csv, hashlib, io, json, os, random, statistics, time
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from .database import get_db
from . import models
from .models import User, RefreshSession, Project, Dataset, DatasetVersion, Example, Document, Pipeline, Experiment, Result, Trace, PlaygroundTrace, Workspace, WorkspaceMember, ProviderCredential, AuditEvent, Recommendation
from .security import get_current_user, hash_password, verify_password, issue_token, issue_refresh_token, refresh_hash
from .engine import run_experiment, chunks, tokens, rank_chunks, build_or_load_index, generate_answer, vector_search
from .embeddings import embed_texts, embedding_key
from .retrieval import rank, cross_encoder_rerank
from .evaluation import evaluate_result
from .pricing import estimate_cost
from .observability import export_spans
from .object_storage import store_original, delete_object
from .tasks import execute_experiment
from .credentials import encrypt_secret, decrypt_secret

app = FastAPI(title="RAGLab API", version="0.1.0", description="RAG evaluation workbench API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

class Credentials(BaseModel): email: str = Field(min_length=3, max_length=320); password: str = Field(min_length=10, max_length=256); name: str = Field(default="", max_length=120)
class Login(BaseModel): email: str; password: str
class RefreshIn(BaseModel): refresh_token: str
class ProjectIn(BaseModel): name: str = Field(min_length=1,max_length=180); description: str=""; domain: str="General"; workspace_id: int | None = None
class ProjectPatch(BaseModel): name: str | None = Field(default=None,min_length=1,max_length=180); description: str | None = None; domain: str | None = Field(default=None,max_length=120)
class DatasetIn(BaseModel): name: str = Field(min_length=1,max_length=180); description: str = ""; kind: str="reference"; examples: list[dict] = Field(default_factory=list)
class PipelineIn(BaseModel): name: str = Field(default="Default pipeline", min_length=1, max_length=180); config: dict = Field(default_factory=dict)
class ExperimentIn(BaseModel): dataset_id: int; pipeline_id: int; split: str = "evaluation"
class DocumentIn(BaseModel): name: str = Field(min_length=1, max_length=255); content: str = Field(min_length=1, max_length=10_000_000)
class ExampleIn(BaseModel): question: str = Field(min_length=1); reference_answer: str | None=None; reference_contexts: list[str] = Field(default_factory=list); relevant_document_ids: list[str] = Field(default_factory=list); split: str = "evaluation"; metadata: dict = Field(default_factory=dict)
class PlaygroundIn(BaseModel): project_id: int; question: str = Field(min_length=1); pipeline_id: int | None=None
class CompareIn(BaseModel): experiment_ids: list[int] = Field(min_length=2,max_length=10); objective: str = "quality"; max_latency_ms: float | None = None; max_cost_usd: float | None = None; min_recall: float | None = None
class WorkspaceIn(BaseModel): name: str = Field(min_length=1, max_length=180)
class MemberIn(BaseModel): email: str = Field(min_length=3, max_length=320); role: str = "member"
class RoleIn(BaseModel): role: str
class ProviderKeyIn(BaseModel): api_key: str = Field(min_length=8, max_length=4096)


_rate_buckets: dict[str, deque[float]] = defaultdict(deque)
_rate_lock = __import__("threading").Lock()
_rate_limit = max(30, int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "300")))


@app.middleware("http")
async def api_rate_limit(request: Request, call_next):
    if request.url.path == "/api/health":
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"
    key = f"{client_ip}:{request.url.path.split('/')[2] if len(request.url.path.split('/')) > 2 else 'root'}"
    current = time.monotonic()
    with _rate_lock:
        bucket = _rate_buckets[key]
        while bucket and bucket[0] <= current - 60:
            bucket.popleft()
        if len(bucket) >= _rate_limit:
            return JSONResponse({"detail": "Rate limit exceeded. Try again shortly."}, status_code=429, headers={"Retry-After": "60"})
        bucket.append(current)
        if len(_rate_buckets) > 5000:
            for old_key in [name for name, events in _rate_buckets.items() if not events or events[-1] <= current - 60]:
                _rate_buckets.pop(old_key, None)
    return await call_next(request)


def _audit(db: Session, user: User | None, action: str, resource_type: str, resource_id: int | str | None = None, project: Project | None = None, metadata: dict | None = None) -> None:
    db.add(AuditEvent(user_id=user.id if user else None, workspace_id=project.workspace_id if project else None, project_id=project.id if project else None, action=action, resource_type=resource_type, resource_id=str(resource_id) if resource_id is not None else None, metadata_json=metadata or {}))


def _snapshot_dataset(db: Session, dataset: Dataset, user: User | None) -> DatasetVersion:
    rows = db.query(Example).filter_by(dataset_id=dataset.id).order_by(Example.id).all()
    snapshot = [{"id": row.id, "question": row.question, "reference_answer": row.reference_answer, "reference_contexts": row.reference_contexts or [], "relevant_document_ids": row.relevant_document_ids or [], "split": row.split, "metadata": row.metadata_json or {}} for row in rows]
    version = DatasetVersion(dataset_id=dataset.id, version=dataset.version, snapshot=snapshot, created_by=user.id if user else None)
    db.add(version)
    return version


def _provider_api_key(user: User, provider: str, db: Session) -> str | None:
    row = db.query(ProviderCredential).filter_by(user_id=user.id, provider=provider).first()
    return decrypt_secret(row.encrypted_value) if row else None


def _workspace_role(workspace_id: int, user: User, db: Session) -> str | None:
    membership = db.query(WorkspaceMember).filter_by(workspace_id=workspace_id, user_id=user.id).first()
    return membership.role if membership else None


def _default_workspace(user: User, db: Session) -> Workspace:
    workspace = db.query(Workspace).filter_by(owner_id=user.id).order_by(Workspace.id).first()
    if workspace:
        if not _workspace_role(workspace.id, user, db):
            db.add(WorkspaceMember(workspace_id=workspace.id, user_id=user.id, role="owner"))
        return workspace
    workspace = Workspace(name=f"{user.name or 'Personal'} workspace", owner_id=user.id)
    db.add(workspace)
    db.flush()
    db.add(WorkspaceMember(workspace_id=workspace.id, user_id=user.id, role="owner"))
    return workspace

def project_or_404(project_id: int, user: User, db: Session, *, write: bool = False, admin: bool = False):
    project = db.query(Project).filter_by(id=project_id).first()
    if not project:
        raise HTTPException(404, "Project not found")
    if project.workspace_id is None:
        role = "owner" if project.owner_id == user.id else None
    else:
        role = _workspace_role(project.workspace_id, user, db)
    if role is None:
        raise HTTPException(404, "Project not found")
    if admin and role not in {"owner", "admin"}:
        raise HTTPException(403, "Workspace owner or admin role required")
    if write and role not in {"owner", "admin", "member"}:
        raise HTTPException(403, "Workspace member role required")
    return project

@app.post("/api/auth/register")
@app.post("/api/auth/signup")
def register(payload: Credentials, db: Session=Depends(get_db)):
    email=payload.email.lower().strip()
    if db.query(User).filter_by(email=email).first(): raise HTTPException(409,"Email already registered")
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise HTTPException(422, "Enter a valid email address")
    user=User(email=email,name=payload.name.strip() or email.split("@")[0],password_hash=hash_password(payload.password)); db.add(user); db.flush(); _default_workspace(user, db); refresh=issue_refresh_token(); db.add(RefreshSession(user_id=user.id,token_hash=refresh_hash(refresh),expires_at=datetime.now(timezone.utc)+timedelta(days=30))); _audit(db,user,"auth.signup","user",user.id); db.commit(); db.refresh(user)
    return {"access_token":issue_token(user.id),"refresh_token":refresh,"token_type":"bearer","user":{"id":user.id,"email":user.email,"name":user.name}}

@app.post("/api/auth/login")
def login(payload: Login, db: Session=Depends(get_db)):
    user=db.query(User).filter_by(email=payload.email.lower().strip()).first()
    if not user or not verify_password(payload.password,user.password_hash): raise HTTPException(401,"Email or password is incorrect")
    refresh=issue_refresh_token();db.add(RefreshSession(user_id=user.id,token_hash=refresh_hash(refresh),expires_at=datetime.now(timezone.utc)+timedelta(days=30)));db.commit()
    return {"access_token":issue_token(user.id),"refresh_token":refresh,"token_type":"bearer","user":{"id":user.id,"email":user.email,"name":user.name}}

@app.post("/api/auth/refresh")
def refresh_session(payload:RefreshIn,db:Session=Depends(get_db)):
    session=db.query(RefreshSession).filter_by(token_hash=refresh_hash(payload.refresh_token),revoked_at=None).first()
    now=datetime.now(timezone.utc)
    expires = session.expires_at if session and session.expires_at.tzinfo else (session.expires_at.replace(tzinfo=timezone.utc) if session else now)
    if not session or expires < now: raise HTTPException(401,"Refresh token is invalid or expired")
    session.revoked_at=now;new_token=issue_refresh_token();db.add(RefreshSession(user_id=session.user_id,token_hash=refresh_hash(new_token),expires_at=now+timedelta(days=30)));db.commit()
    return {"access_token":issue_token(session.user_id),"refresh_token":new_token,"token_type":"bearer"}

@app.get("/api/auth/me")
def me(user: User=Depends(get_current_user)): return {"id":user.id,"email":user.email,"name":user.name}


@app.get("/api/settings/providers")
def provider_settings(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    stored = {row.provider: row for row in db.query(ProviderCredential).filter_by(user_id=user.id).all()}
    providers = ("openai", "anthropic", "google")
    env_names = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}
    return [{"provider": provider, "configured": provider in stored or bool(os.getenv(env_names[provider])), "source": "user" if provider in stored else ("backend" if os.getenv(env_names[provider]) else None), "key_hint": stored[provider].key_hint if provider in stored else None} for provider in providers]


@app.put("/api/settings/providers/{provider}")
def save_provider_key(provider: str, payload: ProviderKeyIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if provider not in {"openai", "anthropic", "google"}:
        raise HTTPException(422, "Provider must be openai, anthropic, or google")
    key = payload.api_key.strip()
    if not key or "\n" in key:
        raise HTTPException(422, "Provider key is invalid")
    row = db.query(ProviderCredential).filter_by(user_id=user.id, provider=provider).first()
    if row is None:
        row = ProviderCredential(user_id=user.id, provider=provider, encrypted_value=encrypt_secret(key), key_hint=key[-4:])
        db.add(row)
    else:
        row.encrypted_value = encrypt_secret(key)
        row.key_hint = key[-4:]
    _audit(db, user, "provider_key.save", "provider_credential", provider, metadata={"provider": provider})
    db.commit()
    return {"provider": provider, "configured": True, "key_hint": row.key_hint}


@app.delete("/api/settings/providers/{provider}")
def delete_provider_key(provider: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if provider not in {"openai", "anthropic", "google"}:
        raise HTTPException(422, "Provider must be openai, anthropic, or google")
    row = db.query(ProviderCredential).filter_by(user_id=user.id, provider=provider).first()
    if row:
        db.delete(row)
        _audit(db, user, "provider_key.delete", "provider_credential", provider, metadata={"provider": provider})
        db.commit()
    return {"provider": provider, "configured": bool(os.getenv({"openai":"OPENAI_API_KEY","anthropic":"ANTHROPIC_API_KEY","google":"GOOGLE_API_KEY"}[provider]))}


@app.get("/api/workspaces")
def workspaces(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    rows = db.query(Workspace, WorkspaceMember.role).join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id).filter(WorkspaceMember.user_id == user.id).order_by(Workspace.id).all()
    return [{"id": workspace.id, "name": workspace.name, "owner_id": workspace.owner_id, "role": role, "created_at": workspace.created_at} for workspace, role in rows]


@app.post("/api/workspaces")
def create_workspace(payload: WorkspaceIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    workspace = Workspace(name=payload.name.strip(), owner_id=user.id)
    db.add(workspace)
    db.flush()
    db.add(WorkspaceMember(workspace_id=workspace.id, user_id=user.id, role="owner"))
    _audit(db, user, "workspace.create", "workspace", workspace.id)
    db.commit()
    return {"id": workspace.id, "name": workspace.name, "role": "owner"}


@app.get("/api/workspaces/{workspace_id}/members")
def workspace_members(workspace_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    role = _workspace_role(workspace_id, user, db)
    if not role:
        raise HTTPException(404, "Workspace not found")
    rows = db.query(WorkspaceMember, User).join(User, User.id == WorkspaceMember.user_id).filter(WorkspaceMember.workspace_id == workspace_id).order_by(WorkspaceMember.id).all()
    return [{"id": member.id, "user_id": member.user_id, "name": member_user.name, "email": member_user.email, "role": member.role} for member, member_user in rows]


@app.post("/api/workspaces/{workspace_id}/members")
def add_workspace_member(workspace_id: int, payload: MemberIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    role = _workspace_role(workspace_id, user, db)
    if not role:
        raise HTTPException(404, "Workspace not found")
    if role not in {"owner", "admin"}:
        raise HTTPException(403, "Workspace owner or admin role required")
    if payload.role not in {"admin", "member", "viewer"}:
        raise HTTPException(422, "Role must be admin, member, or viewer")
    target = db.query(User).filter_by(email=payload.email.lower().strip()).first()
    if not target:
        raise HTTPException(404, "User must create an account before being added")
    existing = db.query(WorkspaceMember).filter_by(workspace_id=workspace_id, user_id=target.id).first()
    if existing:
        raise HTTPException(409, "User is already a workspace member")
    member = WorkspaceMember(workspace_id=workspace_id, user_id=target.id, role=payload.role)
    db.add(member)
    _audit(db, user, "workspace.member.add", "workspace_member", target.id)
    db.commit()
    return {"id": member.id, "user_id": target.id, "email": target.email, "role": member.role}


@app.patch("/api/workspaces/{workspace_id}/members/{member_id}")
def change_workspace_member_role(workspace_id: int, member_id: int, payload: RoleIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    role = _workspace_role(workspace_id, user, db)
    if role != "owner":
        raise HTTPException(403, "Only the workspace owner can change member roles")
    if payload.role not in {"admin", "member", "viewer"}:
        raise HTTPException(422, "Role must be admin, member, or viewer")
    member = db.query(WorkspaceMember).filter_by(id=member_id, workspace_id=workspace_id).first()
    if not member:
        raise HTTPException(404, "Workspace member not found")
    if member.role == "owner":
        raise HTTPException(409, "Transfer workspace ownership through workspace settings")
    member.role = payload.role
    _audit(db, user, "workspace.member.role.update", "workspace_member", member.id)
    db.commit()
    return {"id": member.id, "role": member.role}


@app.delete("/api/workspaces/{workspace_id}/members/{member_id}")
def remove_workspace_member(workspace_id: int, member_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    role = _workspace_role(workspace_id, user, db)
    if role not in {"owner", "admin"}:
        raise HTTPException(403, "Workspace owner or admin role required")
    member = db.query(WorkspaceMember).filter_by(id=member_id, workspace_id=workspace_id).first()
    if not member:
        raise HTTPException(404, "Workspace member not found")
    if member.role == "owner" or (role != "owner" and member.role == "admin"):
        raise HTTPException(403, "This member cannot be removed by your role")
    _audit(db, user, "workspace.member.remove", "workspace_member", member.id)
    db.delete(member)
    db.commit()
    return {"removed": True}


def normalize_pipeline_config(raw: dict) -> dict:
    forbidden = {"api_key", "openai_api_key", "anthropic_api_key", "google_api_key", "credential", "secret"}
    if forbidden & {str(key).lower() for key in raw}:
        raise HTTPException(422, "Provider credentials belong in Settings, not pipeline configuration")
    config = {"chunk_size": 500, "chunk_overlap": 60, "chunk_strategy": "fixed", "embedding_provider": "lexical", "embedding_model": "", "top_k": 5, "retrieval_strategy": "similarity", "reranker_enabled": False, "reranker_model": "Xenova/ms-marco-MiniLM-L-6-v2", "llm_provider": "extractive", "llm_model": "", "temperature": 0.0, "max_output_tokens": 512, "system_prompt": "", "user_prompt_template": "", "ragas_enabled": False, "evaluation_provider": "openai", "evaluation_model": "", **raw}
    try:
        config["chunk_size"] = int(config["chunk_size"])
        config["chunk_overlap"] = int(config["chunk_overlap"])
        config["top_k"] = int(config["top_k"])
        config["max_output_tokens"] = int(config["max_output_tokens"])
        config["temperature"] = float(config["temperature"])
    except (TypeError, ValueError):
        raise HTTPException(422, "Pipeline numeric settings must be valid numbers")
    if not 40 <= config["chunk_size"] <= 2000: raise HTTPException(422, "chunk_size must be between 40 and 2000 words")
    if not 0 <= config["chunk_overlap"] <= config["chunk_size"] // 2: raise HTTPException(422, "chunk_overlap must be no more than half the chunk size")
    if not 1 <= config["top_k"] <= 100: raise HTTPException(422, "top_k must be between 1 and 100")
    if not 16 <= config["max_output_tokens"] <= 32768: raise HTTPException(422, "max_output_tokens must be between 16 and 32768")
    if not 0 <= config["temperature"] <= 2: raise HTTPException(422, "temperature must be between 0 and 2")
    if config["chunk_strategy"] not in {"fixed", "sentence", "recursive", "semantic"}: raise HTTPException(422, "Unsupported chunk strategy")
    if config["embedding_provider"] not in {"lexical", "local", "openai", "google", "ollama"}: raise HTTPException(422, "Unsupported embedding provider")
    if config["retrieval_strategy"] not in {"similarity", "vector", "hybrid", "mmr", "hybrid_mmr"}: raise HTTPException(422, "Unsupported retrieval strategy")
    if config["llm_provider"] not in {"extractive", "openai", "anthropic", "google", "ollama"}: raise HTTPException(422, "Unsupported LLM provider")
    if config["llm_provider"] != "extractive" and not str(config["llm_model"]).strip(): raise HTTPException(422, "Set llm_model for the selected provider")
    if config["ragas_enabled"]:
        if config["evaluation_provider"] not in {"openai", "anthropic", "google", "ollama"}: raise HTTPException(422, "Choose a supported evaluation provider for RAGAS")
        if not str(config["evaluation_model"]).strip(): raise HTTPException(422, "Set evaluation_model when RAGAS is enabled")
    if config["chunk_strategy"] == "semantic" and config["embedding_provider"] == "lexical": raise HTTPException(422, "Semantic chunking requires embeddings")
    if config["retrieval_strategy"] in {"vector", "hybrid", "hybrid_mmr"} and config["embedding_provider"] == "lexical": raise HTTPException(422, "Vector and hybrid retrieval require an embedding provider")
    return config


def _bootstrap_ci(values: list[float], seed: int) -> dict | None:
    if len(values) < 2:
        return None
    generator = random.Random(seed)
    means = [statistics.fmean(generator.choices(values, k=len(values))) for _ in range(1000)]
    means.sort()
    return {"low_95": means[24], "high_95": means[974], "samples": len(values), "bootstrap_iterations": 1000}


def _compare_experiments(project_id: int, payload: CompareIn, db: Session, user: User) -> dict:
    project_or_404(project_id,user,db)
    if len(set(payload.experiment_ids)) != len(payload.experiment_ids):
        raise HTTPException(422,"experiment_ids must be unique")
    if payload.objective not in {"quality","retrieval","latency","cost","tradeoff"}:
        raise HTTPException(422,"objective must be quality, retrieval, latency, cost, or tradeoff")
    experiments=db.query(Experiment).filter(Experiment.project_id==project_id,Experiment.id.in_(payload.experiment_ids)).all()
    if len(experiments) != len(payload.experiment_ids):
        raise HTTPException(404,"One or more experiments were not found in this project")
    if any(row.status != "completed" for row in experiments):
        raise HTTPException(409,"All compared experiments must be completed")
    dataset_versions={(row.dataset_id,row.config_snapshot.get("dataset_version")) for row in experiments}
    if len(dataset_versions) != 1:
        raise HTTPException(422,"Experiments must use the same dataset and dataset version")
    metric_name={"quality":"answer_correctness_token_f1","retrieval":"ndcg_at_k","latency":"total_ms","cost":"cost_per_query_usd","tradeoff":"answer_correctness_token_f1"}[payload.objective]
    result=[]
    for experiment in experiments:
        metrics=experiment.metrics or {}
        value=metrics.get(metric_name)
        latency=metrics.get("total_ms")
        cost=metrics.get("cost_per_query_usd")
        recall=metrics.get("recall_at_k")
        eligible=(payload.max_latency_ms is None or latency is not None and latency <= payload.max_latency_ms) and (payload.max_cost_usd is None or cost is not None and cost <= payload.max_cost_usd) and (payload.min_recall is None or recall is not None and recall >= payload.min_recall)
        objective_score=value
        if payload.objective=="tradeoff":
            objective_score=(value/(cost if cost and cost>0 else 1.0)) if value is not None else None
        if payload.objective in {"latency","cost"} and value is not None:
            objective_score=-float(value)
        result.append({"id":experiment.id,"status":experiment.status,"metrics":metrics,"config":experiment.config_snapshot,"objective_metric":metric_name,"objective_score":objective_score,"eligible":eligible})
    winner=max((row for row in result if row["eligible"] and row["objective_score"] is not None),key=lambda row:row["objective_score"],default=None)
    metric_for_pairs=metric_name
    baseline_id=payload.experiment_ids[0]
    baseline_results={row.example_id:row.scores.get(metric_for_pairs) for row in db.query(Result).filter_by(experiment_id=baseline_id).all()}
    paired=[]
    for row in result:
        if row["id"] == baseline_id:
            continue
        compared={item.example_id:item.scores.get(metric_for_pairs) for item in db.query(Result).filter_by(experiment_id=row["id"]).all()}
        differences=[float(compared[key])-float(baseline_results[key]) for key in baseline_results.keys()&compared.keys() if baseline_results[key] is not None and compared[key] is not None]
        paired.append({"experiment_id":row["id"],"baseline_experiment_id":baseline_id,"metric":metric_for_pairs,"mean_difference":statistics.fmean(differences) if differences else None,"confidence_interval":_bootstrap_ci(differences,project_id*1_000_003+baseline_id*1009+row["id"]),"per_example_differences":differences})
    return {"objective":payload.objective,"metric":metric_name,"winner":winner["id"] if winner else None,"winner_reason":f"Highest eligible score for {payload.objective}; constraints applied before selection." if winner else "No experiment has a computed score that satisfies the selected constraints.","experiments":result,"paired_comparisons":paired,"dataset_version":next(iter(dataset_versions))[1],"constraints":{"max_latency_ms":payload.max_latency_ms,"max_cost_usd":payload.max_cost_usd,"min_recall":payload.min_recall}}

@app.get("/api/projects")
def projects(db: Session=Depends(get_db),user: User=Depends(get_current_user)):
    ids = [workspace_id for (workspace_id,) in db.query(WorkspaceMember.workspace_id).filter_by(user_id=user.id).all()]
    rows = db.query(Project).filter((Project.workspace_id.in_(ids)) | ((Project.workspace_id.is_(None)) & (Project.owner_id == user.id))).order_by(Project.id).all()
    return [{"id":p.id,"name":p.name,"description":p.description,"domain":p.domain,"workspace_id":p.workspace_id,"created_at":p.created_at} for p in rows]

@app.post("/api/demo/bootstrap")
def bootstrap_demo(db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    existing=db.query(Project).filter_by(owner_id=user.id,name="Financial Policy Demo (Synthetic)").first()
    if existing: return {"project_id":existing.id,"created":False,"message":"Synthetic demo already exists"}
    workspace = _default_workspace(user, db)
    project=Project(owner_id=user.id,workspace_id=workspace.id,name="Financial Policy Demo (Synthetic)",description="Reproducible synthetic finance policy corpus for exercising ingestion and evaluation.",domain="Finance · synthetic")
    db.add(project);db.flush()
    groups={
      "Deposits and transfers":["Personal savings account interest is credited on the last calendar day of each month.","A domestic wire transfer submitted before 2 PM local time is processed the same business day.","A standard ACH transfer usually arrives within two business days.","The minimum opening deposit for a standard savings account is 100 dollars.","Customers may make six outgoing transfers per statement cycle without a service charge.","Balances below 500 dollars incur a monthly maintenance fee of 4 dollars.","A returned deposit hold may last up to five business days.","A cashier check costs 8 dollars for standard account holders.","International wire transfers carry a 25 dollar outgoing fee.","Joint account owners have equal transaction and statement access."],
      "Cards and payments":["A replacement debit card is mailed within five to seven business days.","Customers must report a lost card immediately through the secure banking line.","Daily ATM cash withdrawals are limited to 600 dollars.","The contactless purchase limit is 150 dollars per transaction.","A foreign currency card purchase includes a 2 percent conversion fee.","A card dispute should be submitted within 60 days of the statement date.","Temporary digital card numbers are available after account verification.","A declined card authorization does not create a settled transaction.","The standard overdraft transfer fee is 10 dollars per covered item.","Merchant refunds may take up to ten business days to appear."],
      "Loans":["Personal loan applications require proof of income and identity verification.","The illustrative personal loan repayment term ranges from 12 to 60 months.","Extra principal payments on a fixed rate personal loan have no prepayment fee.","Mortgage applications require two recent pay statements and two years of tax forms.","The illustrative mortgage appraisal fee is 450 dollars.","A loan payment received after the 10th calendar day may incur a late fee.","Auto loan decisions are usually issued within two business days after a complete application.","The minimum illustrative auto loan amount is 5,000 dollars.","Cosigners share legal responsibility for repayment of the loan.","Loan payoff statements are available through the servicing portal."],
      "Security and privacy":["Customer support will never ask for a one time passcode by email.","Online banking sessions automatically sign out after 15 minutes of inactivity.","Customers should enable multifactor authentication for online access.","Suspected account fraud should be reported immediately using the number on the card.","Account statements are retained online for seven years.","Only verified account owners may request a change to a mailing address.","Secure messages are answered within two business days.","Paper statement delivery can be disabled in profile settings.","The bank does not send full account numbers in text messages.","A device change may trigger an additional identity verification step."],
      "Service and fees":["A standard checking account has no monthly fee with a 1,000 dollar average balance.","Paper statement delivery costs 2 dollars per month.","Domestic cashier checks are available at branches on business days.","Branch lobby hours are 9 AM to 5 PM Monday through Friday.","The customer service phone line operates from 7 AM to 9 PM local time daily.","Returned payment fees are capped at 25 dollars per item.","A dormant account is one with no customer initiated activity for 24 months.","Customers can update notification preferences in account settings.","A mailed address change may take three business days to process.","The bank observes federal holidays and branch hours may vary."],
    }
    examples=[]
    for title,facts in groups.items():
        doc=Document(project_id=project.id,name=f"Synthetic {title} Policy.txt",content="\n".join(facts));db.add(doc);db.flush()
        for fact in facts:
            question=f"According to the synthetic {title.lower()} policy, what is the rule for: {fact.split(' is ')[0].split(' are ')[0].split(' may ')[0].split(' must ')[0].split(' should ')[0].split(' has ')[0]}?"
            examples.append({"question":question,"reference_answer":fact,"relevant_document_ids":[str(doc.id)],"metadata":{"synthetic":True,"source":doc.name}})
    dataset=Dataset(project_id=project.id,name="Financial policies · 50 synthetic references",kind="reference");db.add(dataset);db.flush()
    for row in examples: db.add(Example(dataset_id=dataset.id,question=row["question"],reference_answer=row["reference_answer"],relevant_document_ids=row["relevant_document_ids"],metadata_json=row["metadata"]))
    for title,config in [("Compact · top 3",{"chunk_size":300,"chunk_overlap":40,"top_k":3}),("Balanced · top 5",{"chunk_size":500,"chunk_overlap":60,"top_k":5}),("Wide · top 5",{"chunk_size":700,"chunk_overlap":80,"top_k":5}),("Broad · top 10",{"chunk_size":500,"chunk_overlap":60,"top_k":10})]:
        db.add(Pipeline(project_id=project.id,name=title,config={**config,"retrieval":"lexical-overlap","generation":"extractive-local-baseline"}))
    db.commit();return {"project_id":project.id,"created":True,"synthetic":True,"documents":5,"examples":len(examples),"pipelines":4}

@app.post("/api/projects")
def create_project(payload: ProjectIn,db: Session=Depends(get_db),user: User=Depends(get_current_user)):
    data = payload.model_dump()
    workspace_id = data.pop("workspace_id")
    if workspace_id is None:
        workspace = _default_workspace(user, db)
    else:
        if _workspace_role(workspace_id, user, db) not in {"owner", "admin", "member"}:
            raise HTTPException(404, "Workspace not found")
        workspace = db.get(Workspace, workspace_id)
    p=Project(owner_id=user.id,workspace_id=workspace.id,**data);db.add(p);db.flush();_audit(db,user,"project.create","project",p.id,p);db.commit();db.refresh(p);return {"id":p.id,"name":p.name,"description":p.description,"domain":p.domain,"workspace_id":p.workspace_id}

@app.get("/api/projects/{project_id}")
def get_project(project_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project=project_or_404(project_id,user,db)
    role=_workspace_role(project.workspace_id,user,db) if project.workspace_id else "owner"
    return {"id":project.id,"name":project.name,"description":project.description,"domain":project.domain,"workspace_id":project.workspace_id,"role":role,"created_at":project.created_at}

@app.patch("/api/projects/{project_id}")
def update_project(project_id:int,payload:ProjectPatch,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project=project_or_404(project_id,user,db,admin=True)
    changes=payload.model_dump(exclude_unset=True)
    for key,value in changes.items():
        if value is not None: setattr(project,key,value.strip() if isinstance(value,str) else value)
    _audit(db,user,"project.update","project",project.id,project,{"fields":list(changes)})
    db.commit()
    return {"id":project.id,"name":project.name,"description":project.description,"domain":project.domain}

@app.delete("/api/projects/{project_id}")
def delete_project(project_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    p=project_or_404(project_id,user,db,admin=True);_audit(db,user,"project.delete","project",p.id,p);db.delete(p);db.commit();return {"deleted":True}

@app.get("/api/projects/{project_id}/summary")
def project_summary(project_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project_or_404(project_id,user,db)
    return {"datasets":db.query(Dataset).filter_by(project_id=project_id).count(),"documents":db.query(Document).filter_by(project_id=project_id).count(),"experiments":db.query(Experiment).filter_by(project_id=project_id).count(),"latest_experiments":[{"id":e.id,"status":e.status,"metrics":e.metrics} for e in db.query(Experiment).filter_by(project_id=project_id).order_by(Experiment.id.desc()).limit(10).all()]}

@app.get("/api/projects/{project_id}/audit-events")
def project_audit_events(project_id:int,limit:int=50,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project=project_or_404(project_id,user,db,admin=True)
    limit=max(1,min(limit,200))
    rows=db.query(AuditEvent).filter_by(project_id=project.id).order_by(AuditEvent.created_at.desc()).limit(limit).all()
    return [{"id":row.id,"user_id":row.user_id,"action":row.action,"resource_type":row.resource_type,"resource_id":row.resource_id,"metadata":row.metadata_json,"created_at":row.created_at} for row in rows]

@app.get("/api/dashboard/summary")
def dashboard_summary(db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    workspace_ids=[workspace_id for (workspace_id,) in db.query(WorkspaceMember.workspace_id).filter_by(user_id=user.id).all()]
    project_rows=db.query(Project).filter((Project.workspace_id.in_(workspace_ids)) | ((Project.workspace_id.is_(None)) & (Project.owner_id==user.id))).all()
    project_ids=[row.id for row in project_rows]
    if not project_ids: return {"projects":0,"experiments":0,"queries_evaluated":0,"average_faithfulness_proxy":None,"average_answer_relevancy_proxy":None,"average_latency_ms":None,"estimated_cost_usd":None,"recent_experiments":[],"quality_over_time":[]}
    experiments=db.query(Experiment).filter(Experiment.project_id.in_(project_ids)).order_by(Experiment.id.desc()).all()
    completed=[row for row in experiments if row.status=="completed"]
    results=db.query(Result).join(Experiment,Experiment.id==Result.experiment_id).filter(Experiment.project_id.in_(project_ids),Experiment.status=="completed").all()
    metric_values={"faithfulness":[],"relevancy":[],"latency":[]}
    for result in results:
        for target,key in [("faithfulness","faithfulness_overlap_proxy"),("relevancy","answer_relevancy_overlap_proxy")]:
            value=(result.scores or {}).get(key)
            if value is not None: metric_values[target].append(float(value))
        metric_values["latency"].append(float(result.latency_ms))
    known_costs=[float(result.cost_data["estimated_usd"]) for result in results if result.cost_data and result.cost_data.get("estimated_usd") is not None]
    all_costed=bool(results) and len(known_costs)==len(results)
    daily={}
    for experiment in completed:
        stamp=(experiment.completed_at or experiment.started_at)
        if not stamp: continue
        day=stamp.date().isoformat()
        bucket=daily.setdefault(day,{"date":day,"experiments":0,"answer_quality":[]})
        bucket["experiments"]+=1
        value=(experiment.metrics or {}).get("answer_correctness_token_f1")
        if value is not None: bucket["answer_quality"].append(float(value))
    quality=[{"date":day,"experiments":bucket["experiments"],"answer_correctness_token_f1":statistics.fmean(bucket["answer_quality"]) if bucket["answer_quality"] else None} for day,bucket in sorted(daily.items())]
    return {"projects":len(project_rows),"experiments":len(experiments),"queries_evaluated":len(results),"average_faithfulness_proxy":statistics.fmean(metric_values["faithfulness"]) if metric_values["faithfulness"] else None,"average_answer_relevancy_proxy":statistics.fmean(metric_values["relevancy"]) if metric_values["relevancy"] else None,"average_latency_ms":statistics.fmean(metric_values["latency"]) if metric_values["latency"] else None,"estimated_cost_usd":sum(known_costs) if all_costed else None,"recent_experiments":[{"id":e.id,"project_id":e.project_id,"status":e.status,"metrics":e.metrics,"completed_at":e.completed_at} for e in experiments[:10]],"quality_over_time":quality}

@app.get("/api/projects/{project_id}/datasets")
def datasets(project_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project_or_404(project_id,user,db)
    return [{"id":d.id,"name":d.name,"version":d.version,"kind":d.kind,"examples":db.query(Example).filter_by(dataset_id=d.id).count()} for d in db.query(Dataset).filter_by(project_id=project_id).all()]

@app.get("/api/datasets/{dataset_id}/versions")
def dataset_versions(dataset_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    dataset=db.get(Dataset,dataset_id)
    if not dataset: raise HTTPException(404,"Dataset not found")
    project_or_404(dataset.project_id,user,db)
    return [{"version":row.version,"example_count":len(row.snapshot),"created_at":row.created_at} for row in db.query(DatasetVersion).filter_by(dataset_id=dataset.id).order_by(DatasetVersion.version.desc()).all()]

@app.post("/api/datasets/{dataset_id}/versions")
def create_dataset_version(dataset_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    dataset=db.get(Dataset,dataset_id)
    if not dataset: raise HTTPException(404,"Dataset not found")
    project=project_or_404(dataset.project_id,user,db,write=True)
    existing=db.query(DatasetVersion).filter_by(dataset_id=dataset.id,version=dataset.version).first()
    if existing: return {"version":existing.version,"created":False,"example_count":len(existing.snapshot)}
    version=_snapshot_dataset(db,dataset,user);_audit(db,user,"dataset.version.create","dataset_version",version.version,project);db.commit()
    return {"version":version.version,"created":True,"example_count":len(version.snapshot)}

@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    d=db.get(Dataset,dataset_id)
    if not d: raise HTTPException(404,"Dataset not found")
    project=project_or_404(d.project_id,user,db,write=True)
    if db.query(Experiment).filter_by(dataset_id=d.id).count(): raise HTTPException(409,"Dataset is referenced by experiment history")
    _audit(db,user,"dataset.delete","dataset",d.id,project);db.delete(d);db.commit();return {"deleted":True}

@app.post("/api/projects/{project_id}/datasets")
def create_dataset(project_id:int,payload:DatasetIn,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project=project_or_404(project_id,user,db,write=True)
    if payload.kind not in ("reference","query_only"): raise HTTPException(422,"kind must be reference or query_only")
    d=Dataset(project_id=project_id,name=payload.name,description=payload.description,kind=payload.kind);db.add(d);db.flush()
    for row in payload.examples:
        if not row.get("question"): raise HTTPException(422,"Every dataset row needs a question")
        db.add(Example(dataset_id=d.id,question=str(row["question"]),reference_answer=row.get("reference_answer"),reference_contexts=row.get("reference_contexts",[]),relevant_document_ids=row.get("relevant_document_ids",[]),split=row.get("split","evaluation"),metadata_json=row.get("metadata",{})))
    db.flush();_snapshot_dataset(db,d,user);_audit(db,user,"dataset.create","dataset",d.id,project);db.commit();db.refresh(d);return {"id":d.id,"name":d.name,"version":d.version,"kind":d.kind,"examples":len(payload.examples)}

@app.post("/api/projects/{project_id}/datasets/import")
async def import_dataset(project_id:int,file:UploadFile=File(...),db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project=project_or_404(project_id,user,db,write=True);raw=(await file.read(10_000_001))
    if len(raw)>10_000_000: raise HTTPException(413,"Dataset file exceeds 10 MB")
    try:
        data=raw.decode("utf-8-sig")
        rows=json.loads(data) if file.filename.lower().endswith(".json") else [json.loads(line) for line in data.splitlines() if line.strip()] if file.filename.lower().endswith(".jsonl") else list(csv.DictReader(io.StringIO(data)))
    except Exception: raise HTTPException(400,"Could not parse CSV, JSON, or JSONL")
    if not isinstance(rows,list): raise HTTPException(400,"Dataset must be an array of records")
    ref=any(r.get("reference_answer") or r.get("answer") for r in rows)
    d=Dataset(project_id=project_id,name=file.filename or "Imported dataset",kind="reference" if ref else "query_only");db.add(d);db.flush()
    for r in rows:
        if r.get("question"):
            rel=r.get("relevant_document_ids",[])
            if isinstance(rel,str):
                try: rel=json.loads(rel)
                except Exception: rel=[x.strip() for x in rel.split(";") if x.strip()]
            contexts=r.get("reference_contexts",[])
            if isinstance(contexts,str):
                try: contexts=json.loads(contexts)
                except Exception: contexts=[contexts]
            split=r.get("split","evaluation")
            if split not in {"train","test","evaluation"}: split="evaluation"
            db.add(Example(dataset_id=d.id,question=str(r["question"]),reference_answer=r.get("reference_answer",r.get("answer")),reference_contexts=contexts,relevant_document_ids=rel,split=split,metadata_json=r.get("metadata",{})))
    db.flush();_snapshot_dataset(db,d,user);_audit(db,user,"dataset.import","dataset",d.id,project,{"examples":len(rows)});db.commit();return {"id":d.id,"name":d.name,"kind":d.kind,"examples":len(rows)}

@app.get("/api/datasets/{dataset_id}/examples")
def dataset_examples(dataset_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    d=db.get(Dataset,dataset_id)
    if not d: raise HTTPException(404,"Dataset not found")
    project_or_404(d.project_id,user,db)
    return [{"id":e.id,"question":e.question,"reference_answer":e.reference_answer,"reference_contexts":e.reference_contexts,"relevant_document_ids":e.relevant_document_ids,"split":e.split,"metadata":e.metadata_json} for e in db.query(Example).filter_by(dataset_id=d.id).all()]

@app.post("/api/datasets/{dataset_id}/examples")
def add_example(dataset_id:int,payload:ExampleIn,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    d=db.get(Dataset,dataset_id)
    if not d: raise HTTPException(404,"Dataset not found")
    project=project_or_404(d.project_id,user,db,write=True)
    if payload.split not in {"train","test","evaluation"}: raise HTTPException(422,"split must be train, test, or evaluation")
    e=Example(dataset_id=d.id,question=payload.question,reference_answer=payload.reference_answer,reference_contexts=payload.reference_contexts,relevant_document_ids=payload.relevant_document_ids,split=payload.split,metadata_json=payload.metadata);db.add(e);d.version+=1;db.flush();_snapshot_dataset(db,d,user);_audit(db,user,"dataset.example.add","example",e.id,project);db.commit();db.refresh(e);return {"id":e.id,"dataset_version":d.version}

@app.put("/api/examples/{example_id}")
def update_example(example_id:int,payload:ExampleIn,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    e=db.get(Example,example_id)
    if not e: raise HTTPException(404,"Example not found")
    d=db.get(Dataset,e.dataset_id);project=project_or_404(d.project_id,user,db,write=True)
    if payload.split not in {"train","test","evaluation"}: raise HTTPException(422,"split must be train, test, or evaluation")
    e.question=payload.question;e.reference_answer=payload.reference_answer;e.reference_contexts=payload.reference_contexts;e.relevant_document_ids=payload.relevant_document_ids;e.split=payload.split;e.metadata_json=payload.metadata;d.version+=1;db.flush();_snapshot_dataset(db,d,user);_audit(db,user,"dataset.example.update","example",e.id,project);db.commit();return {"id":e.id,"dataset_version":d.version}

@app.delete("/api/examples/{example_id}")
def delete_example(example_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    e=db.get(Example,example_id)
    if not e: raise HTTPException(404,"Example not found")
    d=db.get(Dataset,e.dataset_id);project=project_or_404(d.project_id,user,db,write=True);d.version+=1;db.flush();db.delete(e);db.flush();_snapshot_dataset(db,d,user);_audit(db,user,"dataset.example.delete","example",e.id,project);db.commit();return {"deleted":True,"dataset_version":d.version}

@app.get("/api/datasets/{dataset_id}/export")
def export_dataset(dataset_id:int,format:str="jsonl",db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    d=db.get(Dataset,dataset_id)
    if not d: raise HTTPException(404,"Dataset not found")
    project_or_404(d.project_id,user,db);rows=[{"question":e.question,"reference_answer":e.reference_answer,"reference_contexts":e.reference_contexts,"relevant_document_ids":e.relevant_document_ids,"split":e.split,"metadata":e.metadata_json} for e in db.query(Example).filter_by(dataset_id=d.id).all()]
    if format=="json": return rows
    if format=="csv":
        output=io.StringIO();writer=csv.DictWriter(output,fieldnames=["question","reference_answer","reference_contexts","relevant_document_ids","split","metadata"]);writer.writeheader()
        for row in rows: writer.writerow({**row,"reference_contexts":json.dumps(row["reference_contexts"],ensure_ascii=False),"relevant_document_ids":json.dumps(row["relevant_document_ids"],ensure_ascii=False),"metadata":json.dumps(row["metadata"],ensure_ascii=False)})
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(output.getvalue(),media_type="text/csv",headers={"Content-Disposition":f'attachment; filename="dataset-{d.id}.csv"'})
    if format!="jsonl": raise HTTPException(422,"format must be json, jsonl, or csv")
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(json.dumps(r,ensure_ascii=False) for r in rows),media_type="application/x-ndjson",headers={"Content-Disposition":f'attachment; filename="dataset-{d.id}.jsonl"'})

@app.get("/api/projects/{project_id}/documents")
def documents(project_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project_or_404(project_id,user,db);return [{"id":d.id,"name":d.name,"characters":len(d.content),"status":d.status,"created_at":d.created_at} for d in db.query(Document).filter_by(project_id=project_id).all()]

@app.post("/api/projects/{project_id}/documents")
def add_document(project_id:int,payload:DocumentIn,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project=project_or_404(project_id,user,db,write=True)
    digest=hashlib.sha256(payload.content.encode("utf-8")).hexdigest()
    existing=db.query(Document).filter_by(project_id=project_id,content_hash=digest).first()
    if existing: return {"id":existing.id,"name":existing.name,"characters":len(existing.content),"deduplicated":True}
    try: storage_path=store_original(project_id,payload.name,payload.content.encode("utf-8"),"text/plain; charset=utf-8")
    except Exception as exc: raise HTTPException(503,f"Object storage is unavailable: {exc}")
    d=Document(project_id=project_id,name=payload.name[:255],content=payload.content,content_hash=digest,storage_path=storage_path,status="ready");db.add(d);db.flush();_audit(db,user,"document.create","document",d.id,project,{"characters":len(d.content)});db.commit();db.refresh(d);return {"id":d.id,"name":d.name,"characters":len(d.content),"deduplicated":False}

@app.delete("/api/documents/{document_id}")
def delete_document(document_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    d=db.get(Document,document_id)
    if not d: raise HTTPException(404,"Document not found")
    project=project_or_404(d.project_id,user,db,write=True)
    try: delete_object(d.storage_path)
    except Exception as exc: raise HTTPException(503,f"Object storage is unavailable: {exc}")
    _audit(db,user,"document.delete","document",d.id,project);db.delete(d);db.commit();return {"deleted":True}

@app.post("/api/projects/{project_id}/documents/upload")
async def upload_document(project_id:int,file:UploadFile=File(...),db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project=project_or_404(project_id,user,db,write=True);raw=await file.read(10_000_001)
    if len(raw)>10_000_000: raise HTTPException(413,"Document exceeds 10 MB")
    name=file.filename or "upload.txt"; suffix=name.lower().rsplit(".",1)[-1]
    try:
        if suffix=="pdf":
            from pypdf import PdfReader
            text="\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
        elif suffix=="docx":
            from docx import Document as Docx
            text="\n".join(p.text for p in Docx(io.BytesIO(raw)).paragraphs)
        elif suffix in {"txt","md","csv","json","jsonl"}: text=raw.decode("utf-8-sig")
        else: raise HTTPException(415,"Supported document types: PDF, DOCX, TXT, Markdown, CSV, JSON, JSONL")
    except HTTPException: raise
    except Exception as exc: raise HTTPException(400,f"Document extraction failed: {exc}")
    if not text.strip(): raise HTTPException(422,"Document has no extractable text")
    digest=hashlib.sha256(text.encode("utf-8")).hexdigest()
    existing=db.query(Document).filter_by(project_id=project_id,content_hash=digest).first()
    if existing: return {"id":existing.id,"name":existing.name,"characters":len(existing.content),"deduplicated":True}
    try: storage_path=store_original(project_id,name,raw,file.content_type)
    except Exception as exc: raise HTTPException(503,f"Object storage is unavailable: {exc}")
    d=Document(project_id=project_id,name=name[:255],content=text,content_hash=digest,storage_path=storage_path,status="ready",metadata_json={"file_type":suffix,"original_size_bytes":len(raw)});db.add(d);db.flush();_audit(db,user,"document.upload","document",d.id,project,{"file_type":suffix,"original_size_bytes":len(raw)});db.commit();db.refresh(d);return {"id":d.id,"name":d.name,"characters":len(text),"deduplicated":False}

@app.get("/api/projects/{project_id}/pipelines")
def pipelines(project_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project_or_404(project_id,user,db);return [{"id":p.id,"name":p.name,"version":p.version,"config":p.config} for p in db.query(Pipeline).filter_by(project_id=project_id).all()]

@app.post("/api/projects/{project_id}/pipelines")
def create_pipeline(project_id:int,payload:PipelineIn,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project=project_or_404(project_id,user,db,write=True)
    cfg=normalize_pipeline_config(payload.config)
    p=Pipeline(project_id=project_id,name=payload.name,config=cfg);db.add(p);db.flush();_audit(db,user,"pipeline.create","pipeline",p.id,project);db.commit();db.refresh(p);return {"id":p.id,"name":p.name,"version":p.version,"config":p.config}

@app.patch("/api/pipelines/{pipeline_id}")
def update_pipeline(pipeline_id:int,payload:PipelineIn,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    pipeline=db.get(Pipeline,pipeline_id)
    if not pipeline: raise HTTPException(404,"Pipeline not found")
    project=project_or_404(pipeline.project_id,user,db,write=True)
    pipeline.name=payload.name
    pipeline.config=normalize_pipeline_config(payload.config)
    pipeline.version+=1
    _audit(db,user,"pipeline.update","pipeline",pipeline.id,project,{"version":pipeline.version})
    db.commit()
    return {"id":pipeline.id,"name":pipeline.name,"version":pipeline.version,"config":pipeline.config}

@app.delete("/api/pipelines/{pipeline_id}")
def delete_pipeline(pipeline_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    p=db.get(Pipeline,pipeline_id)
    if not p: raise HTTPException(404,"Pipeline not found")
    project=project_or_404(p.project_id,user,db,write=True)
    if db.query(Experiment).filter_by(pipeline_id=p.id).count(): raise HTTPException(409,"Pipeline is referenced by experiment history")
    _audit(db,user,"pipeline.delete","pipeline",p.id,project);db.delete(p);db.commit();return {"deleted":True}

@app.post("/api/projects/{project_id}/experiments")
def create_experiment(project_id:int,payload:ExperimentIn,background:BackgroundTasks,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project=project_or_404(project_id,user,db,write=True);d=db.query(Dataset).filter_by(id=payload.dataset_id,project_id=project_id).first();p=db.query(Pipeline).filter_by(id=payload.pipeline_id,project_id=project_id).first()
    if not d or not p: raise HTTPException(404,"Dataset or pipeline not found in this project")
    if payload.split not in {"train","test","evaluation"}: raise HTTPException(422,"split must be train, test, or evaluation")
    if db.query(DatasetVersion).filter_by(dataset_id=d.id,version=d.version).first() is None: _snapshot_dataset(db,d,user)
    e=Experiment(project_id=project_id,dataset_id=d.id,pipeline_id=p.id,owner_id=user.id,status="queued",config_snapshot={"dataset_version":d.version,"pipeline_version":p.version,"pipeline":p.config,"dataset_split":payload.split,"dataset_kind":d.kind,"engine":"raglab-pipeline-v2"});db.add(e);db.flush();_audit(db,user,"experiment.create","experiment",e.id,project);db.commit();db.refresh(e)
    try: execute_experiment.apply_async(args=[e.id],task_id=f"experiment-{e.id}")
    except Exception: background.add_task(run_experiment,e.id)
    return {"id":e.id,"status":e.status}

@app.get("/api/projects/{project_id}/experiments")
def experiments(project_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project_or_404(project_id,user,db);return [{"id":e.id,"dataset_id":e.dataset_id,"pipeline_id":e.pipeline_id,"status":e.status,"error":e.error,"metrics":e.metrics,"progress_total":e.progress_total,"progress_completed":e.progress_completed,"attempt_count":e.attempt_count,"started_at":e.started_at,"completed_at":e.completed_at} for e in db.query(Experiment).filter_by(project_id=project_id).order_by(Experiment.id.desc()).all()]

@app.get("/api/experiments/{experiment_id}")
def experiment_detail(experiment_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    experiment=db.get(Experiment,experiment_id)
    if not experiment: raise HTTPException(404,"Experiment not found")
    project_or_404(experiment.project_id,user,db)
    return {"id":experiment.id,"project_id":experiment.project_id,"dataset_id":experiment.dataset_id,"pipeline_id":experiment.pipeline_id,"status":experiment.status,"error":experiment.error,"metrics":experiment.metrics,"config_snapshot":experiment.config_snapshot,"progress_total":experiment.progress_total,"progress_completed":experiment.progress_completed,"attempt_count":experiment.attempt_count,"started_at":experiment.started_at,"completed_at":experiment.completed_at}

@app.post("/api/experiments/{experiment_id}/retry")
def retry_experiment(experiment_id:int,background:BackgroundTasks,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    original=db.get(Experiment,experiment_id)
    if not original: raise HTTPException(404,"Experiment not found")
    project=project_or_404(original.project_id,user,db,write=True)
    if original.status not in {"failed","cancelled"}: raise HTTPException(409,"Only failed or cancelled experiments can be retried")
    clone=Experiment(project_id=original.project_id,dataset_id=original.dataset_id,pipeline_id=original.pipeline_id,owner_id=user.id,status="queued",config_snapshot=dict(original.config_snapshot),max_attempts=original.max_attempts)
    db.add(clone);db.flush();_audit(db,user,"experiment.retry","experiment",clone.id,project,{"retry_of":original.id});db.commit();db.refresh(clone)
    try: execute_experiment.apply_async(args=[clone.id],task_id=f"experiment-{clone.id}")
    except Exception: background.add_task(run_experiment,clone.id)
    return {"id":clone.id,"retry_of":original.id,"status":clone.status}

@app.post("/api/experiments/{experiment_id}/cancel")
def cancel_experiment(experiment_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    e=db.get(Experiment,experiment_id)
    if not e: raise HTTPException(404,"Experiment not found")
    project=project_or_404(e.project_id,user,db,write=True)
    if e.status not in ("queued","running"): raise HTTPException(409,"Only queued or running experiments can be cancelled")
    e.cancel_requested=True
    if e.status == "queued":
        e.status="cancelled"
        e.completed_at=datetime.now(timezone.utc)
    _audit(db,user,"experiment.cancel","experiment",e.id,project);db.commit();return {"id":e.id,"status":e.status,"cancel_requested":e.cancel_requested}

@app.get("/api/experiments/{experiment_id}/results")
def experiment_results(experiment_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    e=db.get(Experiment,experiment_id)
    if not e: raise HTTPException(404,"Experiment not found")
    project_or_404(e.project_id,user,db);return [{"id":r.id,"example_id":r.example_id,"answer":r.answer,"retrieved":r.retrieved,"scores":r.scores,"evaluation_details":r.evaluation_details,"latency_ms":r.latency_ms,"latency_data":r.latency_data,"token_usage":r.token_usage,"cost_data":r.cost_data} for r in db.query(Result).filter_by(experiment_id=e.id).all()]

@app.get("/api/experiments/{experiment_id}/metrics")
def experiment_metrics(experiment_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    e=db.get(Experiment,experiment_id)
    if not e: raise HTTPException(404,"Experiment not found")
    project_or_404(e.project_id,user,db);return {"experiment_id":e.id,"status":e.status,"metrics":e.metrics,"config":e.config_snapshot}

@app.get("/api/experiments/{experiment_id}/traces")
def experiment_traces(experiment_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    e=db.get(Experiment,experiment_id)
    if not e: raise HTTPException(404,"Experiment not found")
    project_or_404(e.project_id,user,db);return [{"id":t.id,"result_id":t.result_id,"spans":t.spans,"error":t.error} for t in db.query(Trace).filter_by(experiment_id=e.id).all()]

@app.get("/api/traces/{trace_id}")
def get_trace(trace_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    trace=db.get(Trace,trace_id)
    if not trace: raise HTTPException(404,"Trace not found")
    exp=db.get(Experiment,trace.experiment_id);project_or_404(exp.project_id,user,db)
    result=db.get(Result,trace.result_id);example=db.get(Example,result.example_id)
    return {"id":trace.id,"experiment_id":exp.id,"question":example.question,"result":{"answer":result.answer,"retrieved":result.retrieved,"scores":result.scores,"token_usage":result.token_usage},"spans":trace.spans,"error":trace.error}

@app.get("/api/projects/{project_id}/compare")
def compare(project_id:int,ids:str,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project_or_404(project_id,user,db);wanted=[int(i) for i in ids.split(",") if i.isdigit()]
    return [{"id":e.id,"status":e.status,"metrics":e.metrics,"config":e.config_snapshot} for e in db.query(Experiment).filter(Experiment.project_id==project_id,Experiment.id.in_(wanted)).all()]

@app.post("/api/projects/{project_id}/compare")
def compare_post(project_id:int,payload:CompareIn,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    return _compare_experiments(project_id,payload,db,user)

@app.get("/api/experiments/{experiment_id}/recommendations")
def recommendations(experiment_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    e=db.get(Experiment,experiment_id)
    if not e: raise HTTPException(404,"Experiment not found")
    project_or_404(e.project_id,user,db)
    rows=db.query(Recommendation).filter_by(experiment_id=e.id).order_by(Recommendation.created_at.desc()).all()
    return {"recommendations":[{"id":r.id,"issue":r.issue,"recommendation":r.recommendation,"evidence":r.evidence,"confidence":r.confidence,"expected_tradeoff":r.expected_tradeoff,"next_experiment":r.next_experiment,"created_at":r.created_at} for r in rows],"basis":"Persisted deterministic rules referencing this experiment's actual metric values." if rows else "No recommendations generated yet. Use POST /api/experiments/{id}/recommendations/generate."}

@app.post("/api/experiments/{experiment_id}/recommendations/generate")
def generate_recommendations(experiment_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    experiment=db.get(Experiment,experiment_id)
    if not experiment: raise HTTPException(404,"Experiment not found")
    project=project_or_404(experiment.project_id,user,db,write=True)
    metrics=experiment.metrics or {}
    count=int(metrics.get("example_count",0) or 0)
    if experiment.status!="completed" or count==0:
        raise HTTPException(409,"Generate recommendations after a completed run with examples")
    proposals=[]
    recall=metrics.get("recall_at_k")
    if recall is not None and recall<0.6:
        proposals.append(("retrieval_recall","Labeled document recall is low; test a larger top_k and review chunk boundaries.",{"recall_at_k":recall,"top_k":experiment.config_snapshot.get("pipeline",{}).get("top_k")},"A larger candidate set can improve coverage while increasing retrieval and context cost.","Clone this pipeline with top_k increased by 2 and compare on the same dataset version."))
    precision=metrics.get("precision_at_k")
    if precision is not None and precision<0.5:
        proposals.append(("retrieval_precision","Many retrieved chunks are outside the labeled relevant-document set; compare hybrid ranking or a cross-encoder reranker.",{"precision_at_k":precision,"retrieval_method":metrics.get("retrieval_method")},"Reranking adds latency and model execution cost.","Enable hybrid retrieval or reranking, then compare precision@K and P95 latency."))
    correctness=metrics.get("answer_correctness_token_f1")
    if correctness is not None and correctness<0.35:
        proposals.append(("answer_overlap","Token overlap with reference answers is low; inspect per-example traces to separate retrieval gaps from answer-generation gaps.",{"answer_correctness_token_f1":correctness,"answer_provider":experiment.config_snapshot.get("pipeline",{}).get("llm_provider","extractive")},"Reference token overlap is a lexical proxy and does not establish semantic correctness.","Review the lowest-scoring examples before changing the prompt or model."))
    p95=metrics.get("latency_p95_ms")
    if p95 is not None and p95>3000:
        proposals.append(("latency_p95","The measured P95 query latency exceeds three seconds; inspect stage spans to find the slow component.",{"latency_p95_ms":p95,"retrieval_latency_avg_ms":metrics.get("retrieval_latency_avg_ms"),"generation_latency_avg_ms":metrics.get("generation_latency_avg_ms")},"Reducing latency may trade away context coverage or answer quality.","Change only the slowest stage setting and rerun the same evaluation split."))
    confidence=min(0.85,0.4+count/100)
    for issue,message,evidence,tradeoff,next_run in proposals:
        row=db.query(Recommendation).filter_by(experiment_id=experiment.id,issue=issue).first()
        if row is None:
            row=Recommendation(experiment_id=experiment.id,issue=issue,recommendation=message,evidence=evidence,confidence=confidence,expected_tradeoff=tradeoff,next_experiment=next_run)
            db.add(row)
        else:
            row.recommendation=message;row.evidence=evidence;row.confidence=confidence;row.expected_tradeoff=tradeoff;row.next_experiment=next_run
    _audit(db,user,"recommendations.generate","experiment",experiment.id,project,{"recommendation_count":len(proposals)})
    db.commit()
    return recommendations(experiment_id,db,user)

@app.post("/api/playground/query")
def playground_query(payload:PlaygroundIn,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project=project_or_404(payload.project_id,user,db,write=True);pipeline=None
    if payload.pipeline_id:
        pipeline=db.query(Pipeline).filter_by(id=payload.pipeline_id,project_id=payload.project_id).first()
        if not pipeline: raise HTTPException(404,"Pipeline not found")
    cfg=pipeline.config if pipeline else normalize_pipeline_config({})
    pipeline_id=pipeline.id if pipeline else None
    pipeline_version=pipeline.version if pipeline else 1
    embedding_api_key=_provider_api_key(user,str(cfg.get("embedding_provider","")),db)
    llm_api_key=_provider_api_key(user,str(cfg.get("llm_provider","")),db)
    docs=db.query(Document).filter_by(project_id=payload.project_id,status="ready").all()
    started=time.perf_counter()
    index=build_or_load_index(db,project.id,docs,cfg,pipeline_id,pipeline_version,embedding_api_key)
    index_ready_ms=(time.perf_counter()-started)*1000
    embedding_started=time.perf_counter()
    query_embedding=embed_texts([payload.question],cfg,embedding_api_key)[0]
    embedding_ms=(time.perf_counter()-embedding_started)*1000
    retrieval_started=time.perf_counter()
    top_k=max(1,min(int(cfg.get("top_k",5)),100))
    strategy=str(cfg.get("retrieval_strategy","similarity"))
    vector_order=vector_search(db,project.id,pipeline_id,pipeline_version,embedding_key(cfg),query_embedding, max(top_k*5,50))
    ranked=rank(index,payload.question,top_k,strategy,query_embedding or None,vector_order=vector_order)
    retrieval_ms=(time.perf_counter()-retrieval_started)*1000
    rerank_ms=0.0
    if cfg.get("reranker_enabled") and cfg.get("reranker_model"):
        rerank_started=time.perf_counter()
        ranked=cross_encoder_rerank(payload.question,ranked,str(cfg["reranker_model"]),top_k)
        rerank_ms=(time.perf_counter()-rerank_started)*1000
    context="\n\n".join(c["text"] for c in ranked)
    generation_started=time.perf_counter()
    answer,token_usage,provider_label=generate_answer(payload.question,context,cfg,llm_api_key)
    generation_ms=(time.perf_counter()-generation_started)*1000
    total_ms=(time.perf_counter()-started)*1000
    latency_data={"index_ready_ms":round(index_ready_ms,3),"query_embedding_ms":round(embedding_ms,3),"retrieval_ms":round(retrieval_ms,3),"reranking_ms":round(rerank_ms,3),"generation_ms":round(generation_ms,3),"total_ms":round(total_ms,3)}
    provider,_,model=provider_label.partition(":")
    cost={"estimated_usd":0.0,"status":"zero-cost-local","pricing_version":"not-applicable"} if provider_label=="extractive-local-baseline" else estimate_cost(provider,model,token_usage)
    if str(cfg.get("embedding_provider","lexical")) in {"openai","google","ollama"}:
        cost["embedding_cost_status"]="embedding_token_usage_unavailable";cost["total_cost_status"]="partial_generation_only"
    elif cost.get("status")=="estimated":
        cost["total_cost_status"]="complete_for_generation_and_local_embeddings"
    spans=[{"name":"index","duration_ms":latency_data["index_ready_ms"],"embedding_model":embedding_key(cfg)}, {"name":"query_embedding","duration_ms":latency_data["query_embedding_ms"],"embedding_model":embedding_key(cfg)}, {"name":"retrieval","duration_ms":latency_data["retrieval_ms"],"strategy":strategy,"top_k":top_k,"retrieved_count":len(ranked)}, {"name":"reranking","duration_ms":latency_data["reranking_ms"],"model":cfg.get("reranker_model") if cfg.get("reranker_enabled") else None}, {"name":"generation","duration_ms":latency_data["generation_ms"],"provider_model":provider_label,"token_usage":token_usage}]
    external_trace_id=export_spans("raglab.playground.query",spans,{"project_id":project.id,"provider_model":provider_label,"retrieved_count":len(ranked)})
    trace=PlaygroundTrace(project_id=project.id,user_id=user.id,question=payload.question,answer=answer,retrieved=[{key:value for key,value in row.items() if key!="embedding"} for row in ranked],spans=spans,latency_data=latency_data,token_usage=token_usage,external_trace_id=external_trace_id)
    db.add(trace);_audit(db,user,"playground.query","playground_trace",None,project,{"provider_model":provider_label});db.commit();db.refresh(trace)
    return {"answer":answer,"retrieved":[{key:value for key,value in row.items() if key!="embedding"} for row in ranked],"latency_ms":total_ms,"latency_data":latency_data,"token_usage":token_usage,"cost":cost,"trace_id":trace.id,"phoenix_trace_id":external_trace_id,"engine":f"{strategy} retrieval + {provider_label} generation","caveat":"Overlap metrics are lexical proxies; trace spans and retrieved context are inspectable."}

@app.get("/api/projects/{project_id}/playground/traces")
def playground_traces(project_id:int,limit:int=50,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    project_or_404(project_id,user,db)
    limit=max(1,min(limit,100))
    return [{"id":row.id,"question":row.question,"answer":row.answer,"latency_data":row.latency_data,"created_at":row.created_at} for row in db.query(PlaygroundTrace).filter_by(project_id=project_id).order_by(PlaygroundTrace.id.desc()).limit(limit).all()]

@app.get("/api/playground/traces/{trace_id}")
def playground_trace(trace_id:int,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    trace=db.get(PlaygroundTrace,trace_id)
    if not trace: raise HTTPException(404,"Trace not found")
    project_or_404(trace.project_id,user,db)
    return {"id":trace.id,"question":trace.question,"answer":trace.answer,"retrieved":trace.retrieved,"spans":trace.spans,"latency_data":trace.latency_data,"token_usage":trace.token_usage,"created_at":trace.created_at}

@app.post("/api/auth/logout")
def logout(payload:RefreshIn,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    session=db.query(RefreshSession).filter_by(token_hash=refresh_hash(payload.refresh_token),user_id=user.id,revoked_at=None).first()
    if session: session.revoked_at=datetime.now(timezone.utc);db.commit()
    return {"logged_out":True}


@app.get("/api/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "raglab-api", "version": app.version, "database": "connected"}
