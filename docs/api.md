# API reference

Base path: `/api`. Interactive OpenAPI is available at `/docs`.

## Authentication

- `POST /auth/register`, `POST /auth/login` return access and refresh tokens.
- `POST /auth/refresh` rotates the refresh token and returns a new pair.
- `POST /auth/logout` revokes the supplied refresh token. Access tokens expire after 15 minutes.
- Send `Authorization: Bearer <access_token>` to protected routes.

## Projects and data

- `GET/POST /projects`, `DELETE /projects/{id}`, `GET /projects/{id}/summary`
- `GET/POST /projects/{id}/datasets`; `POST /projects/{id}/datasets/import` accepts CSV, JSON, JSONL.
- `GET /datasets/{id}/examples`, `POST /datasets/{id}/examples`, `PUT/DELETE /examples/{id}`, `GET /datasets/{id}/export?format=json|jsonl|csv`, `DELETE /datasets/{id}`
- `GET/POST /projects/{id}/documents`; `POST /projects/{id}/documents/upload`; `DELETE /documents/{id}`
- `GET/POST /projects/{id}/pipelines`; `DELETE /pipelines/{id}`

## Execution and observability

- `POST/GET /projects/{id}/experiments`, `POST /experiments/{id}/cancel`
- `GET /experiments/{id}/metrics`, `GET /experiments/{id}/results`, `GET /experiments/{id}/traces`, `GET /experiments/{id}/recommendations`, `GET /traces/{trace_id}`
- `GET/POST /projects/{id}/compare`
- `POST /playground/query`
- `POST /demo/bootstrap` creates the explicitly synthetic 50-example demo for the current account.
- `GET /health`

All project reads and mutations currently enforce owner scoping. Workspace roles, pagination, rate limiting, and production-grade audit logging are outstanding.
