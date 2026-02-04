# jobAI Backend (Python)

API backend for the frontend: health check, LLM chat, question formatting, and temporary PDF generation.

---

## Quick start (local)

```bash
cd Backend
cp .env.example .env
python -m venv .venv
. .venv/bin/activate   # Windows: .venv/Scripts/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 3001
```

Runs at **http://localhost:3001**. Point the frontend to it with `VITE_API_URL=http://localhost:3001` in Frontend `.env`.

---

## Environment variables

Copy `.env.example` to `.env` and set:

| Variable | Required | Description |
|----------|----------|-------------|
| `CLERK_SECRET_KEY` | Optional | Clerk Secret Key (if you use Clerk auth) |
| `CLERK_PUBLISHABLE_KEY` | Optional | Clerk Publishable Key |
| `ENCRYPTION_SECRET` | Optional | Min 16 chars; encrypts API keys at rest in DB. Generate: `python -c "import secrets;print(secrets.token_urlsafe(32))"` |
| `GEMINI_API_KEY` | No | Fallback API key if not provided by client |
| `DEEPSEEK_API_KEY` | No | Fallback API key if not provided by client |
| `PORT` | No | Default 3001 |
| `GEMINI_MODEL` | No | Default gemini-2.5-flash |
| `DEEPSEEK_MODEL` | No | Default deepseek-chat |

Never commit `.env`; it is in `.gitignore`.

---

## Deploy to Digital Ocean

See **[DEPLOY-DIGITALOCEAN.md](./DEPLOY-DIGITALOCEAN.md)** for step-by-step setup.

---

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/user/gemini-key` | GET/PUT/DELETE | Manage Gemini API key |
| `/api/user/deepseek-key` | GET/PUT/DELETE | Manage DeepSeek API key |
| `/api/user/llm-provider` | GET/PUT | Set provider |
| `/api/user/chat` | GET/POST/DELETE | Chat history |
| `/api/chat` | POST | Chat: returns `reply`, `questionList`, `styleUpdate`, `pdfUrl` |
| `/generated_papers/<file>.pdf` | GET | Download generated PDFs |

---

## Scripts

From the repo root (folder that contains `Backend` and `Frontend`):

| Script | Description |
|--------|-------------|
| `./run-backend.sh` | Install deps and start backend (port 3001) |
| `./run-frontend.sh` | Install deps and start frontend dev server |
