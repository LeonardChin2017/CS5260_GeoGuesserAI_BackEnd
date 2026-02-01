# jobAI Backend

API backend for the jobAI frontend: health check, Gemini-powered chat, resume upload, and **per-user Gemini API key** (user enters key in Settings → stored encrypted in DB; no LLM key in server env on production).

---

## Quick start (local)

```bash
cd Backend
cp .env.example .env
# Edit .env: set CLERK_SECRET_KEY and ENCRYPTION_SECRET (see below)
npm install
npm start
```

Runs at **http://localhost:3001**. Point the frontend to it with `VITE_API_URL=http://localhost:3001` in Frontend `.env`.

---

## Environment variables

Copy `.env.example` to `.env` and set:

| Variable | Required | Description |
|----------|----------|-------------|
| `CLERK_SECRET_KEY` | Yes (for per-user key storage) | Clerk **Secret** Key from [dashboard](https://dashboard.clerk.com) → API Keys |
| `CLERK_PUBLISHABLE_KEY` | Yes (when using Clerk) | Clerk **Publishable** Key (same app as frontend). Backend needs it to validate requests. |
| `ENCRYPTION_SECRET` | Yes (when using Clerk) | Min 16 chars; encrypts Gemini keys at rest in DB. Generate: `node -e "console.log(require('crypto').randomBytes(32).toString('base64'))"` |
| `GEMINI_API_KEY` | No | Only for fallback when Clerk is not configured (key in request body or env) |
| `PORT` | No | Default 3001 |
| `GEMINI_MODEL` | No | Default gemini-2.5-flash |

Never commit `.env`; it is in `.gitignore`. User Gemini keys are stored in SQLite under `data/` (encrypted).

**Production (with Clerk):** You do **not** store any Gemini/LLM API key in env on the server. Users enter their key in the app (Settings); the backend stores it encrypted per user in the DB and uses it for chat based on Clerk auth. Required in env: `CLERK_SECRET_KEY`, `CLERK_PUBLISHABLE_KEY`, and `ENCRYPTION_SECRET`.

---

## Deploy to Digital Ocean

See **[DEPLOY-DIGITALOCEAN.md](./DEPLOY-DIGITALOCEAN.md)** for step-by-step setup:

- **App Platform** – Connect repo, set source directory to `Backend`, add env vars, deploy.
- **Droplet with Docker** – Build and run the included Dockerfile.
- **Droplet with Node** – Clone, `npm ci`, set `.env`, run with PM2.

After deploy, set the frontend `VITE_API_URL` to your backend URL (e.g. `https://your-app.ondigitalocean.app`).

---

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/hello` | GET | Test message |
| `/api/user/gemini-key` | PUT | Save user's Gemini API key (Clerk auth required). Body `{ apiKey }`. Key stored encrypted in DB. |
| `/api/user/gemini-key` | DELETE | Clear user's saved Gemini API key (Clerk auth required). |
| `/api/chat` | POST | Chat: with Clerk auth uses key from DB; otherwise body `{ message, apiKey? }`. Response `{ reply }`. |
| `/api/resume/upload` | POST | Resume upload: multipart file, response `{ ok, savedPath, originalName }` |

---

## Scripts

From the repo root (folder that contains `Backend` and `Frontend`):

| Script | Description |
|--------|-------------|
| `./run-backend.sh` | Install deps and start backend (port 3001) |
| `./run-frontend.sh` | Install deps and start frontend dev server |

Use two terminals to run backend and frontend separately so you can see backend logs when using chat.
