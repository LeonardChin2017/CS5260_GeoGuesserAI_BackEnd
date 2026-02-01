# jobAI Backend

API backend for the jobAI frontend: health check, Gemini-powered chat, resume upload, and encoded API key for the frontend.

---

## Quick start (local)

```bash
cd Backend
cp .env.example .env
# Edit .env: set GEMINI_API_KEY and GEMINI_KEY_ENCODING_SECRET (see below)
npm install
npm start
```

Runs at **http://localhost:3001**. Point the frontend to it with `VITE_API_URL=http://localhost:3001` in Frontend `.env`.

---

## Environment variables

Copy `.env.example` to `.env` and set:

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes (for chat) | [Google AI Studio](https://aistudio.google.com/apikey) API key |
| `GEMINI_KEY_ENCODING_SECRET` | Yes (for GET /api/gemini-key) | Min 16 chars. Generate: `node -e "console.log(require('crypto').randomBytes(32).toString('base64'))"` |
| `PORT` | No | Default 3001 |
| `GEMINI_MODEL` | No | Default gemini-2.5-flash |

Never commit `.env`; it is in `.gitignore`.

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
| `/api/gemini-key` | GET | Returns encrypted Gemini key for frontend (when backend has key + encoding secret) |
| `/api/chat` | POST | Chat: body `{ message, apiKey? }` or `{ message, encodedKey? }`, response `{ reply }` |
| `/api/resume/upload` | POST | Resume upload: multipart file, response `{ ok, savedPath, originalName }` |

---

## Scripts

From the repo root (folder that contains `Backend` and `Frontend`):

| Script | Description |
|--------|-------------|
| `./run-backend.sh` | Install deps and start backend (port 3001) |
| `./run-frontend.sh` | Install deps and start frontend dev server |

Use two terminals to run backend and frontend separately so you can see backend logs when using chat.
