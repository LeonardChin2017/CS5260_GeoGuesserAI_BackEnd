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
| `/api/user/gemini-key` | GET | Check if user has a key saved. Response `{ hasKey: true|false }` (Clerk auth required). |
| `/api/user/gemini-key` | PUT | Save user's Gemini API key (Clerk auth required). Body `{ apiKey }`. Key stored encrypted in DB. |
| `/api/user/gemini-key` | DELETE | Clear user's saved Gemini API key (Clerk auth required). |
| `/api/user/chat` | GET | Get user's chat history. Response `{ messages: Message[] }` (Clerk auth required). |
| `/api/user/chat` | POST | Save user's chat history. Body `{ messages: Message[] }` (Clerk auth required). |
| `/api/user/chat` | DELETE | Clear user's chat history (Clerk auth required). |
| `/api/chat` | POST | Chat: with Clerk auth uses key from DB; otherwise body `{ message, apiKey? }`. Response `{ reply }`. |
| `/api/resume/upload` | POST | Resume upload: multipart file, response `{ ok, savedPath, originalName }` (see **Resume flow** below). |
| `/api/resume` | GET | Get current user's resume info. Response `{ fileName, originalName, uploadedAt }` (Clerk required). |
| `/api/resume` | DELETE | Delete current user's resume (Clerk required). |
| `/api/user/links` | GET | Get current user's links. Response `{ links: string[] }` (Clerk required). |
| `/api/user/links` | PUT | Save current user's links. Body `{ links: string[] }` (Clerk required). Stored only; see **Links (placeholder)** below. |

---

## Resume: frontend → backend, storage, and agent

- **How the frontend sends the resume**  
  The frontend (e.g. `ResumeUpload`) sends the resume as a **multipart form** to `POST /api/resume/upload`:
  - Field name: `file` (single PDF file).
  - When Clerk is enabled, the request includes `Authorization: Bearer <Clerk JWT>` so the backend can associate the upload with the signed-in user.

- **Where the backend stores it**  
  - **Files:** On disk under `Backend/uploads/`. For signed-in users, files are under `uploads/<userId>/` (one folder per user). Filename is `timestamp-originalName.pdf`.
  - **Database:** Table `user_resume` in `data/keys.db` (SQLite). Columns: `user_id`, `file_path`, `original_name`, `uploaded_at`, `extracted_profile` (JSON), `resume_text` (raw text from PDF). One row per user; a new upload replaces the previous one.

- **How the backend agent uses it**  
  When the user chats via `POST /api/chat`, the backend loads that user’s `extracted_profile` and `resume_text` from `user_resume`, builds a context string with `buildUserProfileContext()`, and injects it into the **Gemini system prompt**. The agent is instructed to answer questions about the user (name, title, experience, skills, etc.) from this resume content only.

- **Known limitation**  
  There is a **pending fix**: resume info is not always retrieved correctly (extraction/parsing). See TODOs in `resume-extract.js` and the resume upload handler in `index.js`.

- **Resume extraction activity (output history)**  
  The backend stores the resume extraction “output history” (Orchestrator → Parser agent → Profile agent → Preferences agent → Ready for follow-up) in `user_resume.activity_steps` (JSON). It is returned as `activitySteps` from **GET /api/resume** and from **POST /api/resume/upload** when extraction completes. The frontend fetches and displays this; it does not build or store the history locally.

---

## Links: frontend → backend and placeholder for extraction

- **How the frontend sends links**  
  The frontend sends the user’s links (e.g. LinkedIn, portfolio, GitHub) to the backend so the agent can use them as extra context:
  - **GET /api/user/links** – Returns `{ links: string[] }` for the current user (Clerk required). The Dashboard loads these on mount and merges with local state.
  - **PUT /api/user/links** – Body `{ links: string[] }` (array of URL strings). The Dashboard (and any profile links UI) should call this when the user adds/edits/removes links (e.g. on blur or save). Clerk auth required.

- **Where the backend stores links**  
  Table `user_links` in `data/keys.db`: `user_id`, `links` (JSON array of strings), `updated_at`. One row per user; PUT overwrites the list (max 20 URLs, non-empty strings only).

- **Backend placeholder (for teammate)**  
  The backend **does not** yet use links for the agent. It only stores and returns them. A teammate should:
  1. In the chat flow (where `userProfileContext` is built), load the user’s links from `user_links`.
  2. Implement a way to **extract information** from each URL (e.g. fetch and parse LinkedIn profile, portfolio page, or other public pages within ToS and rate limits).
  3. Append that extracted data to the context passed to the agent (e.g. into the system prompt or a dedicated “user links context” block) so the agent can use it as a data resource when answering the user.

---

## Scripts

From the repo root (folder that contains `Backend` and `Frontend`):

| Script | Description |
|--------|-------------|
| `./run-backend.sh` | Install deps and start backend (port 3001) |
| `./run-frontend.sh` | Install deps and start frontend dev server |

Use two terminals to run backend and frontend separately so you can see backend logs when using chat.
