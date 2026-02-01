# jobAI Backend

Backend API for the jobAI frontend: health checks, optional LLM-powered chat, and resume upload/extraction.

---

## Where to add your DeepSeek API key

1. **Create a `.env` file in the Backend folder** (same folder as `index.js`):
   ```bash
   cd Backend
   cp .env.example .env
   ```
   (On Windows: `copy .env.example .env`.)

2. **Edit `Backend/.env`** and set your key (no quotes needed):
   ```env
   DEEPSEEK_API_KEY=sk-your-actual-deepseek-api-key-here
   ```

3. **Use it in code** when you add `/api/chat` or resume extraction. For example in Node:
   ```js
   const apiKey = process.env.DEEPSEEK_API_KEY;
   if (!apiKey) {
     console.warn('DEEPSEEK_API_KEY not set – LLM features will not work');
   }
   // Then pass apiKey to your DeepSeek HTTP client / SDK
   ```

4. **Never commit `.env`** – it is listed in `Backend/.gitignore`. Only commit `.env.example` (without real keys).

5. **Load `.env` in Node:** By default Node does not read `.env`. When you add LLM routes, install `dotenv` and at the top of `index.js` add:
   ```js
   require('dotenv').config();
   ```
   Then `process.env.DEEPSEEK_API_KEY` will be set.

Get a DeepSeek API key: https://platform.deepseek.com/

---

## Quick setup (run backend so frontend can use it)

### 1. Install dependencies

From this folder (`Backend`):

```bash
cd Backend
npm install
```

### 2. Start the backend

```bash
npm run dev
```

The server runs at **http://localhost:3001**. You should see:

```
jobAI backend running at http://localhost:3001
```

### 3. Point the frontend to this backend

In the **Frontend** folder:

1. Copy `.env.example` to `.env` (if you don’t have `.env` yet):
   ```bash
   cd ../Frontend
   copy .env.example .env
   ```
   (On macOS/Linux use `cp .env.example .env`.)

2. In `.env`, set the backend URL (no trailing slash):
   ```
   VITE_API_URL=http://localhost:3001
   ```

3. Start the frontend:
   ```bash
   npm run dev
   ```

4. In the app, check the **“Backend: Connected”** status (e.g. in the sidebar). If it says **Connected**, the frontend is talking to this backend.

---

## If the backend doesn't print anything (frontend–backend not talking)

The frontend **only calls the backend when `VITE_API_URL` is set**. If the backend terminal never shows `>>> Incoming: POST /api/chat` when you send a chat message, the frontend is not hitting the backend.

**Checklist:**

1. **Frontend `.env`** – In the **Frontend** folder, create or edit `.env` (copy from `.env.example` if needed) and set:
   ```env
   VITE_API_URL=http://localhost:3001
   ```
   No trailing slash, no quotes.

2. **Restart the frontend dev server** – Vite reads `.env` only at startup. After changing `.env`, stop the frontend (Ctrl+C) and run `./run-frontend.sh` (or `npm run dev` in Frontend) again.

3. **Check the UI** – In the app sidebar you should see **Backend: Connected**. If you see **Backend: Not configured (set VITE_API_URL)** or **Disconnected**, the frontend is not talking to the backend.

4. **Check the browser console** – Open DevTools (F12) → Console. When you send a chat message you should see either:
   - `[Chat] Calling backend: http://localhost:3001/api/chat` → frontend is calling the backend (if the backend still doesn't log, check that the backend is running and nothing is blocking port 3001).
   - `[Chat] Backend not configured (set VITE_API_URL in Frontend/.env)` → frontend has no backend URL; fix step 1 and 2.

5. **Backend running** – In a separate terminal, run `./run-backend.sh`. You should see `jobAI backend running at http://localhost:3001`. When the frontend sends a request, the backend terminal will show `>>> Incoming: POST /api/chat` and then `[CHAT]` lines.

---

## Viewing backend logs (dedicated terminal)

When the frontend shows **"No response"** or something goes wrong, you need to see what the backend received and returned. Use **two terminals** so the backend has its own:

1. **Terminal 1 (backend – keep this open):**  
   From the folder that contains `Backend` and `Frontend`:
   ```bash
   ./run-backend.sh
   ```
   All backend logs appear here: every request (method, path, status, duration) and for `/api/chat` the incoming message and the reply (e.g. `message="..." → reply="No response"`).

2. **Terminal 2 (frontend):**  
   In a second terminal, same folder:
   ```bash
   ./run-frontend.sh
   ```

When you send a chat message and the frontend shows "No response", look at **Terminal 1**. You'll see a line like:
- `[CHAT] DEEPSEEK_API_KEY not set – returning "No response"`  
- `[CHAT] message="your text" → reply="No response"`  
- And the request line: `POST /api/chat 200 …ms`

So you can tell whether the backend got the request, why it returned "No response", and what it sent back. Don't use `./run-all.sh` when you want to monitor backend logs, because the backend runs in the background and its output is mixed with the frontend.

---

## API endpoints (current)

| Endpoint       | Method | Description                          |
|----------------|--------|--------------------------------------|
| `/health`      | GET    | Health check (used by frontend)      |
| `/api/hello`   | GET    | Simple test message from backend     |
| `/api/chat`    | POST   | Chat: body `{ message }`, response `{ reply }` |

---

## Getting the chatbot and resume upload working with the backend

Right now the frontend **chatbot** uses local rule-based replies and **resume upload** uses mock data. To have them use this backend (and an LLM for chat, real parsing for resume), you add backend routes and call them from the frontend.

### Chatbot + LLM

1. **Backend:** Add a route that accepts the user message and calls your LLM (e.g. OpenAI, Anthropic).  
   - Example: `POST /api/chat` with body `{ "message": "..." }`, return `{ "reply": "..." }`.  
   - Use an env var for the LLM API key (e.g. `OPENAI_API_KEY`) and never commit it.

2. **Frontend:** In `OverviewChat.tsx`, replace the local `getBotReply()` + `setTimeout` with a `fetch` to `VITE_API_URL + "/api/chat"` (using `getApiUrl()` from `src/api/client.ts`), send the user message, and display the reply from the backend.

### Resume upload

1. **Backend:** Add a route that accepts the uploaded file and returns extracted profile data.  
   - Example: `POST /api/resume/upload` with `multipart/form-data` (file).  
   - Use a parser (e.g. for PDF/DOCX) or an LLM to extract name, skills, experience, etc., and return JSON in the shape expected by the frontend (see `ResumeUpload.tsx` / `ExtractedProfile`).

2. **Frontend:** In `ResumeUpload.tsx`, replace the mock `MOCK_EXTRACTED` and `setTimeout` with:
   - `fetch(getApiUrl() + "/api/resume/upload", { method: "POST", body: formData })`,
   - then call `onExtracted(extractedData, fileName)` with the backend response.

---

## Optional: other environment variables

See **“Where to add your DeepSeek API key”** above for `DEEPSEEK_API_KEY`. When you add resume parsing via another service, you can add more vars to `Backend/.env`, e.g. `RESUME_PARSER_API_KEY`. Never commit real keys; `Backend/.env` is in `.gitignore`.

---

## Run scripts (same dir as Backend & Frontend)

From the folder that contains `Backend`, `Frontend`, and these scripts:

| Script | What it does |
|--------|----------------|
| `./run-frontend.sh` | `npm install` + `npm run build` + `npm run dev` in Frontend (build then dev server). |
| `./run-backend.sh` | `npm install` + `npm run dev` in Backend (server on port 3001). |
| `./run-all.sh` | Starts backend in background, then runs frontend (build + dev). Ctrl+C stops both. |

To see backend logs (e.g. when chat says "No response"), use **two terminals**: run `./run-backend.sh` in one (dedicated backend terminal) and `./run-frontend.sh` in the other. See **Viewing backend logs (dedicated terminal)** above.

Make scripts executable once: `chmod +x run-*.sh`. On Windows use Git Bash or WSL to run the `.sh` files.

---

## Summary

1. **DeepSeek API key:** Put it in `Backend/.env` as `DEEPSEEK_API_KEY=sk-...` (see section above).  
2. **Backend:** `cd Backend` → `npm install` → `npm run dev` (runs on port 3001).  
3. **Frontend:** Set `VITE_API_URL=http://localhost:3001` in `Frontend/.env`, then run the frontend.  
4. **Chatbot + resume:** Add `/api/chat` and `/api/resume/upload` on the backend and switch the frontend to use those endpoints instead of local/mock logic.
