# Deploy jobAI Backend to Digital Ocean

Two options: **App Platform** (easiest, managed) or **Droplet** (full control).

---

## Option 1: App Platform (recommended)

1. **Digital Ocean** → [Apps](https://cloud.digitalocean.com/apps) → **Create App**.
2. **Source:** Connect your GitHub repo, select the repo and branch. Set **Source Directory** to `Backend` (so the build runs from the Backend folder).
3. **Resources:** Choose **Web Service**.
4. **Build:**
   - **Build Command:** `npm ci`
   - **Run Command:** `npm start`
   - **HTTP Port:** `3001`
5. **Environment Variables** (App-level → Edit):
   - `CLERK_SECRET_KEY` = your Clerk Secret Key ([dashboard](https://dashboard.clerk.com) → API Keys)
   - `ENCRYPTION_SECRET` = a secret (min 16 chars). Generate:  
     `node -e "console.log(require('crypto').randomBytes(32).toString('base64'))"`
   - `NODE_ENV` = `production` (optional)
6. **Deploy.** Your backend URL will be like `https://your-app-xxxxx.ondigitalocean.app`.
7. **Frontend:** Set `VITE_API_URL=https://your-app-xxxxx.ondigitalocean.app` (no trailing slash) in your frontend env (e.g. Vercel).

App Platform sets `PORT` automatically; the app uses `process.env.PORT || 3001`.

---

## Option 2: Deploy with Docker (App Platform or Droplet)

If you prefer to use the Dockerfile (e.g. same image locally and in production):

1. **App Platform:** Create App → **Dockerfile** as source type, point to the repo and set **Dockerfile Path** to `Backend/Dockerfile` (or put Dockerfile in repo root and set **Source Directory** to `Backend` and ensure Dockerfile `COPY` paths match). Add the same env vars as above. **HTTP Port** = `3001`.
2. **Droplet:** On a Droplet with Docker installed:
   ```bash
   git clone https://github.com/your-org/your-repo.git
   cd your-repo/Backend
   docker build -t jobai-backend .
   docker run -d --restart unless-stopped -p 3001:3001 \
     -e CLERK_SECRET_KEY="your-clerk-secret" \
     -e ENCRYPTION_SECRET="your-secret" \
     --name jobai-backend jobai-backend
   ```
   Use a reverse proxy (e.g. Nginx) and Let's Encrypt for HTTPS in front of port 3001.

---

## Option 3: Droplet with Node.js (no Docker)

1. **Create a Droplet** (Ubuntu 22.04). SSH in.
2. **Install Node 20:**
   ```bash
   curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
   sudo apt-get install -y nodejs
   ```
3. **Clone and run:**
   ```bash
   git clone https://github.com/your-org/your-repo.git
   cd your-repo/Backend
   npm ci
   ```
4. **Create `.env`** (copy from `.env.example`, fill in `GEMINI_API_KEY` and `GEMINI_KEY_ENCODING_SECRET`).
5. **Run with PM2** (keeps the app running and restarts on crash):
   ```bash
   sudo npm install -g pm2
   PORT=3001 pm2 start index.js --name jobai-backend
   pm2 save
   pm2 startup
   ```
6. **HTTPS:** Put Nginx (or Caddy) in front, use Let's Encrypt (e.g. `certbot`). Proxy requests to `http://127.0.0.1:3001`.

---

## Required environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CLERK_SECRET_KEY` | Yes | Clerk Secret Key from [dashboard](https://dashboard.clerk.com) → API Keys. Backend verifies JWT and stores Gemini key per user. |
| `ENCRYPTION_SECRET` | Yes | Min 16 characters; encrypts Gemini API keys at rest in the database. Generate: `node -e "console.log(require('crypto').randomBytes(32).toString('base64'))"` |
| `PORT` | No | Default 3001. Set by App Platform automatically |

Optional: `GEMINI_API_KEY` (only when Clerk is not used); `GEMINI_MODEL` (default `gemini-2.5-flash`).

---

## After deploy

- **Health check:** `GET https://your-backend-url/health` → `{ "ok": true }`
- **Frontend:** Set `VITE_API_URL` to your backend URL (with `https://`) so the frontend and chat use the deployed backend.
