# Deploy Backend (Python) to Digital Ocean

## Option 1: App Platform (recommended)

1. Digital Ocean → Apps → Create App.
2. Source: connect repo, set Source Directory to `Backend`.
3. Resources: Web Service.
4. Build:
   ```bash
   cd Backend
   cp .env.example .env
   conda create -n GGSolver python=3.14 -y && conda activate GGSolver
   pip install -r requirements.txt
   uvicorn app:app --host 0.0.0.0 --port 3001
   ```
   - HTTP Port: `3001`
5. Environment Variables:
   - `GOOGLE_MAPS_API_KEY`: Get one at https://console.cloud.google.com/apis/credentials
   - `GEMINI_API_KEY`: Get one at https://aistudio.google.com/app/api-keys.
   - `CORS_ALLOW_ORIGINS`: Comma-separated frontend origins, e.g. `https://your-app.vercel.app`
6. Deploy, then set `VITE_API_URL` in frontend to your backend URL.

## Option 2: Droplet with Docker

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo/Backend
docker build -t jobai-backend .
docker run -d --restart unless-stopped -p 3001:3001 \
  -e ENCRYPTION_SECRET="your-secret" \
  --name jobai-backend jobai-backend
```

## Option 3: Droplet (no Docker)

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv
git clone https://github.com/your-org/your-repo.git
cd your-repo/Backend
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 3001
```
