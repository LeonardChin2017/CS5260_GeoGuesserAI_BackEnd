require('dotenv').config();

const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const multer = require('multer');

const app = express();
const PORT = process.env.PORT || 3001;

// Directory for uploaded resumes (create if missing)
const UPLOADS_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOADS_DIR)) {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
  console.log('[BACKEND] Created uploads dir:', UPLOADS_DIR);
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename: (req, file, cb) => {
    const safe = (file.originalname || 'resume').replace(/[^a-zA-Z0-9._-]/g, '_');
    const name = `${Date.now()}-${safe}`;
    cb(null, name);
  },
});
const upload = multer({ storage, limits: { fileSize: 10 * 1024 * 1024 } }); // 10MB

app.use(cors({ origin: true }));
app.use(express.json());

/** Request logging – log as soon as request hits, then again when response is sent */
app.use((req, res, next) => {
  console.log(`>>> Incoming: ${req.method} ${req.originalUrl}`);
  const start = Date.now();
  res.on('finish', () => {
    const duration = Date.now() - start;
    console.log(`<<< Done: ${req.method} ${req.originalUrl} ${res.statusCode} ${duration}ms`);
  });
  next();
});

/** Health check – for frontend and load balancers */
app.get('/health', (req, res) => {
  res.json({ ok: true, service: 'jobai-backend', timestamp: new Date().toISOString() });
});

/** Simple hello – for testing connectivity from the frontend */
app.get('/api/hello', (req, res) => {
  res.json({ message: 'Hello from jobAI backend', env: process.env.NODE_ENV || 'development' });
});

/** Call Gemini API for chat reply. Uses GEMINI_API_KEY. */
async function getGeminiReply(message) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || !apiKey.trim()) return null;
  const model = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`;
  const body = {
    contents: [{ parts: [{ text: message }] }],
  };
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text();
    console.log('[CHAT] Gemini API error:', res.status, err?.slice(0, 300));
    return null;
  }
  const data = await res.json();
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
  return typeof text === 'string' ? text.trim() : null;
}

/** Chat: frontend sends message, backend returns reply via Gemini (GEMINI_API_KEY). */
app.post('/api/chat', async (req, res) => {
  console.log('\n[CHAT] <<< Received from frontend');
  const message = req.body?.message;
  if (typeof message !== 'string' || !message.trim()) {
    console.log('[CHAT] bad request: message missing or not a string. body:', JSON.stringify(req.body));
    return res.status(400).json({ error: 'message required' });
  }
  console.log('[CHAT] message from frontend: "%s"', message.trim().slice(0, 200));
  let reply = 'No response';
  const geminiReply = await getGeminiReply(message.trim());
  if (geminiReply) reply = geminiReply;
  else if (!process.env.GEMINI_API_KEY?.trim()) console.log('[CHAT] GEMINI_API_KEY not set – replying "No response"');
  console.log('[CHAT] >>> Sending reply to frontend: "%s"\n', reply.slice(0, 200));
  return res.json({ reply });
});

/** Resume upload: frontend sends file, backend saves to local uploads/ and logs */
app.post('/api/resume/upload', upload.single('file'), (req, res) => {
  console.log('\n[RESUME] <<< Received upload from frontend');
  if (!req.file) {
    console.log('[RESUME] bad request: no file in request');
    return res.status(400).json({ error: 'file required' });
  }
  const savedPath = path.join(UPLOADS_DIR, req.file.filename);
  console.log('[RESUME] saved to local: %s (original name: %s, size: %s bytes)\n', savedPath, req.file.originalname, req.file.size);
  res.json({ ok: true, savedPath: req.file.filename, originalName: req.file.originalname });
});

app.listen(PORT, () => {
  console.log(`jobAI backend running at http://localhost:${PORT}`);
  console.log('Watch this terminal for [CHAT] and [RESUME] logs when the frontend sends requests.\n');
});
