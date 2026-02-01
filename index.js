require('dotenv').config();

const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const crypto = require('crypto');
const Database = require('better-sqlite3');

const ALGO = 'aes-256-gcm';
const IV_LEN = 16;
const AUTH_TAG_LEN = 16;

const hasClerk = !!(process.env.CLERK_SECRET_KEY || '').trim();
const hasEncryptionSecret = () => {
  const s = process.env.ENCRYPTION_SECRET || process.env.GEMINI_KEY_ENCODING_SECRET || '';
  return s && s.length >= 16;
};

/** Derive a 32-byte key for AES-256 (DB at-rest encryption). */
function getEncryptionKey() {
  const secret = process.env.ENCRYPTION_SECRET || process.env.GEMINI_KEY_ENCODING_SECRET || '';
  if (!secret || secret.length < 16) return null;
  return crypto.createHash('sha256').update(secret, 'utf8').digest();
}

/** Encrypt plaintext for DB storage; returns base64(iv + authTag + ciphertext). */
function encryptForDb(plaintext) {
  const key = getEncryptionKey();
  if (!key || typeof plaintext !== 'string' || !plaintext.trim()) return null;
  const iv = crypto.randomBytes(IV_LEN);
  const cipher = crypto.createCipheriv(ALGO, key, iv, { authTagLength: AUTH_TAG_LEN });
  const enc = Buffer.concat([cipher.update(plaintext.trim(), 'utf8'), cipher.final()]);
  const authTag = cipher.getAuthTag();
  return Buffer.concat([iv, authTag, enc]).toString('base64');
}

/** Decrypt blob from DB; returns plain API key or null. */
function decryptFromDb(encoded) {
  const key = getEncryptionKey();
  if (!key || typeof encoded !== 'string' || !encoded.trim()) return null;
  let buf;
  try {
    buf = Buffer.from(encoded.trim(), 'base64');
  } catch {
    return null;
  }
  if (buf.length < IV_LEN + AUTH_TAG_LEN + 1) return null;
  const iv = buf.subarray(0, IV_LEN);
  const authTag = buf.subarray(IV_LEN, IV_LEN + AUTH_TAG_LEN);
  const ciphertext = buf.subarray(IV_LEN + AUTH_TAG_LEN);
  const decipher = crypto.createDecipheriv(ALGO, key, iv, { authTagLength: AUTH_TAG_LEN });
  decipher.setAuthTag(authTag);
  try {
    return decipher.update(ciphertext) + decipher.final('utf8');
  } catch {
    return null;
  }
}

/** Legacy: decode encodedKey from frontend (backward compat when not using Clerk). */
function decodeGeminiKey(encoded) {
  return decryptFromDb(encoded);
}

const app = express();
const PORT = process.env.PORT || 3001;

// SQLite DB for per-user Gemini keys (when using Clerk)
const DATA_DIR = path.join(__dirname, 'data');
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
const DB_PATH = path.join(DATA_DIR, 'keys.db');
const db = new Database(DB_PATH);
db.exec(`
  CREATE TABLE IF NOT EXISTS user_gemini_keys (
    user_id TEXT PRIMARY KEY,
    encrypted_key TEXT NOT NULL,
    updated_at TEXT NOT NULL
  )
`);
if (hasClerk) console.log('[BACKEND] Clerk + DB: user Gemini keys stored in', DB_PATH);

// Directory for uploaded resumes (create if missing)
const UPLOADS_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOADS_DIR)) {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
  console.log('[BACKEND] Created uploads dir:', UPLOADS_DIR);
}

// Clerk (only when CLERK_SECRET_KEY is set). Use getAuth(req) for API; do not use requireAuth (redirects).
let getAuth;
if (hasClerk) {
  const clerk = require('@clerk/express');
  getAuth = clerk.getAuth;
  app.use(clerk.clerkMiddleware());
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

/** Save user's Gemini API key (Clerk auth required). Key is encrypted at rest in DB. */
if (hasClerk) {
  app.put('/api/user/gemini-key', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const userId = auth.userId;
    if (!hasEncryptionSecret()) return res.status(503).json({ error: 'ENCRYPTION_SECRET not set' });
    const apiKey = typeof req.body?.apiKey === 'string' ? req.body.apiKey.trim() : '';
    if (!apiKey) return res.status(400).json({ error: 'apiKey required' });
    const encrypted = encryptForDb(apiKey);
    if (!encrypted) return res.status(500).json({ error: 'Failed to encrypt key' });
    const now = new Date().toISOString();
    db.prepare(
      'INSERT INTO user_gemini_keys (user_id, encrypted_key, updated_at) VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET encrypted_key = ?, updated_at = ?'
    ).run(userId, encrypted, now, encrypted, now);
    res.json({ ok: true });
  });
}

/** Call Gemini API for chat reply. apiKey can be raw string (legacy) or from DB (Clerk). */
async function getGeminiReply(message, apiKey) {
  if (!apiKey || !apiKey.trim()) return null;
  const model = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(apiKey.trim())}`;
  const body = { contents: [{ parts: [{ text: message }] }] };
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

/** Chat: with Clerk – use key from DB for authenticated user. Without Clerk – use apiKey/encodedKey in body or env. */
app.post('/api/chat', async (req, res) => {
  console.log('\n[CHAT] <<< Received from frontend');
  const message = req.body?.message;
  if (typeof message !== 'string' || !message.trim()) {
    return res.status(400).json({ error: 'message required' });
  }
  let apiKey = null;
  if (hasClerk && getAuth(req).userId) {
    const userId = getAuth(req).userId;
    const row = db.prepare('SELECT encrypted_key FROM user_gemini_keys WHERE user_id = ?').get(userId);
    if (row && row.encrypted_key) apiKey = decryptFromDb(row.encrypted_key);
  }
  if (!apiKey) {
    const raw = typeof req.body?.apiKey === 'string' ? req.body.apiKey.trim() : '';
    const encoded = typeof req.body?.encodedKey === 'string' ? req.body.encodedKey : null;
    if (raw) apiKey = raw;
    else if (encoded) apiKey = decodeGeminiKey(encoded) || '';
    else apiKey = (process.env.GEMINI_API_KEY || '').trim();
  }
  console.log('[CHAT] message from frontend: "%s"', message.trim().slice(0, 200));
  let reply = 'No response';
  const geminiReply = await getGeminiReply(message.trim(), apiKey);
  if (geminiReply) reply = geminiReply;
  else if (!apiKey) console.log('[CHAT] No API key (set in Settings when using Clerk, or send apiKey/env)');
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
