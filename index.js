require('dotenv').config();

const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const crypto = require('crypto');

const ALGO = 'aes-256-gcm';
const IV_LEN = 16;
const AUTH_TAG_LEN = 16;
const KEY_LEN = 32;

/** Derive a 32-byte key from ENCODING_SECRET for AES-256. */
function getEncodingKey() {
  const secret = process.env.GEMINI_KEY_ENCODING_SECRET || process.env.ENCODING_SECRET || '';
  if (!secret || secret.length < 16) return null;
  return crypto.createHash('sha256').update(secret, 'utf8').digest();
}

/** Encrypt plaintext with server secret; returns base64(iv + authTag + ciphertext). */
function encodeGeminiKey(plaintext) {
  const key = getEncodingKey();
  if (!key || typeof plaintext !== 'string' || !plaintext.trim()) return null;
  const iv = crypto.randomBytes(IV_LEN);
  const cipher = crypto.createCipheriv(ALGO, key, iv, { authTagLength: AUTH_TAG_LEN });
  const enc = Buffer.concat([cipher.update(plaintext.trim(), 'utf8'), cipher.final()]);
  const authTag = cipher.getAuthTag();
  return Buffer.concat([iv, authTag, enc]).toString('base64');
}

/** Decrypt base64 blob from frontend; returns plain API key or null. */
function decodeGeminiKey(encoded) {
  const key = getEncodingKey();
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

/** Return encoded Gemini API key for the identified user. Backend has the key; frontend gets only an encrypted blob.
 *  When you add auth (e.g. Clerk), only return encodedKey if the request is from the identified person. */
app.get('/api/gemini-key', (req, res) => {
  const rawKey = (process.env.GEMINI_API_KEY || '').trim();
  if (!rawKey) {
    return res.status(503).json({ error: 'Gemini API key not configured on server' });
  }
  const encodingKey = getEncodingKey();
  if (!encodingKey) {
    return res.status(503).json({ error: 'Server encoding secret not set (GEMINI_KEY_ENCODING_SECRET)' });
  }
  const encodedKey = encodeGeminiKey(rawKey);
  if (!encodedKey) {
    return res.status(500).json({ error: 'Failed to encode key' });
  }
  res.json({ encodedKey });
});

/** Call Gemini API for chat reply. Uses apiKey from request (raw or decoded from encodedKey) or fallback to GEMINI_API_KEY env. */
async function getGeminiReply(message, apiKeyFromRequest, encodedKeyFromRequest) {
  let apiKey = (typeof apiKeyFromRequest === 'string' && apiKeyFromRequest.trim())
    ? apiKeyFromRequest.trim()
    : '';
  if (!apiKey && encodedKeyFromRequest) {
    apiKey = decodeGeminiKey(encodedKeyFromRequest) || '';
  }
  if (!apiKey) apiKey = (process.env.GEMINI_API_KEY || '').trim();
  if (!apiKey) return null;
  if (!apiKey) return null;
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

/** Chat: frontend sends message and optional apiKey or encodedKey; backend decodes encodedKey or uses apiKey/env. */
app.post('/api/chat', async (req, res) => {
  console.log('\n[CHAT] <<< Received from frontend');
  const message = req.body?.message;
  const apiKey = typeof req.body?.apiKey === 'string' ? req.body.apiKey : undefined;
  const encodedKey = typeof req.body?.encodedKey === 'string' ? req.body.encodedKey : undefined;
  if (typeof message !== 'string' || !message.trim()) {
    console.log('[CHAT] bad request: message missing or not a string. body keys:', Object.keys(req.body || {}));
    return res.status(400).json({ error: 'message required' });
  }
  console.log('[CHAT] message from frontend: "%s"', message.trim().slice(0, 200));
  let reply = 'No response';
  const geminiReply = await getGeminiReply(message.trim(), apiKey, encodedKey);
  if (geminiReply) reply = geminiReply;
  else if (!apiKey?.trim() && !encodedKey && !process.env.GEMINI_API_KEY?.trim()) {
    console.log('[CHAT] No API key in request and GEMINI_API_KEY not set – replying "No response"');
  }
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
