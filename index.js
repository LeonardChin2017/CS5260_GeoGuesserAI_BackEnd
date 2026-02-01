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
db.exec(`
  CREATE TABLE IF NOT EXISTS user_chat (
    user_id TEXT PRIMARY KEY,
    messages TEXT NOT NULL,
    updated_at TEXT NOT NULL
  )
`);
if (hasClerk) console.log('[BACKEND] Clerk + DB: user Gemini keys and chat stored in', DB_PATH);

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

/** Root – API info (no web UI) */
app.get('/', (req, res) => {
  res.json({
    service: 'jobai-backend',
    message: 'API only. Try GET /health or POST /api/chat.',
    health: req.originalUrl.replace(/\/?$/, '') + '/health',
  });
});

/** Health check – for frontend and load balancers */
app.get('/health', (req, res) => {
  res.json({ ok: true, service: 'jobai-backend', timestamp: new Date().toISOString() });
});

/** Simple hello – for testing connectivity from the frontend */
app.get('/api/hello', (req, res) => {
  res.json({ message: 'Hello from jobAI backend', env: process.env.NODE_ENV || 'development' });
});

/** Check if user has a Gemini key saved (Clerk auth required). Returns { hasKey: true|false }; never returns the key. */
if (hasClerk) {
  app.get('/api/user/gemini-key', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const row = db.prepare('SELECT 1 FROM user_gemini_keys WHERE user_id = ?').get(auth.userId);
    res.json({ hasKey: !!row });
  });

  /** Save user's Gemini API key (Clerk auth required). Key is encrypted at rest in DB. */
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

  /** Clear user's saved Gemini API key (Clerk auth required). */
  app.delete('/api/user/gemini-key', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const userId = auth.userId;
    db.prepare('DELETE FROM user_gemini_keys WHERE user_id = ?').run(userId);
    res.json({ ok: true });
  });

  /** Get user's chat history (Clerk auth required). Returns { messages: Message[] }. */
  app.get('/api/user/chat', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const userId = auth.userId;
    const row = db.prepare('SELECT messages FROM user_chat WHERE user_id = ?').get(userId);
    let messages = [];
    if (row && row.messages) {
      try {
        messages = JSON.parse(row.messages);
        if (!Array.isArray(messages)) messages = [];
      } catch {
        messages = [];
      }
    }
    res.json({ messages });
  });

  /** Save user's chat history (Clerk auth required). Body { messages: Array<{ id, role, content, timestamp }> }. */
  app.post('/api/user/chat', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const userId = auth.userId;
    const messages = req.body?.messages;
    if (!Array.isArray(messages)) return res.status(400).json({ error: 'messages array required' });
    const now = new Date().toISOString();
    const json = JSON.stringify(messages);
    db.prepare(
      'INSERT INTO user_chat (user_id, messages, updated_at) VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET messages = ?, updated_at = ?'
    ).run(userId, json, now, json, now);
    res.json({ ok: true });
  });

  /** Clear user's chat history (Clerk auth required). */
  app.delete('/api/user/chat', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const userId = auth.userId;
    db.prepare('DELETE FROM user_chat WHERE user_id = ?').run(userId);
    res.json({ ok: true });
  });
}

/** JobAI agent system prompt: persona + tools the agent can use (results are injected by the backend). */
const JOBAI_SYSTEM_PROMPT = `You are jobAI, a friendly assistant that helps users land their next job offer. You have access to tools that run on the backend; when the user asks to find jobs, scan job offers, check application status, or get profile suggestions, the backend will run those tools and you will receive the results to summarize.

Your tools (backend executes them; you describe and interpret results):
- scan_job_offers: Search for job listings that match the user's profile (role, skills, location). You will receive a list of jobs to present.
- get_application_status: Summarize the user's application status (saved applications, follow-ups).
- suggest_profile_improvements: Suggest improvements to resume or profile based on their goals.

Be concise, encouraging, and actionable. When you receive tool results in the conversation, present them clearly (e.g. list key jobs with title and company). Never make up job titles or companies—only use data from the tool results. If no tool was run yet, invite the user to try "Find jobs that match my profile" or ask about their application status.`;

/** Mock: "scan" job offers (no real JobStreet; returns plausible fake listings for demo). */
function runScanJobOffers() {
  const jobs = [
    { id: '1', title: 'Senior Software Engineer', company: 'TechCorp Singapore', location: 'Singapore', match: '92%', posted: '2 days ago' },
    { id: '2', title: 'Full Stack Developer', company: 'StartupXYZ', location: 'Remote', match: '88%', posted: '1 day ago' },
    { id: '3', title: 'Frontend Engineer', company: 'ProductLabs', location: 'Singapore', match: '85%', posted: '3 days ago' },
    { id: '4', title: 'Software Developer', company: 'FinanceHub', location: 'Singapore / Hybrid', match: '82%', posted: '5 days ago' },
  ];
  return { type: 'job_scan', jobs };
}

/** Mock: application status for the user. */
function runGetApplicationStatus() {
  return {
    type: 'application_status',
    total: 12,
    inProgress: 5,
    interviews: 2,
    offers: 0,
    recent: ['TechCorp – Applied 3 days ago', 'StartupXYZ – Screening'],
  };
}

/** Mock: profile improvement suggestions. */
function runSuggestProfileImprovements() {
  return {
    type: 'profile_improvements',
    suggestions: [
      'Add 2–3 quantifiable achievements to your current role.',
      'Include keywords from job descriptions (e.g. "React", "Node") in your skills section.',
      'Keep resume to 1–2 pages for roles with &lt;10 years experience.',
    ],
  };
}

/** Detect which tool to run from the last user message (keyword-based). */
function getToolToRun(userMessage) {
  const lower = (userMessage || '').toLowerCase();
  if (/\b(find|scan|search|show|get|list|match).*job|job.*(match|find|offer|listing)/.test(lower) || /jobs that match|job offer/.test(lower)) return 'scan_job_offers';
  if (/\b(application|status|applied|follow.?up)\b/.test(lower) || /application status/.test(lower)) return 'get_application_status';
  if (/\b(improve|suggest|tip|improvement|profile|resume)\b/.test(lower) || /suggest improvement/.test(lower)) return 'suggest_profile_improvements';
  return null;
}

/** Run a tool by name and return agentUpdate for frontend. */
function runTool(name) {
  switch (name) {
    case 'scan_job_offers': return runScanJobOffers();
    case 'get_application_status': return runGetApplicationStatus();
    case 'suggest_profile_improvements': return runSuggestProfileImprovements();
    default: return null;
  }
}

/** Build Gemini contents from chat history (messages from frontend: { role, content }[]). */
function buildGeminiContents(messages, currentMessage, toolResultText) {
  const parts = [];
  for (const m of messages || []) {
    const role = m.role === 'user' ? 'user' : 'model';
    const text = typeof m.content === 'string' ? m.content : '';
    if (!text.trim()) continue;
    parts.push({ role, parts: [{ text: text.trim() }] });
  }
  parts.push({ role: 'user', parts: [{ text: currentMessage }] });
  if (toolResultText && toolResultText.trim()) {
    parts.push({ role: 'model', parts: [{ text: '[Tool result from jobAI backend]\n' + toolResultText.trim() }] });
    parts.push({ role: 'user', parts: [{ text: 'Summarize the above for the user in a short, friendly reply.' }] });
  }
  return parts;
}

/** Call Gemini API with system prompt and conversation history. */
async function getGeminiReply(contents, apiKey) {
  if (!apiKey || !apiKey.trim()) return null;
  const model = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(apiKey.trim())}`;
  const body = {
    systemInstruction: { parts: [{ text: JOBAI_SYSTEM_PROMPT }] },
    contents: contents.map((c) => ({ role: c.role, parts: c.parts })),
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

/** Chat: JobAI agent with system prompt; optional history; runs mock tools and returns reply + agentUpdates. */
app.post('/api/chat', async (req, res) => {
  console.log('\n[CHAT] <<< Received from frontend');
  const message = req.body?.message;
  if (typeof message !== 'string' || !message.trim()) {
    return res.status(400).json({ error: 'message required' });
  }
  const history = Array.isArray(req.body?.messages) ? req.body.messages : [];
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

  const agentUpdates = [];
  let toolResultText = '';
  const toolName = getToolToRun(message);
  if (toolName) {
    const update = runTool(toolName);
    if (update) {
      agentUpdates.push(update);
      if (update.type === 'job_scan' && update.jobs) {
        toolResultText = 'Job scan results:\n' + update.jobs.map((j) => `- ${j.title} at ${j.company} (${j.location}) – match ${j.match}`).join('\n');
      } else if (update.type === 'application_status') {
        toolResultText = `Application status: ${update.total} total, ${update.inProgress} in progress, ${update.interviews} interviews. Recent: ${(update.recent || []).join('; ')}`;
      } else if (update.type === 'profile_improvements' && update.suggestions) {
        toolResultText = 'Profile improvement suggestions:\n' + update.suggestions.map((s, i) => `${i + 1}. ${s}`).join('\n');
      }
    }
  }

  const contents = buildGeminiContents(history, message.trim(), toolResultText);
  let reply = 'No response';
  const geminiReply = await getGeminiReply(contents, apiKey);
  if (geminiReply) reply = geminiReply;
  else if (!apiKey) console.log('[CHAT] No API key (set in Settings when using Clerk, or send apiKey/env)');
  console.log('[CHAT] >>> Sending reply + %d agentUpdate(s) to frontend\n', agentUpdates.length);
  return res.json({ reply, agentUpdates: agentUpdates.length ? agentUpdates : undefined });
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
