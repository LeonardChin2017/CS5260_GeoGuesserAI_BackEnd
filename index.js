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
db.exec(`
  CREATE TABLE IF NOT EXISTS user_resume (
    user_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    original_name TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    extracted_profile TEXT,
    resume_text TEXT
  )
`);
try {
  db.exec('ALTER TABLE user_resume ADD COLUMN extracted_profile TEXT');
} catch (e) {
  if (!/duplicate column/i.test(e.message)) throw e;
}
try {
  db.exec('ALTER TABLE user_resume ADD COLUMN resume_text TEXT');
} catch (e) {
  if (!/duplicate column/i.test(e.message)) throw e;
}
try {
  db.exec('ALTER TABLE user_resume ADD COLUMN activity_steps TEXT');
} catch (e) {
  if (!/duplicate column/i.test(e.message)) throw e;
}
db.exec(`
  CREATE TABLE IF NOT EXISTS user_links (
    user_id TEXT PRIMARY KEY,
    links TEXT NOT NULL,
    updated_at TEXT NOT NULL
  )
`);
db.exec(`
  CREATE TABLE IF NOT EXISTS backend_activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    source TEXT NOT NULL,
    type TEXT NOT NULL,
    message TEXT NOT NULL,
    payload TEXT,
    created_at TEXT NOT NULL
  )
`);
db.exec('CREATE INDEX IF NOT EXISTS idx_backend_activity_user_created ON backend_activity(user_id, created_at DESC)');
if (hasClerk) console.log('[BACKEND] Clerk + DB: user Gemini keys, chat, resume, links, and backend activity stored in', DB_PATH);

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

// Store resumes in per-user subfolders (uploads/userId/) for 100+ users; anonymous uploads go to uploads/
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const userId = hasClerk && getAuth(req)?.userId;
    const dir = userId ? path.join(UPLOADS_DIR, userId) : UPLOADS_DIR;
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    const safe = (file.originalname || 'resume').replace(/[^a-zA-Z0-9._-]/g, '_');
    const name = `${Date.now()}-${safe}`;
    cb(null, name);
  },
});
const upload = multer({ storage, limits: { fileSize: 10 * 1024 * 1024 } }); // 10MB

/** Agent: decide what to share to frontend and summarize into one short line. Backend owns format; frontend just displays. */
function summarizeActivityForFrontend(source, type, message, payload) {
  const raw = (message || '').trim();
  if (source === 'resume') {
    if (raw.includes('Received resume upload')) return raw.includes(':') ? `Resume received: ${raw.split(':').slice(1).join(':').trim()}` : 'Resume received.';
    if (raw.includes('File saved') && raw.includes('Starting extraction')) return 'File saved, starting extraction.';
    if (raw.includes('Reading PDF') || raw.includes('extract-text')) return 'Reading PDF.';
    if (raw.includes('Extracted text')) return raw.includes('chars') ? 'Text extracted from PDF.' : 'Text extracted.';
    if (raw.includes('Analyzing') && raw.includes('AI')) return 'Analyzing resume with AI.';
    if (raw.includes('Extraction complete') || raw.includes('Ready for follow-up')) return 'Profile extracted, ready for chat.';
    if (raw.includes('No API key')) return 'File saved. Set API key in Settings to extract profile.';
    if (raw.includes('Extraction error')) return raw.length > 80 ? 'Extraction failed.' : raw;
  }
  if (source === 'chat') {
    if (raw.includes('Chat request received')) return 'Chat request received.';
    if (raw.includes('Running tool:')) return raw.replace('Running tool:', 'Tool:').trim();
    if (raw.includes('Reply sent')) return 'Reply sent.';
  }
  const oneLine = raw.split(/[.!?\n]/)[0].trim();
  return oneLine.length > 72 ? oneLine.slice(0, 69) + '…' : oneLine || 'Activity.';
}

/** Record backend activity for the frontend. Agent summarizes to one line before storing. */
function recordBackendActivity(req, { source, type, message, payload }) {
  const userId = hasClerk && typeof getAuth === 'function' && getAuth(req)?.userId ? getAuth(req).userId : null;
  const oneLine = summarizeActivityForFrontend(source, type, message, payload);
  const now = new Date().toISOString();
  const payloadJson = payload != null ? JSON.stringify(payload) : null;
  db.prepare(
    'INSERT INTO backend_activity (user_id, source, type, message, payload, created_at) VALUES (?, ?, ?, ?, ?, ?)'
  ).run(userId, source, type || 'agent_step', oneLine, payloadJson, now);
  // Keep last 100 per user (and last 100 for anonymous) to avoid unbounded growth
  const keep = 100;
  if (userId) {
    const ids = db.prepare('SELECT id FROM backend_activity WHERE user_id = ? ORDER BY created_at DESC').all(userId);
    if (ids.length > keep) {
      const toDelete = ids.slice(keep).map((r) => r.id);
      if (toDelete.length > 0) db.prepare(`DELETE FROM backend_activity WHERE id IN (${toDelete.join(',')})`).run();
    }
  } else {
    const ids = db.prepare('SELECT id FROM backend_activity WHERE user_id IS NULL ORDER BY created_at DESC').all();
    if (ids.length > keep) {
      const toDelete = ids.slice(keep).map((r) => r.id);
      if (toDelete.length > 0) db.prepare(`DELETE FROM backend_activity WHERE id IN (${toDelete.join(',')})`).run();
    }
  }
}

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

/** Debug: check if backend has activity data. GET /api/debug/activity to see recent backend_activity rows. No auth required. (formatBackendActivityAsStep is defined later in file.) */
app.get('/api/debug/activity', (req, res) => {
  const limit = Math.min(parseInt(req.query?.limit, 10) || 10, 50);
  const rows = db.prepare('SELECT id, user_id, source, type, message, payload, created_at FROM backend_activity ORDER BY id DESC LIMIT ?').all(limit);
  res.json({
    debug: true,
    backend_activity_count: rows.length,
    backend_activity_recent: rows.map((r) => ({ id: r.id, source: r.source, type: r.type, message: r.message, created_at: r.created_at })),
    note: 'Activity steps are in POST /api/resume/upload response (activitySteps) and GET /api/resume (activitySteps). Backend logs: [RESUME] >>> Sending response: activitySteps count.',
  });
});

/** Get recent backend activity (what the backend/agents did). For frontend "agent activity" panel. With Clerk: returns activities for current user (empty if not authenticated). Without Clerk: returns anonymous activities. */
app.get('/api/backend-activity', (req, res) => {
  const userId = hasClerk && typeof getAuth === 'function' && getAuth(req)?.userId ? getAuth(req).userId : null;
  if (hasClerk && !userId) {
    return res.json({ activities: [] });
  }
  const limit = Math.min(parseInt(req.query?.limit, 10) || 50, 100);
  const rows = userId
    ? db.prepare('SELECT id, source, type, message, payload, created_at FROM backend_activity WHERE user_id = ? ORDER BY created_at DESC LIMIT ?').all(userId, limit)
    : db.prepare('SELECT id, source, type, message, payload, created_at FROM backend_activity WHERE user_id IS NULL ORDER BY created_at DESC LIMIT ?').all(limit);
  const activities = rows.map((r) => ({
    id: r.id,
    source: r.source,
    type: r.type,
    message: r.message,
    payload: r.payload ? (() => { try { return JSON.parse(r.payload); } catch { return null; } })() : null,
    timestamp: r.created_at,
  }));
  res.json({ activities });
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

/** JobAI agent system prompt: persona + tools (results injected by backend). No markdown—frontend shows plain text only. */
const JOBAI_SYSTEM_PROMPT = `You are jobAI, a friendly assistant that helps users land their next job offer.

IMPORTANT: Use plain text only. Do not use markdown: no asterisks for bold (**text**), no bullet asterisks (*), no hashtags for headers. Write in clear, readable sentences. The chat UI cannot render markdown.

Your tools (the backend runs them; you summarize the results for the user):
- scan_job_offers: Search for job listings. You will receive a list of jobs to present.
- get_application_status: Summarize the user's applications and follow-ups.
- suggest_profile_improvements: Suggest resume or profile improvements.

When the user wants to find jobs but has not given details yet, ask in plain text, for example:

"To find the best jobs for you, please tell me:
1) What kind of role you want (e.g. Software Engineer, Marketing Manager, Data Analyst)
2) Your key skills
3) Your preferred location (e.g. Singapore, Remote, London)

You can also upload your resume on the right and I can extract this for you. Once I have these details, I can find jobs that match your profile."

Keep answers concise and encouraging. When you receive tool results, present them clearly in plain text (list jobs with title and company). Never invent job titles or companies—only use data from the tool results. If no tool was run yet, invite the user to try "Find jobs that match my profile" or ask about their application status.`;

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

/** Infer display params from user message for "What the agent is doing" panel (e.g. role, location). */
function inferToolParams(userMessage, toolName) {
  const lower = (userMessage || '').toLowerCase();
  const params = {};
  if (toolName === 'scan_job_offers') {
    if (/\b(software\s*engineer|frontend|backend|full\s*stack|developer|engineer)\b/.test(lower)) {
      const m = lower.match(/(software\s*engineer|frontend|backend|full\s*stack|developer|engineer)/);
      params.role = (m ? m[1] : 'Software Engineer').replace(/\b\w/g, (c) => c.toUpperCase());
    } else params.role = 'From your profile';
    if (/\b(sg|singapore|remote|singapore\s*\/\s*hybrid)\b/.test(lower)) {
      if (/sg\b|singapore/.test(lower)) params.location = 'Singapore';
      else if (/remote/.test(lower)) params.location = 'Remote';
      else params.location = 'Singapore / Hybrid';
    } else params.location = 'From your profile';
  }
  if (toolName === 'get_application_status') params.summary = 'Application status';
  if (toolName === 'suggest_profile_improvements') params.summary = 'Profile suggestions';
  return params;
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

/** Max resume text length to inject (leave room for system prompt + conversation). */
const MAX_RESUME_CONTEXT_CHARS = 14000;

/** Build text to inject into system prompt: full resume content so the agent can answer any question from the resume. */
function buildUserProfileContext(extractedProfileJson, resumeText) {
  const intro = '\n\nThe user has uploaded a resume. Use the resume content below to answer any question about the user (name, title, experience, location, nationality, PR status, skills, projects, education, etc.). Answer only from what is stated in the resume. If something is not in the resume, say you do not have that information from the resume.\n\n--- Resume content ---\n';
  if (resumeText && typeof resumeText === 'string' && resumeText.trim()) {
    const text = resumeText.trim().length > MAX_RESUME_CONTEXT_CHARS
      ? resumeText.trim().slice(0, MAX_RESUME_CONTEXT_CHARS) + '\n[... truncated]'
      : resumeText.trim();
    return intro + text + '\n--- End resume ---';
  }
  if (extractedProfileJson && typeof extractedProfileJson === 'string') {
    try {
      const p = JSON.parse(extractedProfileJson);
      const profile = p?.profile || p;
      if (profile && typeof profile === 'object') {
        const parts = [];
        if (profile.name) parts.push('Name: ' + profile.name);
        if (profile.title) parts.push('Title: ' + profile.title);
        if (profile.experience) parts.push('Experience: ' + profile.experience);
        if (profile.location) parts.push('Location: ' + profile.location);
        if (profile.nationalityOrResidency) parts.push('Nationality/Residency: ' + profile.nationalityOrResidency);
        if (profile.email) parts.push('Email: ' + profile.email);
        if (profile.phone) parts.push('Phone: ' + profile.phone);
        if (Array.isArray(profile.skills) && profile.skills.length) parts.push('Skills: ' + profile.skills.join(', '));
        if (parts.length > 0) return intro + parts.join('\n') + '\n--- End resume ---';
      }
    } catch {
      // ignore
    }
  }
  return '';
}

/** Call Gemini API with system prompt and conversation history. */
async function getGeminiReply(contents, apiKey, userProfileContext = '') {
  if (!apiKey || !apiKey.trim()) return null;
  const model = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(apiKey.trim())}`;
  const systemText = JOBAI_SYSTEM_PROMPT + (userProfileContext || '');
  const body = {
    systemInstruction: { parts: [{ text: systemText }] },
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

/** Strip common markdown so the frontend (plain text) does not show ** or *. */
function stripMarkdown(text) {
  if (typeof text !== 'string') return text;
  return text
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/^#+\s*/gm, '')
    .replace(/^\s*[-*]\s+/gm, '• ')
    .trim();
}

/** Chat: JobAI agent with system prompt; optional history; runs mock tools and returns reply + agentUpdates. */
app.post('/api/chat', async (req, res) => {
  console.log('\n[CHAT] <<< Received from frontend');
  recordBackendActivity(req, { source: 'chat', type: 'agent_step', message: 'Chat request received from frontend' });
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
  let toolCall = null;
  let toolResultText = '';
  const toolName = getToolToRun(message);
  if (toolName) {
    toolCall = { name: toolName, params: inferToolParams(message, toolName) };
    recordBackendActivity(req, { source: 'chat', type: 'tool_call', message: `Running tool: ${toolName}` });
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
  let userProfileContext = '';
  if (hasClerk && getAuth(req)?.userId) {
    const resumeRow = db.prepare('SELECT extracted_profile, resume_text FROM user_resume WHERE user_id = ?').get(getAuth(req).userId);
    if (resumeRow && (resumeRow.resume_text || resumeRow.extracted_profile)) {
      userProfileContext = buildUserProfileContext(resumeRow.extracted_profile, resumeRow.resume_text);
    }
    // PLACEHOLDER (teammate): Load user links from user_links table, extract info from each URL (e.g. LinkedIn, portfolio),
    // and append to userProfileContext so the agent can use it as a data resource. No implementation yet.
  }
  let reply = 'No response';
  const geminiReply = await getGeminiReply(contents, apiKey, userProfileContext);
  if (geminiReply) reply = stripMarkdown(geminiReply);
  else if (!apiKey) console.log('[CHAT] No API key (set in Settings when using Clerk, or send apiKey/env)');
  recordBackendActivity(req, { source: 'chat', type: 'done', message: 'Reply sent to frontend' });
  console.log('[CHAT] >>> Sending reply + %d agentUpdate(s) to frontend\n', agentUpdates.length);
  return res.json({
    reply,
    agentUpdates: agentUpdates.length ? agentUpdates : undefined,
    toolCall: toolCall || undefined,
  });
});

/** Push one activity event (Cursor-like stream). type: planning | tool_call | agent_step | done. Backend stores; frontend displays. */
function pushActivityEvent(events, event) {
  events.push({
    id: String(events.length + 1),
    type: event.type || 'agent_step',
    agent: event.agent,
    message: event.message,
    status: event.status || 'done',
    timestamp: event.timestamp || new Date().toISOString(),
  });
}

/** Resume upload: saves file, then MCP-style extract (PDF text + Gemini) and returns extracted profile when possible.
 *  TODO: Pending fix – resume info cannot be retrieved correctly in all cases; backend agent may not get full profile.
 */
const { extractResumeProfile } = require('./resume-extract');

app.post('/api/resume/upload', upload.single('file'), async (req, res) => {
  console.log('\n[RESUME] <<< Received upload from frontend');
  if (!req.file) {
    console.log('[RESUME] bad request: no file in request');
    return res.status(400).json({ error: 'file required' });
  }
  const fileName = req.file.filename;
  const originalName = req.file.originalname || req.file.filename;
  const now = new Date().toISOString();
  let relativePath = fileName;
  if (hasClerk && getAuth(req)?.userId) {
    const userId = getAuth(req).userId;
    relativePath = `${userId}/${fileName}`;
    const prev = db.prepare('SELECT file_path FROM user_resume WHERE user_id = ?').get(userId);
    if (prev?.file_path) {
      const oldPath = path.join(UPLOADS_DIR, prev.file_path);
      if (fs.existsSync(oldPath)) {
        try {
          fs.unlinkSync(oldPath);
        } catch (e) {
          console.log('[RESUME] could not remove previous file:', e.message);
        }
      }
    }
    db.prepare(
      'INSERT INTO user_resume (user_id, file_path, original_name, uploaded_at) VALUES (?, ?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET file_path = ?, original_name = ?, uploaded_at = ?'
    ).run(userId, relativePath, originalName, now, relativePath, originalName, now);
    console.log('[RESUME] saved for user %s: %s (original: %s)\n', userId, relativePath, originalName);
  } else {
    console.log('[RESUME] saved to local: %s (original: %s, size: %s bytes)\n', fileName, originalName, req.file.size);
  }

  const absolutePath = path.join(UPLOADS_DIR, relativePath);
  let extracted = null;
  let greeting = null;
  let apiKey = (process.env.GEMINI_API_KEY || '').trim();
  if (hasClerk && getAuth(req)?.userId) {
    const row = db.prepare('SELECT encrypted_key FROM user_gemini_keys WHERE user_id = ?').get(getAuth(req).userId);
    if (row?.encrypted_key) apiKey = decryptFromDb(row.encrypted_key) || apiKey;
  }
  const activityEvents = [];
  const pushResumeActivity = (ev) => {
    const oneLine = summarizeActivityForFrontend('resume', ev.type, ev.message, ev.agent != null || ev.status != null ? { agent: ev.agent, status: ev.status } : undefined);
    pushActivityEvent(activityEvents, { ...ev, message: oneLine });
    recordBackendActivity(req, {
      source: 'resume',
      type: ev.type || 'agent_step',
      message: oneLine,
      payload: ev.agent != null || ev.status != null ? { agent: ev.agent, status: ev.status } : undefined,
    });
  };

  pushResumeActivity({
    type: 'agent_step',
    message: originalName ? `Received resume upload: ${originalName}` : 'Received resume upload',
    status: 'done',
  });
  pushResumeActivity({
    type: 'agent_step',
    message: 'File saved. Starting extraction.',
    status: 'running',
  });

  if (apiKey) {
    try {
      const result = await extractResumeProfile(absolutePath, apiKey, (ev) => pushResumeActivity(ev));
      if (result) {
        extracted = result.profile;
        greeting = result.greeting || null;
        const resumeText = result.resumeText || null;
        pushResumeActivity({
          type: 'done',
          message: 'Extraction complete. Ready for follow-up in chat.',
          status: 'done',
        });
        if (extracted && hasClerk && getAuth(req)?.userId) {
          console.log('[RESUME] Agent extracted profile for', originalName);
          db.prepare('UPDATE user_resume SET extracted_profile = ?, resume_text = ?, activity_steps = ? WHERE user_id = ?').run(
            JSON.stringify(extracted),
            resumeText ? resumeText.slice(0, 50000) : null,
            JSON.stringify(activityEvents),
            getAuth(req).userId
          );
        }
        if (greeting) console.log('[RESUME] Agent greeting:', greeting.slice(0, 80) + '...');
      }
    } catch (err) {
      console.log('[RESUME] Extract/analyze error:', err?.message);
      pushResumeActivity({ type: 'agent_step', message: `Extraction error: ${err?.message || 'Unknown'}`, status: 'done' });
    }
    if (hasClerk && getAuth(req)?.userId && !extracted) {
      db.prepare('UPDATE user_resume SET extracted_profile = NULL, resume_text = NULL, activity_steps = NULL WHERE user_id = ?').run(getAuth(req).userId);
    }
  } else {
    console.log('[RESUME] No Gemini API key – skip extraction (set in Settings or GEMINI_API_KEY)');
    pushResumeActivity({
      type: 'done',
      message: 'No API key set. Set Gemini API key in Settings to extract profile from resume.',
      status: 'done',
    });
  }

  const activityStepsForResponse = activityEvents.length > 0 ? activityEvents : undefined;
  console.log('[RESUME] >>> Sending response: activitySteps count =', activityEvents.length, activityStepsForResponse ? activityStepsForResponse.length : 0);
  if (activityEvents.length > 0) {
    activityEvents.forEach((ev, i) => console.log('[RESUME]   step', i + 1, ':', ev.message));
  }
  res.json({
    ok: true,
    savedPath: fileName,
    originalName,
    uploadedAt: now,
    extracted: extracted || undefined,
    greeting: greeting || undefined,
    activitySteps: activityStepsForResponse,
  });
});

/** Format backend_activity row into the same shape as activity steps (frontend displays whatever backend sends). */
function formatBackendActivityAsStep(row) {
  let payload = null;
  if (row.payload && typeof row.payload === 'string') {
    try {
      payload = JSON.parse(row.payload);
    } catch {
      payload = null;
    }
  } else if (row.payload && typeof row.payload === 'object') payload = row.payload;
  const agent = payload?.agent || (row.source === 'resume' ? 'Resume' : 'Chat');
  const status = payload?.status || 'done';
  return {
    id: String(row.id),
    type: row.type || 'agent_step',
    message: row.message,
    status,
    agent,
    timestamp: row.created_at,
  };
}

/** Get current user's resume info (Clerk required). Returns { fileName, originalName, uploadedAt, activitySteps? } or 404. activitySteps = resume steps + backend activity, formatted by backend. */
if (hasClerk) {
  app.get('/api/resume', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const row = db.prepare('SELECT file_path, original_name, uploaded_at, activity_steps FROM user_resume WHERE user_id = ?').get(auth.userId);
    if (!row) return res.status(404).json({ error: 'No resume' });
    let resumeSteps = [];
    if (row.activity_steps && typeof row.activity_steps === 'string') {
      try {
        const parsed = JSON.parse(row.activity_steps);
        if (Array.isArray(parsed)) resumeSteps = parsed;
      } catch {
        // ignore
      }
    }
    const limit = 50;
    const backendRows = db
      .prepare('SELECT id, source, type, message, payload, created_at FROM backend_activity WHERE user_id = ? ORDER BY created_at DESC LIMIT ?')
      .all(auth.userId, limit);
    const backendSteps = backendRows.map(formatBackendActivityAsStep);
    const combined = [...backendSteps, ...resumeSteps].sort((a, b) => {
      const ta = a.timestamp ? new Date(a.timestamp).getTime() : 0;
      const tb = b.timestamp ? new Date(b.timestamp).getTime() : 0;
      return tb - ta;
    });
    res.json({
      fileName: row.file_path,
      originalName: row.original_name,
      uploadedAt: row.uploaded_at,
      activitySteps: combined.length > 0 ? combined : undefined,
    });
  });

  /** Delete current user's resume from backend (file + DB). */
  app.delete('/api/resume', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const row = db.prepare('SELECT file_path FROM user_resume WHERE user_id = ?').get(auth.userId);
    if (!row) return res.status(404).json({ error: 'No resume' });
    const filePath = path.join(UPLOADS_DIR, row.file_path);
    if (fs.existsSync(filePath)) {
      try {
        fs.unlinkSync(filePath);
      } catch (e) {
        console.log('[RESUME] delete file error:', e.message);
      }
    }
    db.prepare('DELETE FROM user_resume WHERE user_id = ?').run(auth.userId);
    console.log('[RESUME] deleted for user %s\n', auth.userId);
    res.json({ ok: true });
  });

  /** Get current user's links (Clerk required). Returns { links: string[] }. Used by agent as data resource – teammate: implement extraction and inject into agent context. */
  app.get('/api/user/links', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const row = db.prepare('SELECT links FROM user_links WHERE user_id = ?').get(auth.userId);
    let links = [];
    if (row && row.links) {
      try {
        links = JSON.parse(row.links);
        if (!Array.isArray(links)) links = [];
      } catch {
        links = [];
      }
    }
    res.json({ links });
  });

  /** Save current user's links (Clerk required). Body { links: string[] }. Frontend sends these; backend stores only. Teammate: add logic to extract info from URLs and feed to agent as context. */
  app.put('/api/user/links', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const links = req.body?.links;
    if (!Array.isArray(links)) return res.status(400).json({ error: 'links array required' });
    const sanitized = links.filter((u) => typeof u === 'string' && u.trim()).slice(0, 20);
    const now = new Date().toISOString();
    const json = JSON.stringify(sanitized);
    db.prepare(
      'INSERT INTO user_links (user_id, links, updated_at) VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET links = ?, updated_at = ?'
    ).run(auth.userId, json, now, json, now);
    res.json({ ok: true, links: sanitized });
  });
}

app.listen(PORT, () => {
  console.log(`jobAI backend running at http://localhost:${PORT}`);
  console.log('Watch this terminal for [CHAT] and [RESUME] logs when the frontend sends requests.\n');
});
