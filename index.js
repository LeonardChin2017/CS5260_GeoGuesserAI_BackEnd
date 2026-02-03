require('dotenv').config();

const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
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
const clerkMissingMessage = 'Clerk backend is not configured. Set CLERK_SECRET_KEY and CLERK_PUBLISHABLE_KEY, then restart the backend.';

const googleClientId = (process.env.GOOGLE_CLIENT_ID || '').trim();
const googleClientSecret = (process.env.GOOGLE_CLIENT_SECRET || '').trim();
const googleRedirectUri = (process.env.GOOGLE_OAUTH_REDIRECT_URI || '').trim();
const googleScopes = (process.env.GMAIL_OAUTH_SCOPES || 'https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile').trim();
const jobStreetGoogleLoginUrl = (process.env.JOBSTREET_GOOGLE_LOGIN_URL || 'https://www.jobstreet.com.sg/en/login').trim();
const pendingGoogleOAuthStates = new Map();
const GOOGLE_STATE_TTL_MS = 10 * 60 * 1000;

function isGoogleOAuthConfigured() {
  return !!(googleClientId && googleClientSecret && googleRedirectUri && hasEncryptionSecret());
}

function createGoogleOAuthState(payload) {
  const state = crypto.randomBytes(18).toString('hex');
  pendingGoogleOAuthStates.set(state, {
    ...payload,
    expiresAt: Date.now() + GOOGLE_STATE_TTL_MS,
  });
  return state;
}

function consumeGoogleOAuthState(state) {
  if (!state || typeof state !== 'string') return null;
  const entry = pendingGoogleOAuthStates.get(state);
  if (!entry) return null;
  pendingGoogleOAuthStates.delete(state);
  if (entry.expiresAt < Date.now()) return null;
  return entry;
}

setInterval(() => {
  const now = Date.now();
  for (const [key, value] of pendingGoogleOAuthStates.entries()) {
    if (value.expiresAt < now) pendingGoogleOAuthStates.delete(key);
  }
}, GOOGLE_STATE_TTL_MS).unref?.();

function buildGoogleOAuthUrl(state) {
  const params = new URLSearchParams({
    client_id: googleClientId,
    redirect_uri: googleRedirectUri,
    response_type: 'code',
    access_type: 'offline',
    include_granted_scopes: 'true',
    prompt: 'consent',
    scope: googleScopes,
    state,
  });
  return `https://accounts.google.com/o/oauth2/v2/auth?${params.toString()}`;
}

function sendGmailCallbackPage(res, { success, message }) {
  const safeMessage = (message || '').replace(/[<>]/g, (ch) => (ch === '<' ? '&lt;' : '&gt;')) || (success ? 'Gmail connected. You may close this tab.' : 'Unable to connect Gmail.');
  const title = success ? 'Gmail connected' : 'Gmail connection failed';
  const color = success ? '#065f46' : '#b91c1c';
  const border = success ? '#34d399' : '#f87171';
  const statusText = success ? 'Success' : 'Error';
  const html = `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>${title}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body { font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
      .card { background: rgba(15,23,42,0.9); border-radius: 16px; padding: 32px; border: 1px solid ${border}; max-width: 460px; box-shadow: 0 25px 70px rgba(15,23,42,0.65); text-align: center; }
      h1 { margin: 0 0 12px; font-size: 24px; }
      p { margin: 0; line-height: 1.5; font-size: 15px; }
      .status { font-size: 13px; letter-spacing: 0.2em; text-transform: uppercase; color: ${color}; margin-bottom: 16px; }
      .hint { margin-top: 18px; font-size: 13px; opacity: 0.75; }
    </style>
  </head>
  <body>
    <div class="card">
      <div class="status">${statusText}</div>
      <h1>${title}</h1>
      <p>${safeMessage}</p>
      <p class="hint">You can close this tab and return to jobAI.</p>
    </div>
    <script>
      setTimeout(() => { try { window.close(); } catch (_) {} }, ${success ? 2500 : 6000});
    </script>
  </body>
</html>`;
  res.status(success ? 200 : 400).send(html);
}

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

function getStoredEmailTokens(userId, provider) {
  if (!userId || !provider) return null;
  let row;
  try {
    row = db.prepare('SELECT encrypted_tokens FROM user_email_tokens WHERE user_id = ? AND provider = ?').get(userId, provider);
  } catch {
    row = null;
  }
  if (!row?.encrypted_tokens) return null;
  const plain = decryptFromDb(row.encrypted_tokens);
  if (!plain) return null;
  try {
    return JSON.parse(plain);
  } catch {
    return null;
  }
}

function saveEmailTokens({ userId, provider, tokens, email, scope, expiresAt }) {
  if (!userId || !provider || !tokens) return false;
  const encrypted = encryptForDb(JSON.stringify(tokens));
  if (!encrypted) return false;
  const now = new Date().toISOString();
  db.prepare(
    'INSERT INTO user_email_tokens (user_id, provider, email, encrypted_tokens, scope, expires_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(user_id, provider) DO UPDATE SET email = excluded.email, encrypted_tokens = excluded.encrypted_tokens, scope = excluded.scope, expires_at = excluded.expires_at, updated_at = excluded.updated_at'
  ).run(userId, provider, email || null, encrypted, scope || null, expiresAt || null, now, now);
  return true;
}

async function revokeGoogleToken(token) {
  if (!token) return;
  try {
    await fetch('https://oauth2.googleapis.com/revoke', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ token }),
    });
  } catch {
    // ignore revoke errors
  }
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
  CREATE TABLE IF NOT EXISTS user_deepseek_keys (
    user_id TEXT PRIMARY KEY,
    encrypted_key TEXT NOT NULL,
    updated_at TEXT NOT NULL
  )
`);
db.exec(`
  CREATE TABLE IF NOT EXISTS user_llm_preference (
    user_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
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
  CREATE TABLE IF NOT EXISTS user_portal_credentials (
    user_id TEXT NOT NULL,
    portal TEXT NOT NULL,
    encrypted_credentials TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, portal)
  )
`);
db.exec(`
  CREATE TABLE IF NOT EXISTS user_email_tokens (
    user_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    email TEXT,
    encrypted_tokens TEXT NOT NULL,
    scope TEXT,
    expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, provider)
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

app.get('/api/email/gmail/callback', async (req, res) => {
  if (!isGoogleOAuthConfigured()) {
    return sendGmailCallbackPage(res, { success: false, message: 'Backend missing Google OAuth configuration. Ask your jobAI admin to set GOOGLE_CLIENT_ID/SECRET and GOOGLE_OAUTH_REDIRECT_URI.' });
  }
  if (typeof req.query?.error === 'string' && req.query.error) {
    return sendGmailCallbackPage(res, { success: false, message: `Google returned an error: ${req.query.error}` });
  }
  const code = typeof req.query?.code === 'string' ? req.query.code : null;
  const stateParam = typeof req.query?.state === 'string' ? req.query.state : null;
  if (!stateParam) {
    return sendGmailCallbackPage(res, { success: false, message: 'Missing OAuth state. Please restart Gmail Connect from jobAI Settings.' });
  }
  const stateData = consumeGoogleOAuthState(stateParam);
  if (!stateData?.userId) {
    return sendGmailCallbackPage(res, { success: false, message: 'Your login session expired. Please start the Gmail connection again from jobAI.' });
  }
  if (!code) {
    return sendGmailCallbackPage(res, { success: false, message: 'Authorization code not provided by Google.' });
  }
  try {
    const tokenRes = await fetch('https://oauth2.googleapis.com/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        code,
        client_id: googleClientId,
        client_secret: googleClientSecret,
        redirect_uri: googleRedirectUri,
        grant_type: 'authorization_code',
      }),
    });
    const tokenJson = await tokenRes.json().catch(() => ({}));
    if (!tokenRes.ok || !tokenJson?.access_token) {
      const errMsg = tokenJson?.error_description || tokenJson?.error || 'Failed to exchange authorization code with Google.';
      throw new Error(errMsg);
    }
    const existingTokens = getStoredEmailTokens(stateData.userId, 'gmail');
    const expiresAtIso = tokenJson.expires_in ? new Date(Date.now() + Number(tokenJson.expires_in) * 1000).toISOString() : null;
    const storedTokens = {
      access_token: tokenJson.access_token,
      refresh_token: tokenJson.refresh_token || existingTokens?.refresh_token || null,
      scope: tokenJson.scope || googleScopes,
      token_type: tokenJson.token_type || 'Bearer',
      id_token: tokenJson.id_token || null,
      expires_in: tokenJson.expires_in || null,
      obtained_at: Date.now(),
    };
    let profileEmail = null;
    try {
      const profileRes = await fetch('https://www.googleapis.com/oauth2/v2/userinfo', {
        headers: { Authorization: `Bearer ${tokenJson.access_token}` },
      });
      if (profileRes.ok) {
        const profile = await profileRes.json().catch(() => ({}));
        if (profile?.email) profileEmail = profile.email;
      }
    } catch {
      // ignore profile fetch errors
    }
    const saved = saveEmailTokens({
      userId: stateData.userId,
      provider: 'gmail',
      tokens: storedTokens,
      email: profileEmail,
      scope: storedTokens.scope,
      expiresAt: expiresAtIso,
    });
    if (!saved) throw new Error('Server was unable to store Gmail credentials. Contact support.');
    recordBackendActivity(stateData.userId, {
      source: 'email',
      type: 'agent_step',
      message: profileEmail ? `Gmail connected (${profileEmail}).` : 'Gmail connected.',
    });
    return sendGmailCallbackPage(res, {
      success: true,
      message: profileEmail ? `Gmail connected as ${profileEmail}. You can close this tab.` : 'Gmail connected. You can close this tab.',
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unexpected error while connecting Gmail.';
    recordBackendActivity(stateData.userId, {
      source: 'email',
      type: 'agent_step',
      message: `Gmail connect failed: ${msg}`,
    });
    return sendGmailCallbackPage(res, { success: false, message: msg });
  }
});

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
function recordBackendActivity(reqOrUserId, { source, type, message, payload }) {
  let userId = null;
  if (typeof reqOrUserId === 'string') {
    userId = reqOrUserId;
  } else if (reqOrUserId && hasClerk && typeof getAuth === 'function') {
    try {
      userId = getAuth(reqOrUserId)?.userId || null;
    } catch {
      userId = null;
    }
  }
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

/** Simple hello – for testing connectivity only. */
app.get('/api/hello', (req, res) => {
  res.json({ message: 'Hello from jobAI backend', env: process.env.NODE_ENV || 'development' });
});

/** Debug: full trace. GET /api/debug/activity or GET /api/debug/activity?limit=50 for recent backend_activity and server info. */
app.get('/api/debug/activity', (req, res) => {
  const limit = Math.min(parseInt(req.query?.limit, 10) || 50, 100);
  let rows = [];
  try {
    rows = db.prepare('SELECT id, user_id, source, type, message, payload, created_at FROM backend_activity ORDER BY id DESC LIMIT ?').all(limit);
  } catch (e) {
    rows = [];
  }
  res.json({
    debug: true,
    timestamp: new Date().toISOString(),
    backend_activity_count: rows.length,
    backend_activity_recent: rows.map((r) => ({
      id: r.id,
      user_id: r.user_id,
      source: r.source,
      type: r.type,
      message: r.message,
      payload: r.payload ? (() => { try { return JSON.parse(r.payload); } catch { return r.payload; } })() : null,
      created_at: r.created_at,
    })),
    server: {
      env: process.env.NODE_ENV || 'development',
      port: process.env.PORT || 3001,
      hasClerk: !!hasClerk,
    },
    request: { method: req.method, path: req.path, query: req.query },
    note: 'Activity steps are in POST /api/resume/upload (activitySteps) and GET /api/resume (activitySteps).',
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

  app.get('/api/user/deepseek-key', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const row = db.prepare('SELECT 1 FROM user_deepseek_keys WHERE user_id = ?').get(auth.userId);
    res.json({ hasKey: !!row });
  });

  app.put('/api/user/deepseek-key', (req, res) => {
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
      'INSERT INTO user_deepseek_keys (user_id, encrypted_key, updated_at) VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET encrypted_key = ?, updated_at = ?'
    ).run(userId, encrypted, now, encrypted, now);
    res.json({ ok: true });
  });

  app.delete('/api/user/deepseek-key', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const userId = auth.userId;
    db.prepare('DELETE FROM user_deepseek_keys WHERE user_id = ?').run(userId);
    res.json({ ok: true });
  });

  app.get('/api/user/llm-provider', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const row = db.prepare('SELECT provider FROM user_llm_preference WHERE user_id = ?').get(auth.userId);
    const provider = row?.provider === 'deepseek' ? 'deepseek' : 'gemini';
    res.json({ provider });
  });

  app.put('/api/user/llm-provider', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const userId = auth.userId;
    const raw = typeof req.body?.provider === 'string' ? req.body.provider.trim().toLowerCase() : '';
    const provider = raw === 'deepseek' ? 'deepseek' : 'gemini';
    const now = new Date().toISOString();
    db.prepare(
      'INSERT INTO user_llm_preference (user_id, provider, updated_at) VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET provider = ?, updated_at = ?'
    ).run(userId, provider, now, provider, now);
    res.json({ ok: true, provider });
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

  /** Get portal connection status (Clerk auth required). Returns { jobstreet: { connected }, mycareersfuture: { connected } }. */
  app.get('/api/user/portal-credentials', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const rows = db.prepare('SELECT portal FROM user_portal_credentials WHERE user_id = ?').all(auth.userId);
    const portals = new Set(rows.map((r) => r.portal));
    res.json({
      jobstreet: { connected: portals.has('jobstreet') },
      mycareersfuture: { connected: portals.has('mycareersfuture') },
    });
  });

  /** Save portal credentials (Clerk auth required). Body { portal: 'jobstreet', email, password }. POC: JobStreet only. Stored encrypted. */
  app.put('/api/user/portal-credentials', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const { portal, email, password } = req.body || {};
    if (portal !== 'jobstreet' || !email || typeof password !== 'string') {
      return res.status(400).json({ error: 'portal must be "jobstreet", and email and password required' });
    }
    const plain = JSON.stringify({ email: String(email).trim(), password: String(password) });
    const encrypted = encryptForDb(plain);
    if (!encrypted) return res.status(500).json({ error: 'Encryption not configured (set ENCRYPTION_SECRET)' });
    const now = new Date().toISOString();
    db.prepare(
      'INSERT INTO user_portal_credentials (user_id, portal, encrypted_credentials, updated_at) VALUES (?, ?, ?, ?) ON CONFLICT(user_id, portal) DO UPDATE SET encrypted_credentials = ?, updated_at = ?'
    ).run(auth.userId, portal, encrypted, now, encrypted, now);
    res.json({ ok: true, portal });
  });

  /** Disconnect a portal (Clerk auth required). Query ?portal=jobstreet (POC: JobStreet only). */
  app.delete('/api/user/portal-credentials', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const portal = req.query?.portal;
    if (portal !== 'jobstreet') {
      return res.status(400).json({ error: 'Query param portal=jobstreet required' });
    }
    db.prepare('DELETE FROM user_portal_credentials WHERE user_id = ? AND portal = ?').run(auth.userId, portal);
    res.json({ ok: true, portal });
  });

  app.get('/api/email/gmail/status', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    let row;
    try {
      row = db.prepare('SELECT email, expires_at FROM user_email_tokens WHERE user_id = ? AND provider = ?').get(auth.userId, 'gmail');
    } catch {
      row = null;
    }
    if (!row) return res.json({ connected: false });
    res.json({ connected: true, email: row.email || null, expiresAt: row.expires_at || null });
  });

  app.post('/api/email/gmail/connect', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    if (!isGoogleOAuthConfigured()) {
      return res.status(503).json({ error: 'Google OAuth is not configured on the backend (set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_OAUTH_REDIRECT_URI, ENCRYPTION_SECRET).' });
    }
    const state = createGoogleOAuthState({ userId: auth.userId, purpose: 'gmail-connect' });
    const authUrl = buildGoogleOAuthUrl(state);
    recordBackendActivity(req, { source: 'email', type: 'agent_step', message: 'Launching Gmail OAuth connect flow.' });
    res.json({ authUrl });
  });

  app.delete('/api/email/gmail/connect', async (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const userId = auth.userId;
    const tokens = getStoredEmailTokens(userId, 'gmail');
    db.prepare('DELETE FROM user_email_tokens WHERE user_id = ? AND provider = ?').run(userId, 'gmail');
    if (tokens?.refresh_token) await revokeGoogleToken(tokens.refresh_token);
    else if (tokens?.access_token) await revokeGoogleToken(tokens.access_token);
    recordBackendActivity(req, { source: 'email', type: 'agent_step', message: 'Gmail disconnected.' });
    res.json({ ok: true });
  });

  app.post('/api/user/portal-google-login', (req, res) => {
    const auth = getAuth(req);
    if (!auth?.userId) return res.status(401).json({ error: 'Unauthorized' });
    const url = jobStreetGoogleLoginUrl && jobStreetGoogleLoginUrl.startsWith('http')
      ? jobStreetGoogleLoginUrl
      : 'https://www.jobstreet.com.sg/en/login';
    res.json({ authUrl: url });
  });
} else {
  const respondClerkMissing = (req, res) => res.status(503).json({ error: clerkMissingMessage });
  app.get('/api/user/gemini-key', respondClerkMissing);
  app.put('/api/user/gemini-key', respondClerkMissing);
  app.delete('/api/user/gemini-key', respondClerkMissing);
  app.get('/api/user/deepseek-key', respondClerkMissing);
  app.put('/api/user/deepseek-key', respondClerkMissing);
  app.delete('/api/user/deepseek-key', respondClerkMissing);
  app.get('/api/user/llm-provider', respondClerkMissing);
  app.put('/api/user/llm-provider', respondClerkMissing);
  app.get('/api/user/chat', respondClerkMissing);
  app.post('/api/user/chat', respondClerkMissing);
  app.delete('/api/user/chat', respondClerkMissing);
  app.get('/api/user/portal-credentials', respondClerkMissing);
  app.put('/api/user/portal-credentials', respondClerkMissing);
  app.delete('/api/user/portal-credentials', respondClerkMissing);
  app.get('/api/email/gmail/status', respondClerkMissing);
  app.post('/api/email/gmail/connect', respondClerkMissing);
  app.delete('/api/email/gmail/connect', respondClerkMissing);
  app.post('/api/user/portal-google-login', respondClerkMissing);
}

function readPromptFile(fileName, fallback) {
  try {
    const promptPath = path.join(__dirname, 'prompts', fileName);
    const raw = fs.readFileSync(promptPath, 'utf8');
    const cleaned = raw.replace(/\r\n/g, '\n').trim();
    return cleaned || fallback;
  } catch {
    return fallback;
  }
}

/** Exam question drafting system prompt: persona. No markdown—frontend shows plain text only. */
const JOBAI_SYSTEM_PROMPT = readPromptFile(
  'system.txt',
  `You are an exam question drafting assistant. Your job is to generate clear, well-structured exam questions for a given subject, topic, and difficulty.

IMPORTANT: Use plain text only. Do not use markdown: no asterisks for bold (**text**), no bullet asterisks (*), no hashtags for headers. Write in clear, readable sentences. The chat UI cannot render markdown.

When the user asks for questions:
- Ask one brief clarifying question if the topic, level, or question type is missing.
- Otherwise generate a concise set of questions in plain text.
- Prefer numbered questions, one per line, and include the question type if relevant (e.g. Multiple Choice, Short Answer, Essay).

Keep answers concise and helpful.`
);

/** Format-guard prompt: normalize output into a clean question list. */
const QUESTION_FORMAT_GUARD_PROMPT = readPromptFile(
  'format_guard.txt',
  `You are a format-guard agent. You only format and clean the assistant's output.

Rules:
- Output plain text only. No markdown, no bullet symbols, no headers.
- Return only the questions, one per line.
- Each line must start with a number and a period (e.g. "1. ...").
- If a question is missing a question mark, add one.
- Remove any preambles, explanations, or extra text.
- Keep the original meaning of each question.`
);

/** Get decrypted portal credentials for a user (for Playwright auto-search). Returns { jobstreet?: { email, password }, mycareersfuture?: { email, password } }. */
function getPortalCredentialsForUser(userId) {
  if (!userId || !hasEncryptionSecret()) return {};
  const rows = db.prepare('SELECT portal, encrypted_credentials FROM user_portal_credentials WHERE user_id = ?').all(userId);
  const out = {};
  for (const row of rows) {
    const plain = decryptFromDb(row.encrypted_credentials);
    if (!plain) continue;
    try {
      const cred = JSON.parse(plain);
      if (cred && typeof cred.email === 'string' && typeof cred.password === 'string') {
        out[row.portal] = { email: cred.email.trim(), password: cred.password };
      }
    } catch {
      // ignore
    }
  }
  return out;
}

/** Optional: run job search on a portal using Playwright (login + scrape). Only JobStreet supports email/password; MCF uses Singpass. Returns null if Playwright unavailable or portal not supported. */
async function runPlaywrightPortalSearch(portal, query, location, credentials, onStatusUpdate) {
  if (portal !== 'jobstreet' || !credentials?.email || !credentials?.password) return null;
  let playwright;
  try {
    playwright = require('playwright');
  } catch (e) {
    console.log('[TOOL] Playwright not installed; skipping portal auto-search. Run: npm install playwright && npx playwright install chromium');
    return null;
  }
  const portalLinks = {
    jobstreet: `https://www.jobstreet.com.sg/en/job-search/job-vacancy.php?key=${encodeURIComponent(query)}&location=${encodeURIComponent(location)}`,
    mycareersfuture: `https://www.mycareersfuture.gov.sg/search?search=${encodeURIComponent(query + ' ' + location)}&sortBy=last_posted&page=0`,
  };
  let browser;
  try {
    if (onStatusUpdate) onStatusUpdate({ type: 'tool_call', agent: 'Playwright', message: 'Opening JobStreet (logged-in search)...', status: 'running' });
    browser = await playwright.chromium.launch({ headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'] });
    const context = await browser.newContext({ userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36' });
    const page = await context.newPage();
    page.setDefaultTimeout(25000);

    // JobStreet Express SG login
    if (onStatusUpdate) onStatusUpdate({ type: 'tool_call', agent: 'Playwright', message: 'Logging in to JobStreet...', status: 'running' });
    await page.goto('https://sg.jobstreetexpress.com/login', { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('input[type="email"], input[name="email"], input[id="email"], input[type="text"]', { timeout: 10000 }).catch(() => null);
    const emailSel = await page.$('input[type="email"]') || await page.$('input[name="email"]') || await page.$('input[id="email"]') || await page.$('input[type="text"]');
    const passSel = await page.$('input[type="password"]');
    if (!emailSel || !passSel) {
      console.log('[TOOL] JobStreet login form not found; falling back to web search.');
      await browser.close();
      return null;
    }
    await emailSel.fill(credentials.email);
    await passSel.fill(credentials.password);
    const submitBtn = await page.$('button[type="submit"]') || await page.$('input[type="submit"]') || await page.$('button').catch(() => null);
    if (submitBtn) await submitBtn.click().catch(() => page.keyboard.press('Enter'));
    else await page.keyboard.press('Enter');
    await page.waitForNavigation({ waitUntil: 'domcontentloaded', timeout: 15000 }).catch(() => {});

    // Search: JobStreet Express uses /jobs-in-Singapore or similar; try search with query
    if (onStatusUpdate) onStatusUpdate({ type: 'tool_call', agent: 'Playwright', message: 'Searching jobs on JobStreet...', status: 'running' });
    const searchPath = `/jobs-in-${(location || 'Singapore').replace(/\s+/g, '-')}`;
    const searchUrl = `https://sg.jobstreetexpress.com${searchPath}?keyword=${encodeURIComponent(query)}`;
    await page.goto(searchUrl, { waitUntil: 'domcontentloaded', timeout: 15000 }).catch(() => {});

    const jobs = [];
    const cards = await page.$$('article a[href*="/job"], a[href*="/jobs/"] [class*="job"], [data-testid*="job"], [class*="JobCard"], .job-card, [class*="listing"]');
    for (let i = 0; i < Math.min(cards.length, 15); i++) {
      try {
        const el = cards[i];
        const href = await el.getAttribute('href').catch(() => '');
        const text = await el.innerText().catch(() => '');
        if (!href || !text || text.length < 5) continue;
        const fullUrl = href.startsWith('http') ? href : `https://sg.jobstreetexpress.com${href.startsWith('/') ? '' : '/'}${href}`;
        const title = text.split('\n')[0].trim().slice(0, 120) || 'Job';
        jobs.push({
          id: String(i + 1),
          title,
          company: 'Company',
          location: location || 'Singapore',
          match: `${85 + Math.floor(Math.random() * 10)}%`,
          posted: 'Recently',
          link: fullUrl,
        });
      } catch {
        // skip
      }
    }
    if (onStatusUpdate) onStatusUpdate({ type: 'tool_call', agent: 'Playwright', message: `Found ${jobs.length} jobs on JobStreet (logged in)`, status: 'done' });
    await browser.close();
    return { type: 'job_scan', jobs, portalLinks };
  } catch (err) {
    console.log('[TOOL] Playwright JobStreet search failed:', err.message);
    if (onStatusUpdate) onStatusUpdate({ type: 'tool_call', agent: 'Playwright', message: `Portal search failed: ${err.message}`, status: 'done' });
    if (browser) await browser.close().catch(() => {});
    return null;
  }
}

/** Real web search for job offers using DuckDuckGo HTML scraping. Free, no API key needed. */
async function runWebSearchJobs(query, location = 'Singapore', onStatusUpdate) {
  if (onStatusUpdate) onStatusUpdate({ type: 'tool_call', agent: 'WebSearch', message: `Searching for "${query}" jobs in ${location}`, status: 'running' });

  try {
    // Search DuckDuckGo HTML (lite version is faster)
    const searchQuery = `${query} jobs ${location}`;
    const url = `https://lite.duckduckgo.com/lite/?q=${encodeURIComponent(searchQuery)}`;
    
    if (onStatusUpdate) onStatusUpdate({ type: 'tool_call', agent: 'WebSearch', message: `Connecting to DuckDuckGo...`, status: 'running' });
    
    const res = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
      },
    });
    
    if (!res.ok) {
      console.log('[TOOL] DuckDuckGo error:', res.status, res.statusText);
      throw new Error(`DuckDuckGo returned ${res.status}`);
    }
    
    if (onStatusUpdate) onStatusUpdate({ type: 'tool_call', agent: 'WebSearch', message: `Fetching search results...`, status: 'running' });
    
    const html = await res.text();
    console.log('[TOOL] DuckDuckGo HTML length:', html.length, 'chars');
    
    if (onStatusUpdate) onStatusUpdate({ type: 'tool_call', agent: 'WebSearch', message: `Parsing HTML results (${Math.round(html.length / 1000)}KB)...`, status: 'running' });
    
    const jobs = [];
    const seenUrls = new Set();
    const websitesAccessed = new Set();
    
    // Extract all external HTTP/HTTPS links from the HTML
    const linkPattern = /<a[^>]*href="(https?:\/\/[^"]+)"[^>]*>([\s\S]{5,500}?)<\/a>/gi;
    const candidateLinks = [];
    let linkMatch;
    
    while ((linkMatch = linkPattern.exec(html)) !== null && candidateLinks.length < 100) {
      const url = linkMatch[1];
      const linkText = linkMatch[2].replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
      
      // Skip DuckDuckGo internal links
      if (url.includes('duckduckgo.com') || url.includes('duck.com')) continue;
      
      // Track websites accessed
      try {
        const urlObj = new URL(url);
        const domain = urlObj.hostname.replace('www.', '');
        websitesAccessed.add(domain);
      } catch {
        // ignore
      }
      
      // Filter for job-related content
      const lowerText = linkText.toLowerCase();
      const lowerUrl = url.toLowerCase();
      const isJobRelated = 
        lowerText.includes('job') || 
        lowerText.includes('career') || 
        lowerText.includes('hiring') ||
        lowerText.includes('position') ||
        lowerText.includes('opportunity') ||
        lowerText.includes('recruit') ||
        lowerText.includes('apply') ||
        lowerUrl.includes('linkedin.com/jobs') ||
        lowerUrl.includes('indeed.com') ||
        lowerUrl.includes('jobstreet') ||
        lowerUrl.includes('glassdoor.com') ||
        lowerUrl.includes('monster.com') ||
        lowerUrl.includes('ziprecruiter.com') ||
        lowerUrl.includes('jobs.') ||
        lowerUrl.includes('/jobs/') ||
        lowerUrl.includes('/careers/') ||
        lowerUrl.includes('/job/');
      
      if (isJobRelated && linkText.length > 8) {
        candidateLinks.push({ url, text: linkText });
      }
    }
    
    console.log('[TOOL] Found', candidateLinks.length, 'candidate job links from DuckDuckGo');
    
    if (onStatusUpdate) onStatusUpdate({ type: 'tool_call', agent: 'WebSearch', message: `Found ${candidateLinks.length} candidate links, extracting job listings...`, status: 'running' });
    
    // Process candidates into job listings
    let idx = 0;
    for (const candidate of candidateLinks) {
      if (idx >= 10) break;
      if (seenUrls.has(candidate.url)) continue;
      seenUrls.add(candidate.url);
      
      // Extract title (clean HTML and split by common separators)
      let title = candidate.text
        .split(' - ')[0]
        .split(' | ')[0]
        .split(' : ')[0]
        .split(' · ')[0]
        .split(' • ')[0]
        .trim();
      
      // Extract company name from title
      let company = 'Company';
      const companyPatterns = [
        /at\s+([^|•\-–\n,·]+)/i,
        /@\s*([^|•\-–\n,·]+)/i,
        /-\s*([^|•\-–\n,·]+)\s*$/i,
        /:\s*([^|•\-–\n,·]+)/i,
        /·\s*([^|•\-–\n,]+)/i,
        /,\s*([^|•\-–\n]+)/i,
      ];
      
      for (const pattern of companyPatterns) {
        const m = title.match(pattern);
        if (m && m[1] && m[1].trim().length > 1) {
          company = m[1].trim().split(/\s+/).slice(0, 4).join(' ');
          title = title.replace(pattern, '').trim();
          break;
        }
      }
      
      // If no company in title, try extracting from URL
      if (company === 'Company') {
        try {
          const urlObj = new URL(candidate.url);
          const hostname = urlObj.hostname.replace('www.', '');
          const parts = hostname.split('.');
          if (parts.length > 2 && parts[0] !== 'jobs' && parts[0] !== 'www' && parts[0].length > 2) {
            company = parts[0].charAt(0).toUpperCase() + parts[0].slice(1).replace(/-/g, ' ');
          } else if (parts.length === 2 && parts[0] !== 'jobs') {
            company = parts[0].charAt(0).toUpperCase() + parts[0].slice(1).replace(/-/g, ' ');
          }
        } catch {
          // ignore
        }
      }
      
      // Clean title further
      title = title.replace(/^[^a-zA-Z0-9]*/, '').trim();
      
      // Skip if title is too generic or too short
      if (!title || title.length < 5 || title.toLowerCase() === 'job opening' || title.toLowerCase() === 'more') {
        continue;
      }
      
      jobs.push({
        id: String(idx + 1),
        title: title,
        company: company,
        location: location,
        match: `${85 + Math.floor(Math.random() * 10)}%`,
        posted: 'Recently',
        link: candidate.url,
      });
      idx++;
    }
    
    // Final status update with summary
    if (onStatusUpdate) {
      if (jobs.length > 0) {
        const websiteList = Array.from(websitesAccessed).slice(0, 5).join(', ');
        const moreSites = websitesAccessed.size > 5 ? ` and ${websitesAccessed.size - 5} more` : '';
        onStatusUpdate({ 
          type: 'tool_call', 
          agent: 'WebSearch', 
          message: `Found ${jobs.length} job listings from ${websitesAccessed.size} website${websitesAccessed.size !== 1 ? 's' : ''} (${websiteList}${moreSites})`, 
          status: 'done' 
        });
      } else {
        const websiteList = Array.from(websitesAccessed).slice(0, 3).join(', ');
        onStatusUpdate({ 
          type: 'tool_call', 
          agent: 'WebSearch', 
          message: `Searched ${websitesAccessed.size} website${websitesAccessed.size !== 1 ? 's' : ''} (${websiteList}) but no job listings found`, 
          status: 'done' 
        });
      }
    }
    
    // Only return jobs if we found real ones - no fallback to mock data
    if (jobs.length === 0) {
      console.log('[TOOL] No jobs found in DuckDuckGo results for query:', searchQuery);
      const q = encodeURIComponent(`${query} ${location}`);
      const portalLinks = {
        jobstreet: `https://www.jobstreet.com.sg/en/job-search/job-vacancy.php?key=${encodeURIComponent(query)}&location=${encodeURIComponent(location)}`,
        mycareersfuture: `https://www.mycareersfuture.gov.sg/search?search=${encodeURIComponent(query + ' ' + location)}&sortBy=last_posted&page=0`,
      };
      return { type: 'job_scan', jobs: null, portalLinks };
    }
    
    console.log('[TOOL] Successfully extracted', jobs.length, 'job listings from', websitesAccessed.size, 'websites');
    // Deep links for JobStreet and MyCareersFuture (user can open, log in there, see correct results)
    const q = encodeURIComponent(`${query} ${location}`);
    const portalLinks = {
      jobstreet: `https://www.jobstreet.com.sg/en/job-search/job-vacancy.php?key=${encodeURIComponent(query)}&location=${encodeURIComponent(location)}`,
      mycareersfuture: `https://www.mycareersfuture.gov.sg/search?search=${encodeURIComponent(query + ' ' + location)}&sortBy=last_posted&page=0`,
    };
    return { type: 'job_scan', jobs, portalLinks };
  } catch (err) {
    console.log('[TOOL] DuckDuckGo search failed:', err.message);
    if (onStatusUpdate) onStatusUpdate({ type: 'tool_call', agent: 'WebSearch', message: `Search failed: ${err.message}`, status: 'done' });
    const portalLinks = {
      jobstreet: `https://www.jobstreet.com.sg/en/job-search/job-vacancy.php?key=${encodeURIComponent(query)}&location=${encodeURIComponent(location)}`,
      mycareersfuture: `https://www.mycareersfuture.gov.sg/search?search=${encodeURIComponent(query + ' ' + location)}&sortBy=last_posted&page=0`,
    };
    return { type: 'job_scan', jobs: null, portalLinks };
  }
}

/** Mock: "scan" job offers (fallback when web search unavailable). */
function runScanJobOffers(query = 'software', location = 'Singapore') {
  const jobs = [
    { id: '1', title: 'Senior Software Engineer', company: 'TechCorp Singapore', location: 'Singapore', match: '92%', posted: '2 days ago' },
    { id: '2', title: 'Full Stack Developer', company: 'StartupXYZ', location: 'Remote', match: '88%', posted: '1 day ago' },
    { id: '3', title: 'Frontend Engineer', company: 'ProductLabs', location: 'Singapore', match: '85%', posted: '3 days ago' },
    { id: '4', title: 'Software Developer', company: 'FinanceHub', location: 'Singapore / Hybrid', match: '82%', posted: '5 days ago' },
  ];
  const portalLinks = {
    jobstreet: `https://www.jobstreet.com.sg/en/job-search/job-vacancy.php?key=${encodeURIComponent(query)}&location=${encodeURIComponent(location)}`,
    mycareersfuture: `https://www.mycareersfuture.gov.sg/search?search=${encodeURIComponent(query + ' ' + location)}&sortBy=last_posted&page=0`,
  };
  return { type: 'job_scan', jobs, portalLinks };
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

/** Run a tool by name and return agentUpdate for frontend. Supports async tools. options.getPortalCredentials = async () => ({ jobstreet? }). */
async function runTool(name, params = {}, onStatusUpdate = null, options = {}) {
  switch (name) {
    case 'web_search_jobs':
    case 'scan_job_offers': {
      const query = params.query || params.role || 'software engineer';
      const location = params.location || 'Singapore';
      const portalLinks = {
        jobstreet: `https://www.jobstreet.com.sg/en/job-search/job-vacancy.php?key=${encodeURIComponent(query)}&location=${encodeURIComponent(location)}`,
        mycareersfuture: `https://www.mycareersfuture.gov.sg/search?search=${encodeURIComponent(query + ' ' + location)}&sortBy=last_posted&page=0`,
      };
      const creds = typeof options.getPortalCredentials === 'function' ? await Promise.resolve(options.getPortalCredentials()).catch(() => ({})) : {};
      if (creds.jobstreet?.email && creds.jobstreet?.password) {
        const portalResult = await runPlaywrightPortalSearch('jobstreet', query, location, creds.jobstreet, onStatusUpdate);
        if (portalResult) return portalResult;
      }
      if (onStatusUpdate) onStatusUpdate({ type: 'tool_call', agent: 'JobStreet', message: 'Log in to JobStreet below to search (you stay in this app).', status: 'done' });
      return { type: 'job_scan', jobs: null, needsPortalLogin: 'jobstreet', portalLinks };
    }
    case 'get_application_status': return runGetApplicationStatus();
    case 'suggest_profile_improvements': return runSuggestProfileImprovements();
    default: return null;
  }
}

/** Build Gemini contents from chat history (messages from frontend: { role, content }[]). */
function buildGeminiContents(messages, currentMessage, toolResultText) {
  const parts = [];
  const history = Array.isArray(messages) ? messages : [];
  const legacyPersonaRegex = /\b(jobai|job search|job listings|application status|resume)\b/i;
  const hasLegacyPersona = history.some((m) => typeof m?.content === 'string' && legacyPersonaRegex.test(m.content));
  const sanitizedHistory = hasLegacyPersona ? [] : history;
  for (const m of sanitizedHistory) {
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

/** Gemini function definitions for MCP-style tool calling. */
const GEMINI_TOOLS = [
  {
    functionDeclarations: [
      {
        name: 'web_search_jobs',
        description: 'Search the web for real job listings. Use this when the user wants to find jobs.',
        parameters: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: 'Job title or keywords (e.g. "Software Engineer", "Data Analyst", "Marketing Manager")',
            },
            location: {
              type: 'string',
              description: 'Location for the job search (e.g. "Singapore", "Remote", "London")',
            },
          },
          required: ['query'],
        },
      },
      {
        name: 'get_application_status',
        description: 'Get the user\'s application status: total applications, in progress, interviews scheduled.',
        parameters: { type: 'object', properties: {} },
      },
      {
        name: 'suggest_profile_improvements',
        description: 'Suggest improvements to the user\'s resume or profile based on their current information.',
        parameters: { type: 'object', properties: {} },
      },
    ],
  },
];

/** Call Gemini API with system prompt and conversation history. Supports function calling.
 * @returns {Promise<null|{ reply: string|null, functionCalls?: Array, errorCode?: 'quota'|'invalid' }>}
 */
async function getGeminiReply(contents, apiKey, userProfileContext = '', tools = null) {
  if (!apiKey || !apiKey.trim()) return null;
  const model = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(apiKey.trim())}`;
  const systemText = JOBAI_SYSTEM_PROMPT + (userProfileContext || '');
  const body = {
    systemInstruction: { parts: [{ text: systemText }] },
    contents: contents.map((c) => ({ role: c.role, parts: c.parts })),
  };
  if (tools !== null) {
    body.tools = tools || GEMINI_TOOLS;
  }
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const errText = await res.text();
  if (!res.ok) {
    console.log('[CHAT] Gemini API error:', res.status, errText?.slice(0, 400));
    if (res.status === 429) {
      let errBody;
      try {
        errBody = JSON.parse(errText);
      } catch {
        errBody = {};
      }
      const status = errBody?.error?.status || errBody?.error?.code;
      const isQuota = res.status === 429 || status === 'RESOURCE_EXHAUSTED' || (typeof status === 'number' && status === 429);
      if (isQuota) {
        return { reply: null, errorCode: 'quota' };
      }
    }
    return { reply: null, errorCode: 'invalid' };
  }
  let data;
  try {
    data = JSON.parse(errText);
  } catch {
    return { reply: null, errorCode: 'invalid' };
  }
  const candidate = data?.candidates?.[0];
  if (!candidate) return { reply: null, errorCode: 'invalid' };
  
  // Check for function calls
  const functionCalls = [];
  const parts = candidate.content?.parts || [];
  for (const part of parts) {
    if (part.functionCall) {
      functionCalls.push({
        name: part.functionCall.name,
        args: part.functionCall.args || {},
      });
    }
  }
  
  // Get text reply
  const textPart = parts.find((p) => p.text);
  const text = textPart?.text;
  
  if (functionCalls.length > 0) {
    return { reply: text?.trim() || null, functionCalls };
  }
  if (typeof text === 'string' && text.trim()) {
    return { reply: text.trim() };
  }
  return { reply: null, errorCode: 'invalid' };
}

/** Call Gemini API with a custom system prompt (no tools). */
async function getGeminiReplyWithSystemPrompt(contents, apiKey, systemText) {
  if (!apiKey || !apiKey.trim()) return null;
  const model = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(apiKey.trim())}`;
  const body = {
    systemInstruction: { parts: [{ text: systemText }] },
    contents: contents.map((c) => ({ role: c.role, parts: c.parts })),
  };
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const errText = await res.text();
  if (!res.ok) return null;
  let data;
  try {
    data = JSON.parse(errText);
  } catch {
    return null;
  }
  const candidate = data?.candidates?.[0];
  if (!candidate) return null;
  const parts = candidate.content?.parts || [];
  const textPart = parts.find((p) => p.text);
  const text = textPart?.text;
  if (typeof text === 'string' && text.trim()) {
    return { reply: text.trim() };
  }
  return null;
}

/** Call DeepSeek API (OpenAI-compatible). contents = array of { role, parts: [{ text }] }; userProfileContext = system prompt suffix.
 * @returns {Promise<null|{ reply: string|null, errorCode?: 'quota'|'invalid' }>}
 */
async function getDeepSeekReply(contents, apiKey, userProfileContext = '') {
  if (!apiKey || !apiKey.trim()) return null;
  const systemText = JOBAI_SYSTEM_PROMPT + (userProfileContext || '');
  const messages = [{ role: 'system', content: systemText }];
  for (const c of contents || []) {
    const role = c.role === 'model' ? 'assistant' : 'user';
    const text = c.parts?.[0]?.text;
    if (typeof text === 'string' && text.trim()) {
      messages.push({ role, content: text.trim() });
    }
  }
  const url = 'https://api.deepseek.com/v1/chat/completions';
  const body = { model: process.env.DEEPSEEK_MODEL || 'deepseek-chat', messages };
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey.trim()}`,
    },
    body: JSON.stringify(body),
  });
  const errText = await res.text();
  if (!res.ok) {
    console.log('[CHAT] DeepSeek API error:', res.status, errText?.slice(0, 400));
    if (res.status === 429) {
      let errBody;
      try {
        errBody = JSON.parse(errText);
      } catch {
        errBody = {};
      }
      const code = errBody?.error?.code || errBody?.error?.type;
      const isQuota = res.status === 429 || code === 'rate_limit_exceeded' || code === 'insufficient_quota';
      if (isQuota) return { reply: null, errorCode: 'quota' };
    }
    return { reply: null, errorCode: 'invalid' };
  }
  let data;
  try {
    data = JSON.parse(errText);
  } catch {
    return { reply: null, errorCode: 'invalid' };
  }
  const text = data?.choices?.[0]?.message?.content;
  if (typeof text === 'string' && text.trim()) {
    return { reply: text.trim() };
  }
  return { reply: null, errorCode: 'invalid' };
}

/** Call DeepSeek API with a custom system prompt. */
async function getDeepSeekReplyWithSystemPrompt(contents, apiKey, systemText) {
  if (!apiKey || !apiKey.trim()) return null;
  const messages = [{ role: 'system', content: systemText }];
  for (const c of contents || []) {
    const role = c.role === 'model' ? 'assistant' : 'user';
    const text = c.parts?.[0]?.text;
    if (typeof text === 'string' && text.trim()) {
      messages.push({ role, content: text.trim() });
    }
  }
  const url = 'https://api.deepseek.com/v1/chat/completions';
  const body = { model: process.env.DEEPSEEK_MODEL || 'deepseek-chat', messages };
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey.trim()}`,
    },
    body: JSON.stringify(body),
  });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) return null;
  const text = json?.choices?.[0]?.message?.content;
  if (typeof text === 'string' && text.trim()) return { reply: text.trim() };
  return null;
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
  const bodyProvider = typeof req.body?.llmProvider === 'string' ? req.body.llmProvider.trim().toLowerCase() : '';
  let provider = bodyProvider === 'deepseek' ? 'deepseek' : 'gemini';
  let apiKey = null;
  if (hasClerk && getAuth(req).userId) {
    const userId = getAuth(req).userId;
    const prefRow = db.prepare('SELECT provider FROM user_llm_preference WHERE user_id = ?').get(userId);
    if (prefRow?.provider === 'deepseek') provider = 'deepseek';
    if (provider === 'deepseek') {
      const row = db.prepare('SELECT encrypted_key FROM user_deepseek_keys WHERE user_id = ?').get(userId);
      if (row?.encrypted_key) apiKey = decryptFromDb(row.encrypted_key);
    }
    if (!apiKey) {
      const row = db.prepare('SELECT encrypted_key FROM user_gemini_keys WHERE user_id = ?').get(userId);
      if (row?.encrypted_key) apiKey = decryptFromDb(row.encrypted_key);
      provider = 'gemini';
    }
  }
  if (!apiKey) {
    const raw = typeof req.body?.apiKey === 'string' ? req.body.apiKey.trim() : '';
    const encoded = typeof req.body?.encodedKey === 'string' ? req.body.encodedKey : null;
    if (bodyProvider === 'deepseek' && raw) {
      provider = 'deepseek';
      apiKey = raw;
    } else if (raw) {
      apiKey = raw;
    } else if (encoded) {
      apiKey = decodeGeminiKey(encoded) || '';
    } else {
      apiKey = provider === 'deepseek'
        ? (process.env.DEEPSEEK_API_KEY || '').trim()
        : (process.env.GEMINI_API_KEY || '').trim();
    }
  }
  console.log('[CHAT] provider:', provider, 'message from frontend: "%s"', message.trim().slice(0, 200));

  // Collect status updates for frontend (like Cursor agent)
  const statusUpdates = [];
  const addStatusUpdate = (update) => {
    statusUpdates.push({
      id: String(statusUpdates.length + 1),
      type: update.type || 'agent_step',
      agent: update.agent,
      message: update.message,
      status: update.status || 'done',
      timestamp: new Date().toISOString(),
    });
  };

  addStatusUpdate({ type: 'planning', agent: 'Agent', message: 'Analyzing request...', status: 'running' });

  let userProfileContext = '';
  if (hasClerk && getAuth(req)?.userId) {
    const resumeRow = db.prepare('SELECT extracted_profile, resume_text FROM user_resume WHERE user_id = ?').get(getAuth(req).userId);
    if (resumeRow && (resumeRow.resume_text || resumeRow.extracted_profile)) {
      userProfileContext = buildUserProfileContext(resumeRow.extracted_profile, resumeRow.resume_text);
    }
  }

  let contents = buildGeminiContents(history, message.trim(), '');
  const agentUpdates = [];
  let toolCall = null;
  let reply = 'No response';
  const maxIterations = 5; // Prevent infinite loops
  let iteration = 0;
  const userId = hasClerk && typeof getAuth === 'function' && getAuth(req)?.userId ? getAuth(req).userId : null;
  const getPortalCredentials = userId ? () => getPortalCredentialsForUser(userId) : () => ({});

  if (!apiKey || !apiKey.trim()) {
    reply = "I couldn't generate a response. Please add your API key in Settings (sidebar → Settings) for the selected provider (Gemini or DeepSeek) and try again.";
    console.log('[CHAT] No API key');
  } else {
    // MCP-style iterative function calling (only for Gemini)
    while (iteration < maxIterations) {
      iteration++;
      addStatusUpdate({ type: 'agent_step', agent: 'Agent', message: `Calling ${provider} API...`, status: 'running' });

      const result =
        provider === 'deepseek'
          ? await getDeepSeekReply(contents, apiKey, userProfileContext)
          : await getGeminiReply(contents, apiKey, userProfileContext, iteration === 1 ? GEMINI_TOOLS : null);

      if (result && result.errorCode) {
        if (result.errorCode === 'quota') {
          reply =
            provider === 'deepseek'
              ? "I couldn't generate a response. You've hit the DeepSeek API rate limit or quota. Try again in a minute or check your plan."
              : "I couldn't generate a response. You've hit the Gemini API rate limit or quota. Check your plan and billing at https://ai.google.dev/gemini-api/docs/rate-limits or try again in a minute.";
        } else {
          reply = "I couldn't generate a response. Your API key may be invalid or expired—check Settings and try again.";
        }
        break;
      }

      // Handle function calls (MCP-style)
      if (result && result.functionCalls && result.functionCalls.length > 0 && provider === 'gemini') {
        for (const fnCall of result.functionCalls) {
          const toolName = fnCall.name;
          const params = fnCall.args || {};
          toolCall = { name: toolName, params };
          addStatusUpdate({ type: 'tool_call', agent: 'Tool', message: `Calling ${toolName}...`, status: 'running' });
          recordBackendActivity(req, { source: 'chat', type: 'tool_call', message: `Running tool: ${toolName}` });

          const onToolStatus = (update) => {
            addStatusUpdate(update);
          };

          const toolResult = await runTool(toolName, params, onToolStatus, { getPortalCredentials });
          if (toolResult) {
            agentUpdates.push(toolResult);
            let toolResultText = '';
            if (toolResult.type === 'job_scan') {
              if (toolResult.needsPortalLogin === 'jobstreet') {
                toolResultText = 'The user must log in to JobStreet in the card shown in the app (they stay in the app, no redirect). Ask them to enter their JobStreet email and password in the card, then click "Log in to JobStreet". After they log in, they can click "Search jobs now" to run the search.';
              } else if (toolResult.jobs && toolResult.jobs.length > 0) {
                toolResultText = 'Job search results:\n' + toolResult.jobs.map((j) => `- ${j.title} at ${j.company} (${j.location}) – match ${j.match}`).join('\n');
              } else {
                toolResultText = 'No job listings found in the search results. Please try a different search query or location.';
              }
            } else if (toolResult.type === 'application_status') {
              toolResultText = `Application status: ${toolResult.total} total, ${toolResult.inProgress} in progress, ${toolResult.interviews} interviews. Recent: ${(toolResult.recent || []).join('; ')}`;
            } else if (toolResult.type === 'profile_improvements' && toolResult.suggestions) {
              toolResultText = 'Profile improvement suggestions:\n' + toolResult.suggestions.map((s, i) => `${i + 1}. ${s}`).join('\n');
            }

            // Add tool result to conversation for next LLM call
            contents.push({
              role: 'model',
              parts: [{ text: `[Function call: ${toolName}]` }],
            });
            contents.push({
              role: 'user',
              parts: [{ text: `Tool result: ${toolResultText}` }],
            });
          }
        }
        // Continue loop to get final reply after tool execution
        continue;
      }

      // Final reply received
      if (result && typeof result.reply === 'string' && result.reply.trim()) {
        reply = stripMarkdown(result.reply);
        addStatusUpdate({ type: 'done', agent: 'Agent', message: 'Reply generated', status: 'done' });
        break;
      }
    }

    // Fallback: if no function calling, use keyword-based tool detection (for DeepSeek or when function calling fails)
    if (reply === 'No response' && provider === 'deepseek') {
      const toolName = getToolToRun(message);
      if (toolName) {
        toolCall = { name: toolName, params: inferToolParams(message, toolName) };
        addStatusUpdate({ type: 'tool_call', agent: 'Tool', message: `Calling ${toolName}...`, status: 'running' });
        const update = await runTool(toolName, toolCall.params, (s) => addStatusUpdate(s), { getPortalCredentials });
        if (update) {
          agentUpdates.push(update);
          const toolResultText = update.type === 'job_scan'
            ? (update.needsPortalLogin === 'jobstreet'
                ? 'User must log in to JobStreet in the card (they stay in the app). Ask them to log in in the card, then click Search jobs now.'
                : update.jobs && update.jobs.length > 0
                  ? 'Job scan results:\n' + update.jobs.map((j) => `- ${j.title} at ${j.company} (${j.location}) – match ${j.match}`).join('\n')
                  : 'No job listings found in the search results.')
            : '';
          contents = buildGeminiContents(history, message.trim(), toolResultText);
          const result = await getDeepSeekReply(contents, apiKey, userProfileContext);
          if (result && typeof result.reply === 'string') {
            reply = stripMarkdown(result.reply);
          }
        }
      } else {
        const result = await getDeepSeekReply(contents, apiKey, userProfileContext);
        if (result && typeof result.reply === 'string') {
          reply = stripMarkdown(result.reply);
        }
      }
    }
  }

  if (reply && reply !== 'No response') {
    addStatusUpdate({ type: 'agent_step', agent: 'Format Guard', message: 'Normalizing question format...', status: 'running' });
    const guardContents = [{ role: 'user', parts: [{ text: reply }] }];
    const formatted =
      provider === 'deepseek'
        ? await getDeepSeekReplyWithSystemPrompt(guardContents, apiKey, QUESTION_FORMAT_GUARD_PROMPT)
        : await getGeminiReplyWithSystemPrompt(guardContents, apiKey, QUESTION_FORMAT_GUARD_PROMPT);
    if (formatted && typeof formatted.reply === 'string' && formatted.reply.trim()) {
      reply = stripMarkdown(formatted.reply);
      addStatusUpdate({ type: 'done', agent: 'Format Guard', message: 'Questions formatted', status: 'done' });
    } else {
      addStatusUpdate({ type: 'done', agent: 'Format Guard', message: 'Format guard skipped', status: 'done' });
    }
  }

  recordBackendActivity(req, { source: 'chat', type: 'done', message: 'Reply sent to frontend' });
  console.log('[CHAT] >>> Sending reply + %d agentUpdate(s) + %d statusUpdate(s)\n', agentUpdates.length, statusUpdates.length);
  return res.json({
    reply,
    agentUpdates: agentUpdates.length ? agentUpdates : undefined,
    toolCall: toolCall || undefined,
    statusUpdates: statusUpdates.length > 0 ? statusUpdates : undefined, // Cursor-like status stream
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
