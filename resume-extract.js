/**
 * Resume extraction: MCP-style design.
 * 1. PDF text extraction (equivalent to MCP PDF tool, e.g. mcpmarket.com/server/pdf or
 *    @mcp-apps/pdf-tools-mcp-server extractText). Step 1 can be swapped for an MCP client
 *    that spawns/calls a PDF MCP server when you run one.
 * 2. Agent (Gemini) analyzes the text and returns structured ExtractedProfile.
 */

const path = require('path');
const fs = require('fs');

/** Extract text from a PDF file (MCP PDF tool equivalent). Returns plain text or null. */
async function extractTextFromPdf(filePath) {
  if (!filePath || !fs.existsSync(filePath)) return null;
  const ext = path.extname(filePath).toLowerCase();
  if (ext !== '.pdf') return null;
  try {
    const pdfParse = require('pdf-parse');
    const dataBuffer = fs.readFileSync(filePath);
    const data = await pdfParse(dataBuffer);
    return typeof data?.text === 'string' ? data.text.trim() : null;
  } catch (err) {
    console.log('[RESUME] pdf-parse error:', err?.message);
    return null;
  }
}

/** Prompt for Gemini: structured profile + greeting summary (master agent summarizes for the user). */
const RESUME_ANALYSIS_PROMPT = `You are a resume analyst. Extract key information from the following resume text and return a single JSON object only (no markdown, no code fence). Use this exact structure; infer or use placeholder when missing.

Also add a "greeting" field: one short message (2-3 sentences) to greet the user by name and summarize what you found. Example: "Hi Sarah, I've read your resume—it seems like you are a Senior Software Engineer with 5+ years experience, based in Singapore. I've extracted the key info below; you can use the chat to personalize your job matches."

{
  "profile": {
    "name": "full name or Unknown",
    "email": "email or empty string",
    "phone": "phone or empty string",
    "title": "current/latest job title",
    "experience": "e.g. 5+ years",
    "location": "city/country",
    "skills": ["skill1", "skill2", ...]
  },
  "preferences": {
    "jobTitles": ["desired role 1", ...],
    "industries": ["industry 1", ...],
    "locations": ["location 1", ...],
    "salaryRange": "e.g. S$8k – S$15k or Unknown",
    "workType": "Full-time or Part-time or Contract or Remote"
  },
  "stats": {
    "jobsApplied": 0,
    "activeAgents": 0,
    "nextInterview": "e.g. None or date",
    "pendingAssessments": 0
  },
  "greeting": "Hi [first name], I've read your resume—it seems like you are [one sentence summary]. I've extracted the key info below; reply in the chat to personalize your job matches."
}

Resume text:
`;

/** Call Gemini to analyze resume text and return structured ExtractedProfile. */
async function analyzeResumeWithGemini(resumeText, apiKey) {
  if (!apiKey || !apiKey.trim() || !resumeText || !resumeText.trim()) return null;
  const model = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(apiKey.trim())}`;
  const prompt = RESUME_ANALYSIS_PROMPT + resumeText.slice(0, 28000);
  const body = {
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    generationConfig: { responseMimeType: 'application/json' },
  };
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      console.log('[RESUME] Gemini analysis error:', res.status);
      return null;
    }
    const data = await res.json();
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!text || typeof text !== 'string') return null;
    const raw = text.trim().replace(/^```json?\s*|\s*```$/g, '');
    const parsed = JSON.parse(raw);
    const profile = normalizeExtractedProfile(parsed);
    const greeting = typeof parsed.greeting === 'string' && parsed.greeting.trim() ? parsed.greeting.trim() : null;
    return { profile, greeting };
  } catch (err) {
    console.log('[RESUME] Gemini analysis failed:', err?.message);
    return null;
  }
}

function normalizeExtractedProfile(obj) {
  if (!obj || typeof obj !== 'object') return null;
  const p = obj.profile && typeof obj.profile === 'object' ? obj.profile : {};
  const pref = obj.preferences && typeof obj.preferences === 'object' ? obj.preferences : {};
  const s = obj.stats && typeof obj.stats === 'object' ? obj.stats : {};
  return {
    profile: {
      name: typeof p.name === 'string' ? p.name : 'Unknown',
      email: typeof p.email === 'string' ? p.email : '',
      phone: typeof p.phone === 'string' ? p.phone : '',
      title: typeof p.title === 'string' ? p.title : '',
      experience: typeof p.experience === 'string' ? p.experience : '',
      location: typeof p.location === 'string' ? p.location : '',
      skills: Array.isArray(p.skills) ? p.skills.filter((x) => typeof x === 'string') : [],
    },
    preferences: {
      jobTitles: Array.isArray(pref.jobTitles) ? pref.jobTitles.filter((x) => typeof x === 'string') : [],
      industries: Array.isArray(pref.industries) ? pref.industries.filter((x) => typeof x === 'string') : [],
      locations: Array.isArray(pref.locations) ? pref.locations.filter((x) => typeof x === 'string') : [],
      salaryRange: typeof pref.salaryRange === 'string' ? pref.salaryRange : 'Unknown',
      workType: typeof pref.workType === 'string' ? pref.workType : 'Full-time',
    },
    stats: {
      jobsApplied: typeof s.jobsApplied === 'number' ? s.jobsApplied : 0,
      activeAgents: typeof s.activeAgents === 'number' ? s.activeAgents : 0,
      nextInterview: typeof s.nextInterview === 'string' ? s.nextInterview : 'None',
      pendingAssessments: typeof s.pendingAssessments === 'number' ? s.pendingAssessments : 0,
    },
  };
}

/**
 * MCP-style pipeline: extract text (PDF tool) then agent analysis.
 * @param {string} absoluteFilePath - Full path to the saved resume file
 * @param {string} apiKey - Gemini API key
 * @returns {Promise<{ profile: object, greeting: string|null }|null>} Profile + greeting from master agent, or null
 */
async function extractResumeProfile(absoluteFilePath, apiKey) {
  const text = await extractTextFromPdf(absoluteFilePath);
  if (!text || text.length < 10) {
    console.log('[RESUME] No PDF text extracted (or not PDF)');
    return null;
  }
  console.log('[RESUME] Extracted', text.length, 'chars from PDF');
  return analyzeResumeWithGemini(text, apiKey);
}

module.exports = { extractTextFromPdf, analyzeResumeWithGemini, extractResumeProfile, normalizeExtractedProfile };
