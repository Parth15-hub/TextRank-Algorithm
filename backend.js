import express from "express";
import multer from "multer";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

// ─── pdf-parse & mammoth are CommonJS — use createRequire to import them ───
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const pdfParse = require("pdf-parse");
const mammoth  = require("mammoth");

const app = express();
app.use(express.json({ limit: "1mb" }));
app.use(express.static("public"));

// ─────────────────────────────────────────────────────────────────────────────
// FILE UPLOAD — multer config
// Files are stored temporarily in /uploads, deleted after text extraction
// ─────────────────────────────────────────────────────────────────────────────
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const UPLOAD_DIR = path.join(__dirname, "uploads");
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename:    (req, file, cb) => cb(null, `upload_${Date.now()}_${file.originalname}`)
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB max
  fileFilter: (req, file, cb) => {
    const allowed = [".txt", ".pdf", ".docx"];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowed.includes(ext)) cb(null, true);
    else cb(new Error(`Unsupported file type: ${ext}. Use .txt, .pdf, or .docx`));
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// SHARED STOPWORDS (reused by all models — existing preprocessing preserved)
// ─────────────────────────────────────────────────────────────────────────────
const STOPWORDS = new Set([
  "a","an","and","are","as","at","be","been","but","by","can","could","did","do","does","doing",
  "for","from","had","has","have","having","he","her","hers","him","his","how","i","if","in","into",
  "is","it","its","just","may","might","more","most","must","my","no","not","of","on","or","our",
  "ours","she","should","so","some","such","than","that","the","their","theirs","them","then",
  "there","these","they","this","those","to","too","was","we","were","what","when","where","which",
  "who","whom","why","will","with","you","your","yours"
]);

// ─────────────────────────────────────────────────────────────────────────────
// SHARED PREPROCESSING PIPELINE (unchanged from original)
// ─────────────────────────────────────────────────────────────────────────────
function splitSentences(text) {
  return text
    .replace(/\s+/g, " ")
    .trim()
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(Boolean);
}

function tokenize(sentence) {
  return sentence
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .filter(tok => tok.length > 1)
    .filter(tok => !STOPWORDS.has(tok));
}

function buildTfidfVectors(sentTokens) {
  const N = sentTokens.length;
  const df = new Map();

  for (const tokens of sentTokens) {
    const uniq = new Set(tokens);
    for (const t of uniq) df.set(t, (df.get(t) || 0) + 1);
  }

  const idf = new Map();
  for (const [t, d] of df.entries()) {
    idf.set(t, Math.log((N + 1) / (d + 1)) + 1);
  }

  return sentTokens.map(tokens => {
    const tf = new Map();
    for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);

    const vec = new Map();
    const len = tokens.length || 1;
    for (const [t, c] of tf.entries()) {
      vec.set(t, (c / len) * (idf.get(t) || 0));
    }
    return vec;
  });
}

function cosineSim(vecA, vecB) {
  let dot = 0;
  const [small, large] = vecA.size < vecB.size ? [vecA, vecB] : [vecB, vecA];

  for (const [t, wa] of small.entries()) {
    const wb = large.get(t);
    if (wb) dot += wa * wb;
  }

  let normA = 0;
  for (const wa of vecA.values()) normA += wa * wa;
  normA = Math.sqrt(normA);

  let normB = 0;
  for (const wb of vecB.values()) normB += wb * wb;
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) return 0;
  return dot / (normA * normB);
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGERANK / TEXTRANK CORE (unchanged from original)
// ─────────────────────────────────────────────────────────────────────────────
function textRank(scoresMatrix, d = 0.85, maxIter = 50, tol = 1e-6) {
  const n = scoresMatrix.length;
  const ranks = new Array(n).fill(1 / n);
  const outSum = new Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += scoresMatrix[i][j];
    outSum[i] = s;
  }

  for (let iter = 0; iter < maxIter; iter++) {
    const newRanks = new Array(n).fill((1 - d) / n);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const wji = scoresMatrix[j][i];
        if (wji <= 0) continue;
        const denom = outSum[j] || 1;
        newRanks[i] += d * (wji / denom) * ranks[j];
      }
    }

    let diff = 0;
    for (let i = 0; i < n; i++) diff += Math.abs(newRanks[i] - ranks[i]);
    for (let i = 0; i < n; i++) ranks[i] = newRanks[i];

    if (diff < tol) break;
  }

  return ranks;
}

function buildSimMatrix(vectors, threshold = 0.1) {
  const n = vectors.length;
  const sim = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const s = cosineSim(vectors[i], vectors[j]);
      const w = s >= threshold ? s : 0;
      sim[i][j] = w;
      sim[j][i] = w;
    }
  }
  return sim;
}

function pickTopK(sentences, scores, k) {
  const idx = [...Array(sentences.length).keys()];
  idx.sort((a, b) => scores[b] - scores[a]);
  const selected = new Set(idx.slice(0, k));
  return sentences.filter((_, i) => selected.has(i)).join(" ");
}

// ─────────────────────────────────────────────────────────────────────────────
// MODEL 1 — TextRank (unchanged logic)
// ─────────────────────────────────────────────────────────────────────────────
function summarizeTextRank(text, numSentences) {
  const sentences = splitSentences(text);
  if (sentences.length === 0) return { summary: "", graphData: null };

  const k = Math.max(1, Math.min(numSentences, sentences.length));
  const sentTokens = sentences.map(tokenize);
  const vectors = buildTfidfVectors(sentTokens);
  const sim = buildSimMatrix(vectors, 0.1);
  const ranks = textRank(sim);
  const summary = pickTopK(sentences, ranks, k);
  const graphData = buildGraphData(sentences, sim, ranks, k);

  return { summary, ranks, graphData };
}

// ─────────────────────────────────────────────────────────────────────────────
// MODEL 2 — TF-IDF Baseline 
// ─────────────────────────────────────────────────────────────────────────────
function summarizeTfIdf(text, numSentences) {
  const sentences = splitSentences(text);
  if (sentences.length === 0) return { summary: "", graphData: null };

  const k = Math.max(1, Math.min(numSentences, sentences.length));
  const sentTokens = sentences.map(tokenize);
  const vectors = buildTfidfVectors(sentTokens);

  const scores = vectors.map(vec => {
    let total = 0;
    for (const w of vec.values()) total += w;
    return total;
  });

  const sim = buildSimMatrix(vectors, 0.1);
  const graphData = buildGraphData(sentences, sim, scores, k);
  const summary = pickTopK(sentences, scores, k);

  return { summary, scores, graphData };
}

// ─────────────────────────────────────────────────────────────────────────────
// MODEL 3 — LexRank
// ─────────────────────────────────────────────────────────────────────────────
function summarizeLexRank(text, numSentences, threshold = 0.1) {
  const sentences = splitSentences(text);
  if (sentences.length === 0) return { summary: "", graphData: null };

  const k = Math.max(1, Math.min(numSentences, sentences.length));
  const sentTokens = sentences.map(tokenize);
  const vectors = buildTfidfVectors(sentTokens);
  const n = sentences.length;

  const cosMatrix = buildSimMatrix(vectors, threshold);

  const degree = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (cosMatrix[i][j] > 0) degree[i]++;
    }
    if (degree[i] === 0) degree[i] = 1;
  }

  const transMatrix = Array.from({ length: n }, (_, i) =>
    cosMatrix[i].map(v => v / degree[i])
  );

  const ranks = textRank(transMatrix, 0.85, 100, 1e-6);
  const graphData = buildGraphData(sentences, cosMatrix, ranks, k);
  const summary = pickTopK(sentences, ranks, k);

  return { summary, ranks, graphData };
}

// ─────────────────────────────────────────────────────────────────────────────
// GRAPH DATA BUILDER
// ─────────────────────────────────────────────────────────────────────────────
function buildGraphData(sentences, simMatrix, scores, topK) {
  const n = sentences.length;
  const maxScore = Math.max(...scores, 1e-9);

  const idx = [...Array(n).keys()].sort((a, b) => scores[b] - scores[a]);
  const selectedSet = new Set(idx.slice(0, topK));

  const nodes = sentences.map((sent, i) => ({
    id: i,
    label: `S${i + 1}`,
    score: scores[i],
    normScore: scores[i] / maxScore,
    selected: selectedSet.has(i),
    preview: sent.length > 60 ? sent.slice(0, 60) + "…" : sent,
  }));

  const edges = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const w = simMatrix[i][j];
      if (w > 0.1) edges.push({ source: i, target: j, weight: w });
    }
  }

  return { nodes, edges };
}

// ─────────────────────────────────────────────────────────────────────────────
// ROUGE EVALUATION MODULE
// ─────────────────────────────────────────────────────────────────────────────
function getNgrams(tokens, n) {
  const ngrams = new Map();
  for (let i = 0; i <= tokens.length - n; i++) {
    const gram = tokens.slice(i, i + n).join(" ");
    ngrams.set(gram, (ngrams.get(gram) || 0) + 1);
  }
  return ngrams;
}

function rougeN(candidateTokens, referenceTokens, n) {
  if (referenceTokens.length < n || candidateTokens.length < n) return { p: 0, r: 0, f: 0 };

  const refGrams  = getNgrams(referenceTokens, n);
  const candGrams = getNgrams(candidateTokens, n);

  let overlapCount = 0;
  for (const [gram, count] of candGrams.entries()) {
    overlapCount += Math.min(count, refGrams.get(gram) || 0);
  }

  const refTotal  = [...refGrams.values()].reduce((a, b) => a + b, 0);
  const candTotal = [...candGrams.values()].reduce((a, b) => a + b, 0);

  const recall    = refTotal   > 0 ? overlapCount / refTotal   : 0;
  const precision = candTotal  > 0 ? overlapCount / candTotal  : 0;
  const f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;

  return {
    p: parseFloat(precision.toFixed(4)),
    r: parseFloat(recall.toFixed(4)),
    f: parseFloat(f1.toFixed(4))
  };
}

function lcs(a, b) {
  const m = a.length, n = b.length;
  let prev = new Array(n + 1).fill(0);
  let curr = new Array(n + 1).fill(0);
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      curr[j] = a[i - 1] === b[j - 1] ? prev[j - 1] + 1 : Math.max(curr[j - 1], prev[j]);
    }
    [prev, curr] = [curr, prev];
    curr.fill(0);
  }
  return prev[n];
}

function rougeL(candidateTokens, referenceTokens) {
  const lcsLen = lcs(candidateTokens, referenceTokens);
  const recall    = referenceTokens.length > 0 ? lcsLen / referenceTokens.length : 0;
  const precision = candidateTokens.length  > 0 ? lcsLen / candidateTokens.length  : 0;
  const f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;

  return {
    p: parseFloat(precision.toFixed(4)),
    r: parseFloat(recall.toFixed(4)),
    f: parseFloat(f1.toFixed(4))
  };
}

function computeRouge(candidateSummary, referenceSummary) {
  const tokenizeRouge = str =>
    str.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);

  const cand = tokenizeRouge(candidateSummary);
  const ref  = tokenizeRouge(referenceSummary);

  return {
    "rouge1": rougeN(cand, ref, 1),
    "rouge2": rougeN(cand, ref, 2),
    "rougeL": rougeL(cand, ref)
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// ACCURACY SCORE — derived from ROUGE F1 scores
// Formula: ((ROUGE-1 F1 + ROUGE-2 F1 + ROUGE-L F1) / 3) × 100
// Returns null when no rouge scores are available
// ─────────────────────────────────────────────────────────────────────────────
function computeAccuracy(rouge) {
  if (!rouge || typeof rouge !== "object") return null;

  const r1 = rouge["rouge1"]?.f ?? null;
  const r2 = rouge["rouge2"]?.f ?? null;
  const rL = rouge["rougeL"]?.f ?? null;

  if (r1 === null || r2 === null || rL === null) return null;

  const avg = (r1 + r2 + rL) / 3;
  return parseFloat((avg * 100).toFixed(2));
}

// ─────────────────────────────────────────────────────────────────────────────
// TEXT EXTRACTION HELPERS
// ─────────────────────────────────────────────────────────────────────────────

/** Extract plain text from a .txt file */
async function extractTxt(filePath) {
  return fs.readFileSync(filePath, "utf-8");
}

/** Extract plain text from a .pdf file using pdf-parse */
async function extractPdf(filePath) {
  const dataBuffer = fs.readFileSync(filePath);
  const result = await pdfParse(dataBuffer);
  return result.text;
}

/** Extract plain text from a .docx file using mammoth */
async function extractDocx(filePath) {
  const result = await mammoth.extractRawText({ path: filePath });
  return result.value;
}

/** Route to the right extractor based on file extension */
async function extractTextFromFile(filePath, originalName) {
  const ext = path.extname(originalName).toLowerCase();
  switch (ext) {
    case ".txt":  return extractTxt(filePath);
    case ".pdf":  return extractPdf(filePath);
    case ".docx": return extractDocx(filePath);
    default: throw new Error(`Unsupported file type: ${ext}`);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// API ROUTES
// ─────────────────────────────────────────────────────────────────────────────

// ── File upload → text extraction endpoint ──────────────────────────────
// POST /api/upload   (multipart/form-data, field name: "article")
// Returns: { text: "extracted plain text..." }
app.post("/api/upload", upload.single("article"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded. Send a .txt, .pdf, or .docx file." });
  }

  const filePath    = req.file.path;
  const originalName = req.file.originalname;

  try {
    const extracted = await extractTextFromFile(filePath, originalName);

    // Clean up the temporary file immediately after extraction
    fs.unlink(filePath, () => {});

    const cleaned = extracted
      .replace(/\r\n/g, "\n")          // normalise line endings
      .replace(/\n{3,}/g, "\n\n")      // collapse excessive blank lines
      .replace(/[ \t]{2,}/g, " ")      // collapse multiple spaces
      .trim();

    if (!cleaned || cleaned.length < 30) {
      return res.status(422).json({ error: "Could not extract readable text from this file. It may be scanned/image-based." });
    }

    res.json({
      text: cleaned,
      filename: originalName,
      charCount: cleaned.length,
      wordCount: cleaned.split(/\s+/).filter(Boolean).length
    });

  } catch (err) {
    // Clean up on failure too
    fs.unlink(filePath, () => {});
    console.error("File extraction error:", err);
    res.status(500).json({ error: `Text extraction failed: ${err.message}` });
  }
});

// ── Main summarization endpoint ───────────────────────
app.post("/api/summarize", (req, res) => {
  try {
    const { text, numSentences, model = "textrank", referenceSummary } = req.body || {};

    if (!text || String(text).trim().length < 50) {
      return res.status(400).json({ error: "Please enter a longer article." });
    }

    const n = Math.max(1, Math.min(5, Number(numSentences || 3)));
    const inputText = String(text);
    let result;

    switch (model.toLowerCase()) {
      case "tfidf":
        result = summarizeTfIdf(inputText, n);
        break;
      case "lexrank":
        result = summarizeLexRank(inputText, n);
        break;
      case "textrank":
      default:
        result = summarizeTextRank(inputText, n);
        break;
    }

    if (!result.summary) {
      return res.status(422).json({ error: "Could not generate summary. Try a longer article." });
    }

    let rouge = null;
    if (referenceSummary && String(referenceSummary).trim().length > 10) {
      rouge = computeRouge(result.summary, String(referenceSummary).trim());
    }

    const accuracy = computeAccuracy(rouge);

    res.json({
      summary: result.summary,
      method: model.toUpperCase(),
      graphData: result.graphData,
      rouge,
      accuracy
    });
  } catch (e) {
    console.error("Server error:", e);
    res.status(500).json({ error: "Server error", details: String(e) });
  }
});

// ── Existing: ROUGE standalone endpoint ─────────────────────────
app.post("/api/rouge", (req, res) => {
  try {
    const { candidate, reference } = req.body || {};
    if (!candidate || !reference) {
      return res.status(400).json({ error: "Both 'candidate' and 'reference' fields are required." });
    }
    const scores = computeRouge(String(candidate), String(reference));
    res.json({ rouge: scores });
  } catch (e) {
    res.status(500).json({ error: "Server error", details: String(e) });
  }
});

// ── BERT summarization endpoint (HuggingFace Inference API) ─────────────
// POST /api/bert
// Body: { text, referenceSummary? }
// Returns: { summary, graphData, rouge, accuracy }
//
// Set HUGGINGFACE_API_KEY in your environment before starting the server.
// e.g.  export HUGGINGFACE_API_KEY="hf_xxxxxxxxxxxxxxxxxxxx"
app.post("/api/bert", async (req, res) => {
  try {
    const { text, referenceSummary } = req.body || {};

    if (!text || String(text).trim().length < 50) {
      return res.status(400).json({ error: "Please provide a longer article (min 50 chars)." });
    }

    const inputText = String(text).trim();
    const apiKey    = process.env.HUGGINGFACE_API_KEY;

    if (!apiKey) {
      return res.status(500).json({
        error: "HUGGINGFACE_API_KEY environment variable is not set on the server."
      });
    }

    // ── Call HuggingFace Inference API ───────────────────────────────────────
    const HF_MODEL_URL =
      "https://api-inference.huggingface.co/models/facebook/bart-large-cnn";

    let hfSummary;
    try {
      const hfResponse = await fetch(HF_MODEL_URL, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${apiKey}`,
          "Content-Type":  "application/json"
        },
        body: JSON.stringify({
          inputs: inputText.slice(0, 1024), // BART has a 1024-token context limit
          parameters: {
            max_length:   150,
            min_length:   30,
            do_sample:    false
          }
        })
      });

      if (!hfResponse.ok) {
        const errBody = await hfResponse.text();
        throw new Error(`HuggingFace API error ${hfResponse.status}: ${errBody}`);
      }

      const hfData = await hfResponse.json();

      // HF returns [{ summary_text: "..." }]
      if (!Array.isArray(hfData) || !hfData[0]?.summary_text) {
        throw new Error("Unexpected HuggingFace API response format.");
      }

      hfSummary = hfData[0].summary_text.trim();
    } catch (fetchErr) {
      console.error("HuggingFace API call failed:", fetchErr.message);
      return res.status(502).json({
        error: "Failed to reach HuggingFace Inference API.",
        details: fetchErr.message
      });
    }

    // ── Build graphData using the same pipeline as other models ─────────────
    // Align the BERT summary sentences back to the original sentence graph so
    // the frontend visualisation remains compatible with the existing schema.
    const sentences    = splitSentences(inputText);
    const sentTokens   = sentences.map(tokenize);
    const vectors      = buildTfidfVectors(sentTokens);
    const sim          = buildSimMatrix(vectors, 0.1);

    // Score each sentence by cosine similarity to the BERT summary
    const summaryTokens  = tokenize(hfSummary);
    const summaryVec     = buildTfidfVectors([summaryTokens])[0];
    const bertScores     = vectors.map(v => cosineSim(v, summaryVec));

    const k          = Math.max(1, Math.min(3, sentences.length));
    const graphData  = buildGraphData(sentences, sim, bertScores, k);

    // ── ROUGE & accuracy ─────────────────────────────────────────────────────
    let rouge    = null;
    let accuracy = null;

    if (referenceSummary && String(referenceSummary).trim().length > 10) {
      rouge    = computeRouge(hfSummary, String(referenceSummary).trim());
      accuracy = computeAccuracy(rouge);
    }

    res.json({
      summary:  hfSummary,
      method:   "BERT",
      graphData,
      rouge,
      accuracy
    });

  } catch (e) {
    console.error("BERT endpoint error:", e);
    res.status(500).json({ error: "Server error", details: String(e) });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ Running at http://localhost:${PORT}`);
});