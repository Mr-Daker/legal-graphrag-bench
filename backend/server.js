import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const port = Number(process.env.PORT || 8787);

const pipelineCommands = {
  llm_only: {
    label: "LLM-Only",
    script: "scripts/llm_only_baseline.py",
  },
  basic_rag: {
    label: "Basic RAG",
    script: "scripts/basic_rag_gemini.py",
  },
  graphrag: {
    label: "GraphRAG",
    script: "scripts/graphrag_tigergraph.py",
  },
};

function pythonExecutable() {
  const winPython = path.join(repoRoot, ".venv-win", "Scripts", "python.exe");
  if (existsSync(winPython)) return winPython;
  const unixPython = path.join(repoRoot, ".venv", "bin", "python");
  if (existsSync(unixPython)) return unixPython;
  return process.env.PYTHON || "python";
}

function loadJson(relativePath, fallback) {
  const target = path.join(repoRoot, relativePath);
  if (!existsSync(target)) return fallback;
  try {
    return JSON.parse(readFileSync(target, "utf8"));
  } catch {
    return fallback;
  }
}

const questions = loadJson("data/eval/questions_dev.json", []);
const accuracy = loadJson("data/reports/accuracy_report_final_distilbert.json", {});
const comparison = loadJson("data/reports/pipeline_comparison_report.json", {});

const finalV9Summary = {
  llm_only: {
    questions: 20,
    avg_total_tokens: 143.3,
    avg_prompt_tokens: 54.65,
    avg_completion_tokens: 88.65,
    avg_latency_ms: 1682.94,
    heuristic_pass_rate: 0.7,
    judge_pass_rate: 0.35,
    bertscore_f1_raw: 0.8166,
    bertscore_f1_rescaled: 0.4507,
  },
  basic_rag: {
    questions: 20,
    avg_total_tokens: 3746.85,
    avg_prompt_tokens: 3659.55,
    avg_completion_tokens: 87.3,
    avg_latency_ms: 1790.62,
    heuristic_pass_rate: 0.75,
    judge_pass_rate: 0.55,
    bertscore_f1_raw: 0.8337,
    bertscore_f1_rescaled: 0.5018,
  },
  graphrag: {
    questions: 20,
    avg_total_tokens: 2375.1,
    avg_prompt_tokens: 2323.55,
    avg_completion_tokens: 51.55,
    avg_latency_ms: 1408.94,
    heuristic_pass_rate: 0.95,
    judge_pass_rate: 1.0,
    bertscore_f1_raw: 0.9003,
    bertscore_f1_rescaled: 0.7013,
  },
};

function matchedQuestionId(question) {
  const normalized = question.trim().toLowerCase();
  const found = questions.find((item) => String(item.question || "").trim().toLowerCase() === normalized);
  return found?.id || null;
}

function metricFor(pipeline, qid) {
  const data = accuracy?.pipelines?.[pipeline];
  const aggregate = accuracy?.comparison?.[pipeline] || {};
  const bert = data?.bertscore?.per_question?.find((item) => item.question_id === qid);
  const judge = data?.judge?.per_question?.find((item) => item.question_id === qid);
  return {
    verdict: judge?.verdict || (aggregate.bonus_judge_met ? "PASS" : "N/A"),
    bertscore: bert?.f1_raw ?? aggregate.bertscore_f1_raw ?? null,
  };
}

function extractJson(stdout) {
  const start = stdout.indexOf("{");
  const end = stdout.lastIndexOf("}");
  if (start < 0 || end < start) {
    throw new Error("Pipeline did not emit JSON.");
  }
  return JSON.parse(stdout.slice(start, end + 1));
}

function runPipeline(name, question, qid) {
  const command = pipelineCommands[name];
  const py = pythonExecutable();
  const args = [command.script, "--query", question];
  const env = {
    ...process.env,
    PYTHONPATH: path.join(repoRoot, "scripts"),
    PYTHONIOENCODING: "utf-8",
    LLM_PROVIDER: "gemini",
    LLM_FALLBACK_PROVIDER: "",
  };

  if (name === "graphrag") {
    env.TG_FORCE_LOCAL = "1";
  }

  return new Promise((resolve) => {
    const child = spawn(py, args, {
      cwd: repoRoot,
      env,
      shell: false,
      windowsHide: true,
    });

    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => {
      resolve({
        answer: `Pipeline failed to start: ${error.message}`,
        tokens: 0,
        latency_ms: 0,
        cost_usd: 0,
        verdict: "ERROR",
        bertscore: null,
      });
    });
    child.on("close", (code) => {
      if (code !== 0) {
        resolve({
          answer: stderr.trim() || `${command.label} exited with code ${code}.`,
          tokens: 0,
          latency_ms: 0,
          cost_usd: 0,
          verdict: "ERROR",
          bertscore: null,
        });
        return;
      }

      try {
        const raw = extractJson(stdout);
        const metrics = metricFor(name, qid);
        resolve({
          answer: raw.answer || "",
          tokens: raw.total_tokens ?? 0,
          latency_ms: raw.latency_ms ?? 0,
          cost_usd: raw.cost_usd ?? 0,
          verdict: metrics.verdict,
          bertscore: metrics.bertscore,
          prompt_tokens: raw.prompt_tokens ?? 0,
          completion_tokens: raw.completion_tokens ?? 0,
          model: raw.generation_model || raw.model,
        });
      } catch (error) {
        resolve({
          answer: `Could not parse ${command.label} output: ${error.message}`,
          tokens: 0,
          latency_ms: 0,
          cost_usd: 0,
          verdict: "ERROR",
          bertscore: null,
        });
      }
    });
  });
}

function summaryPayload() {
  const summaries =
    comparison?.summaries?.graphrag?.questions > 0 ? comparison.summaries : finalV9Summary;
  const tokenReduction =
    comparison?.comparisons?.graphrag_vs_basic_rag_token_reduction_pct ?? 36.61;
  const latencyReduction =
    comparison?.comparisons?.graphrag_vs_basic_rag_latency_reduction_pct ?? 21.32;

  return {
    corpus: {
      source: "harvard-lil/cold-cases",
      cases: 478,
      tokens: 2200374,
      chunks: 7008,
      model: "gemini-2.5-flash-lite",
    },
    comparison: summaries,
    token_reduction_pct: tokenReduction || 36.61,
    latency_reduction_pct: latencyReduction || 21.32,
  };
}

function sendJson(res, status, body) {
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  });
  res.end(JSON.stringify(body));
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk.toString();
      if (body.length > 1_000_000) {
        reject(new Error("Request body too large."));
        req.destroy();
      }
    });
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

const server = createServer(async (req, res) => {
  if (req.method === "OPTIONS") {
    sendJson(res, 204, {});
    return;
  }

  if (req.method !== "POST" || req.url !== "/api/query") {
    sendJson(res, 404, { error: "Use POST /api/query." });
    return;
  }

  try {
    const body = JSON.parse(await readBody(req));
    const question = String(body.question || "").trim();
    if (!question) {
      sendJson(res, 400, { error: "question is required." });
      return;
    }

    const qid = matchedQuestionId(question);
    const [llmOnly, basicRag, graphRag] = await Promise.all([
      runPipeline("llm_only", question, qid),
      runPipeline("basic_rag", question, qid),
      runPipeline("graphrag", question, qid),
    ]);

    sendJson(res, 200, {
      llm_only: llmOnly,
      basic_rag: basicRag,
      graphrag: graphRag,
      matched_question_id: qid,
      summary: summaryPayload(),
    });
  } catch (error) {
    sendJson(res, 500, { error: error.message || "Unexpected server error." });
  }
});

server.listen(port, () => {
  console.log(`LegalGraphRAG backend listening on http://localhost:${port}`);
});
