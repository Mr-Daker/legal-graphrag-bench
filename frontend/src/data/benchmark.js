export const defaultQuestion =
  "What constitutional rights are most frequently raised in the criminal appeal cases in this corpus?";

export const corpusInfo = {
  source: "harvard-lil/cold-cases",
  cases: 478,
  tokens: "2.2M",
  chunks: "7,008",
  model: "gemini-2.5-flash-lite",
};

export const sampleQuestionGroups = [
  {
    label: "Local factual",
    tone: "local",
    questions: [
      "What court decided Commonwealth v. Jarabek, and what citations does the case carry?",
      "What is the citation for the case Bruce v. ICI Americas, Inc.?",
      "What court decided State v. Howerton?",
    ],
  },
  {
    label: "Global synthesis",
    tone: "global",
    questions: [
      "What constitutional rights are most frequently raised in the criminal appeal cases in this corpus?",
      "What are the most common procedural grounds on which courts deny or dismiss appeals in this corpus?",
      "What types of civil disputes appear in this corpus alongside the criminal cases?",
    ],
  },
  {
    label: "Multi-hop",
    tone: "multi",
    questions: [
      "How do federal circuit courts in this corpus approach constitutional questions differently from state appellate courts?",
      "In criminal cases where ineffective assistance of counsel is claimed, how do courts in this corpus apply the two-part Strickland test?",
      "How does the standard of review applied by an appellate court affect the outcome of cases involving evidentiary rulings?",
    ],
  },
];

export const pipelineOrder = ["llm_only", "basic_rag", "graphrag"];

export const pipelineMeta = {
  llm_only: {
    name: "LLM-Only",
    color: "#f2a21b",
  },
  basic_rag: {
    name: "Basic RAG",
    color: "#4aa3ff",
  },
  graphrag: {
    name: "GraphRAG",
    color: "#10b981",
  },
};

export const initialResults = {
  llm_only: {
    answer:
      "The constitutional rights most frequently raised include the Fourth Amendment, Fifth Amendment, and Sixth Amendment.",
    tokens: 143,
    latency_ms: 1326,
    cost_usd: 0.000041,
    verdict: "FAIL",
    bertscore: { value: 0.817, matched: false },
  },
  basic_rag: {
    answer:
      "The provided context does not contain information about criminal appeal cases or the constitutional rights most frequently raised in them.",
    tokens: 3747,
    latency_ms: 1318,
    cost_usd: 0.000401,
    verdict: "FAIL",
    bertscore: { value: 0.834, matched: false },
  },
  graphrag: {
    answer:
      "The most frequently raised constitutional rights in criminal appeal cases are due process, Fourth Amendment search-and-seizure protections, Fifth Amendment self-incrimination and double-jeopardy rights, Sixth Amendment counsel and fair-trial rights, and Fourteenth Amendment equal-protection guarantees.",
    tokens: 2375,
    latency_ms: 1312,
    cost_usd: 0.000253,
    verdict: "PASS",
    bertscore: { value: 0.900, matched: false },
  },
};

export const summaryRows = [
  {
    id: "llm_only",
    name: "LLM-Only",
    judge: "35%",
    bertRaw: "0.817",
    bertRescaled: "0.451",
    avgTokens: "143",
    avgLatency: "1,326ms",
  },
  {
    id: "basic_rag",
    name: "Basic RAG",
    judge: "55%",
    bertRaw: "0.834",
    bertRescaled: "0.502",
    avgTokens: "3,747",
    avgLatency: "1,318ms",
  },
  {
    id: "graphrag",
    name: "GraphRAG",
    judge: "100%",
    bertRaw: "0.900",
    bertRescaled: "0.701",
    avgTokens: "2,375",
    avgLatency: "1,312ms",
  },
];

export const bonusCards = [
  {
    label: "Judge >=90%",
    value: "100%",
  },
  {
    label: "BERT raw >=0.88",
    value: "0.900",
  },
  {
    label: "BERT rescaled >=0.55",
    value: "0.701",
  },
  {
    label: "Token reduction >=30%",
    value: "36.6%",
  },
];

const labels = Array.from({ length: 20 }, (_, index) => `q${String(index + 1).padStart(3, "0")}`);

export const benchmarkSeries = {
  labels,
  latency: {
    title: "Latency by question",
    unit: "ms",
    formatter: (value) => `${Math.round(value).toLocaleString()}ms`,
    series: {
      llm_only: [1280, 2688, 1447, 937, 803, 1089, 1355, 756, 1977, 1159, 1148, 1074, 1043, 1444, 1135, 1377, 779, 3328, 902, 805],
      basic_rag: [1003, 2453, 1307, 923, 1044, 1068, 973, 1074, 1020, 1406, 1101, 1087, 1060, 935, 1248, 1802, 971, 2523, 1877, 1484],
      graphrag: [1685, 1579, 1485, 1062, 871, 967, 864, 805, 1010, 1703, 1073, 1177, 1226, 1256, 1117, 1043, 2936, 1206, 1305, 1879],
    },
  },
  bertRaw: {
    title: "BERT raw by question",
    unit: "",
    formatter: (value) => value.toFixed(3),
    series: {
      llm_only: [0.8913, 0.7329, 0.7582, 0.941, 0.7736, 0.9343, 0.8738, 0.8504, 0.8474, 0.8612, 0.7593, 0.8153, 0.8169, 0.7724, 0.7546, 0.7359, 0.8169, 0.7854, 0.8009, 0.8112],
      basic_rag: [0.9589, 0.6684, 0.7357, 0.9501, 0.8681, 0.9389, 0.9381, 0.86, 0.9163, 0.8303, 0.8536, 0.7596, 0.8799, 0.7742, 0.7713, 0.7455, 0.8205, 0.8107, 0.7901, 0.8037],
      graphrag: [0.9589, 0.9579, 0.8708, 0.8885, 0.8764, 0.9193, 0.9381, 0.9304, 0.9178, 0.8534, 0.8705, 0.9136, 0.8943, 0.8619, 0.9279, 0.8829, 0.8801, 0.8756, 0.9061, 0.8816],
    },
  },
  bertRescaled: {
    title: "BERT rescaled by question",
    unit: "",
    formatter: (value) => value.toFixed(3),
    series: {
      llm_only: [0.6743, 0.1998, 0.2757, 0.8231, 0.3218, 0.8031, 0.6218, 0.5519, 0.5427, 0.5841, 0.2789, 0.4467, 0.4515, 0.3183, 0.2648, 0.2087, 0.4515, 0.3572, 0.4034, 0.4343],
      basic_rag: [0.8769, 0.0067, 0.2083, 0.8506, 0.6048, 0.817, 0.8146, 0.5805, 0.7493, 0.4917, 0.5615, 0.2798, 0.6402, 0.3237, 0.3148, 0.2375, 0.4624, 0.4328, 0.371, 0.412],
      graphrag: [0.8769, 0.874, 0.6128, 0.6661, 0.6298, 0.7582, 0.8146, 0.7913, 0.7536, 0.5609, 0.612, 0.7413, 0.6834, 0.5863, 0.7839, 0.6491, 0.6407, 0.6274, 0.7186, 0.6453],
    },
  },
  tokens: {
    title: "Token usage by question",
    unit: "tok",
    formatter: (value) => `${Math.round(value).toLocaleString()} tok`,
    series: {
      llm_only: [62, 246, 232, 91, 82, 107, 110, 82, 100, 162, 114, 123, 126, 73, 181, 268, 76, 463, 80, 88],
      basic_rag: [3745, 3842, 3436, 3733, 3651, 3674, 3709, 3792, 3459, 3886, 3854, 3843, 3770, 3784, 3713, 3933, 3623, 3934, 3744, 3812],
      graphrag: [2174, 2173, 2255, 2173, 2549, 2382, 2158, 2342, 2116, 2313, 2564, 2780, 2606, 2652, 2100, 2180, 2240, 2712, 2393, 2640],
    },
  },
};

export const questionTypeSummary = [
  {
    type: "Local factual",
    values: {
      llm_only: 0,
      basic_rag: 100,
      graphrag: 100,
    },
  },
  {
    type: "Global synthesis",
    values: {
      llm_only: 57,
      basic_rag: 29,
      graphrag: 100,
    },
  },
  {
    type: "Multi-hop",
    values: {
      llm_only: 50,
      basic_rag: 33,
      graphrag: 100,
    },
  },
];
