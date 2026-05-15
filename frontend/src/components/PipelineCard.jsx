import { useEffect, useMemo, useState } from "react";

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function formatMoney(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  return `$${Number(value).toFixed(4)}`;
}

function formatBert(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  return Number(value).toFixed(3);
}

function metricClass(highlight) {
  return highlight ? "metric-cell metric-cell-highlight" : "metric-cell";
}

function parseAnswerTokens(value) {
  const text = String(value || "No answer yet.")
    .replace(/(^|\s)\*\s+/g, "$1")
    .replace(/\r\n/g, "\n");
  const tokens = [];
  const boldPattern = /\*\*([\s\S]*?)\*\*/g;
  let cursor = 0;
  let match;

  while ((match = boldPattern.exec(text)) !== null) {
    if (match.index > cursor) {
      tokens.push({ text: text.slice(cursor, match.index).replaceAll("*", ""), bold: false });
    }

    tokens.push({ text: match[1].replaceAll("*", ""), bold: true });
    cursor = match.index + match[0].length;
  }

  if (cursor < text.length) {
    tokens.push({ text: text.slice(cursor).replaceAll("*", ""), bold: false });
  }

  return tokens.filter((token) => token.text.length > 0);
}

function sliceTokens(tokens, visibleCharacters) {
  let remaining = visibleCharacters;
  const visibleTokens = [];

  for (const token of tokens) {
    if (remaining <= 0) break;
    const text = token.text.slice(0, remaining);
    visibleTokens.push({ ...token, text });
    remaining -= text.length;
  }

  return visibleTokens;
}

function AnswerText({ answer }) {
  const tokens = useMemo(() => parseAnswerTokens(answer), [answer]);
  const totalCharacters = useMemo(
    () => tokens.reduce((total, token) => total + token.text.length, 0),
    [tokens],
  );
  const [visibleCharacters, setVisibleCharacters] = useState(totalCharacters);

  useEffect(() => {
    setVisibleCharacters(0);
    if (totalCharacters === 0) return undefined;

    const step = Math.max(2, Math.ceil(totalCharacters / 190));
    const timer = window.setInterval(() => {
      setVisibleCharacters((current) => {
        const next = Math.min(current + step, totalCharacters);
        if (next >= totalCharacters) {
          window.clearInterval(timer);
        }
        return next;
      });
    }, 34);

    return () => window.clearInterval(timer);
  }, [totalCharacters, answer]);

  const visibleTokens = sliceTokens(tokens, visibleCharacters);

  return (
    <div className="answer-text" aria-live="polite">
      {visibleTokens.map((token, index) =>
        token.bold ? (
          <strong className="answer-emphasis" key={`${index}-${token.text}`}>
            {token.text}
          </strong>
        ) : (
          <span key={`${index}-${token.text}`}>{token.text}</span>
        ),
      )}
      {visibleCharacters < totalCharacters ? <span className="typing-caret" aria-hidden="true" /> : null}
    </div>
  );
}

export default function PipelineCard({ name, color, answer, tokens, latency, cost, bertscore, verdict, isGraph }) {
  const verdictOk = String(verdict).toUpperCase() === "PASS";
  const isBenchmarkMatched = Boolean(bertscore?.matched);
  const bertValue = typeof bertscore === "object" && bertscore !== null ? bertscore.value : bertscore;

  return (
    <article className={isGraph ? "pipeline-card pipeline-card-graph" : "pipeline-card"}>
      <header className="pipeline-header">
        <div className="pipeline-name">
          <span className="pipeline-dot" style={{ background: color }} />
          <h2>{name}</h2>
        </div>
        <div className="pipeline-badges">
          {isGraph ? (
            <div className="graph-tag">
              <span>TigerGraph</span>
              <strong>Savanna</strong>
            </div>
          ) : null}
          <div className={verdictOk ? "verdict verdict-pass" : "verdict verdict-fail"}>
            {verdict || "N/A"}
          </div>
        </div>
      </header>

      <div className="answer-block">
        <p>Answer</p>
        <AnswerText answer={answer} />
      </div>

      <div className="metric-grid">
        <div className={metricClass(isGraph)}>
          <span>Tokens</span>
          <strong>{formatNumber(tokens)}</strong>
        </div>
        <div className={metricClass(isGraph)}>
          <span>Latency</span>
          <strong>{formatNumber(latency)}ms</strong>
        </div>
        <div className={metricClass(isGraph)}>
          <span>Cost</span>
          <strong>{formatMoney(cost)}</strong>
        </div>
        <div className={metricClass(isGraph)}>
          <span title={isBenchmarkMatched ? "Matched benchmark question BERTScore" : "Aggregate benchmark BERTScore shown for non-benchmark questions"}>
            BERTScore {isBenchmarkMatched ? "matched" : "aggregate"}
          </span>
          <strong>{formatBert(bertValue)}</strong>
        </div>
      </div>
    </article>
  );
}
