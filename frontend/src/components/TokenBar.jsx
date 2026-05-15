import { useEffect, useRef, useState } from "react";
import { pipelineMeta, pipelineOrder } from "../data/benchmark";

function formatTokens(value) {
  if (!value) return "0 tok";
  return `${Number(value).toLocaleString()} tok`;
}

export default function TokenBar({ results, reduction = 36.61 }) {
  const panelRef = useRef(null);
  const [isVisible, setIsVisible] = useState(false);
  const basicTokens = Number(results.basic_rag?.tokens || 1);

  useEffect(() => {
    const node = panelRef.current;
    if (!node) return undefined;

    setIsVisible(false);
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.unobserve(entry.target);
        }
      },
      { threshold: 0.35 },
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, [results]);

  return (
    <section className={isVisible ? "panel token-panel token-panel-visible" : "panel token-panel"} ref={panelRef}>
      <div className="section-heading">
        <h2>Token usage comparison</h2>
        <span>GraphRAG saves {Number(reduction).toFixed(1)}% vs Basic RAG</span>
      </div>
      <div className="bars">
        {pipelineOrder.map((id, index) => {
          const meta = pipelineMeta[id];
          const tokens = Number(results[id]?.tokens || 0);
          const width = Math.max(5, Math.min(100, (tokens / basicTokens) * 100));
          return (
            <div className="bar-row" key={id} style={{ "--bar-index": index }}>
              <div className="bar-label">
                <span>{meta.name}{id === "graphrag" ? " ✓" : ""}</span>
                <strong className="bar-value">{formatTokens(tokens)}</strong>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ "--target-width": `${width}%`, background: meta.color }} />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
