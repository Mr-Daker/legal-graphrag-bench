import { useEffect, useRef, useState } from "react";
import { benchmarkSeries, pipelineMeta, pipelineOrder, questionTypeSummary } from "../data/benchmark";

const chartKeys = ["latency", "bertRaw", "bertRescaled", "tokens"];

function flatten(series) {
  return pipelineOrder.flatMap((id) => series[id]);
}

function points(values, min, max, width, height, pad) {
  const range = max - min || 1;
  const step = (width - pad * 2) / (values.length - 1);

  return values
    .map((value, index) => {
      const x = pad + step * index;
      const y = height - pad - ((value - min) / range) * (height - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function coordinateList(values, min, max, width, height, pad) {
  const range = max - min || 1;
  const step = (width - pad * 2) / (values.length - 1);

  return values.map((value, index) => ({
    x: pad + step * index,
    y: height - pad - ((value - min) / range) * (height - pad * 2),
  }));
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function MetricLineChart({ config }) {
  const cardRef = useRef(null);
  const [isVisible, setIsVisible] = useState(false);
  const width = 560;
  const height = 210;
  const pad = 24;
  const allValues = flatten(config.series);
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);

  useEffect(() => {
    const node = cardRef.current;
    if (!node) return undefined;

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
  }, []);

  return (
    <article className={isVisible ? "metric-chart-card metric-chart-visible" : "metric-chart-card"} ref={cardRef}>
      <div className="metric-chart-head">
        <h3>{config.title}</h3>
        <span>{benchmarkSeries.labels[0]} → {benchmarkSeries.labels.at(-1)} · 7 local · 7 global · 6 multi-hop</span>
      </div>
      <svg className="metric-line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={config.title}>
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} />
        <text x={pad} y={18}>{config.formatter(max)}</text>
        <text x={pad} y={height - 6}>{config.formatter(min)}</text>
        {pipelineOrder.map((id, lineIndex) => {
          const coordinates = coordinateList(config.series[id], min, max, width, height, pad);
          return (
            <g key={id} style={{ "--line-index": lineIndex }}>
              <polyline
                className="metric-chart-line"
                fill="none"
                pathLength="1"
                points={points(config.series[id], min, max, width, height, pad)}
                stroke={pipelineMeta[id].color}
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={id === "graphrag" ? 4 : 3}
              />
              {coordinates.map((point, pointIndex) => (
                <circle
                  className="metric-chart-point"
                  cx={point.x}
                  cy={point.y}
                  fill={pipelineMeta[id].color}
                  key={`${id}-${pointIndex}`}
                  r={id === "graphrag" ? 3.1 : 2.6}
                  style={{ "--point-index": pointIndex }}
                />
              ))}
            </g>
          );
        })}
      </svg>
      <div className="metric-chart-legend">
        {pipelineOrder.map((id) => (
          <span key={id}>
            <i style={{ background: pipelineMeta[id].color }} />
            {pipelineMeta[id].name}
            <strong>{config.formatter(average(config.series[id]))}</strong>
          </span>
        ))}
      </div>
    </article>
  );
}

function JudgeTypeChart() {
  const cardRef = useRef(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const node = cardRef.current;
    if (!node) return undefined;

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
  }, []);

  return (
    <article className={isVisible ? "judge-type-card judge-type-visible" : "judge-type-card"} ref={cardRef}>
      <div className="metric-chart-head">
        <h3>Judge pass rate by question type</h3>
        <span>7 local · 7 global · 6 multi-hop</span>
      </div>
      <div className="judge-type-grid">
        {questionTypeSummary.map((group, groupIndex) => (
          <div className="judge-type-group" key={group.type}>
            <strong>{group.type}</strong>
            {pipelineOrder.map((id, barIndex) => (
              <div className="judge-type-row" key={id}>
                <span>
                  <i style={{ background: pipelineMeta[id].color }} />
                  {pipelineMeta[id].name}
                </span>
                <div className="judge-type-track">
                  <div
                    className="judge-type-fill"
                    style={{
                      "--target-width": `${group.values[id]}%`,
                      "--bar-index": groupIndex * pipelineOrder.length + barIndex,
                      background: pipelineMeta[id].color,
                    }}
                  />
                </div>
                <em>{group.values[id]}%</em>
              </div>
            ))}
          </div>
        ))}
      </div>
    </article>
  );
}

export default function BenchmarkCharts() {
  return (
    <div className="benchmark-charts" aria-label="Pre-computed benchmark trend charts">
      {chartKeys.map((key) => (
        <MetricLineChart config={benchmarkSeries[key]} key={key} />
      ))}
      <JudgeTypeChart />
    </div>
  );
}
