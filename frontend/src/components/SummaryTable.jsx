import { pipelineMeta, summaryRows } from "../data/benchmark";
import BenchmarkCharts from "./BenchmarkCharts";

export default function SummaryTable() {
  return (
    <section className="panel">
      <div className="section-heading">
        <h2>Aggregate benchmark — 20 questions (pre-computed v9 results)</h2>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Pipeline</th>
              <th>Judge</th>
              <th>BERT raw</th>
              <th>BERT rescaled</th>
              <th>Avg tokens</th>
              <th>Avg latency</th>
            </tr>
          </thead>
          <tbody>
            {summaryRows.map((row) => {
              const isGraph = row.id === "graphrag";
              return (
                <tr className={isGraph ? "summary-graph-row" : ""} key={row.id}>
                  <td>
                    <span className="pipeline-dot" style={{ background: pipelineMeta[row.id].color }} />
                    {row.name}
                  </td>
                  <td>{row.judge}{isGraph ? " ✓" : ""}</td>
                  <td>{row.bertRaw}{isGraph ? " ✓" : ""}</td>
                  <td>{row.bertRescaled}{isGraph ? " ✓" : ""}</td>
                  <td>{row.avgTokens}</td>
                  <td>{row.avgLatency}{isGraph ? " ✓" : ""}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="table-note">
        Results from the latest full v9 benchmark run. GraphRAG is built on the official TigerGraph GraphRAG repo and every GraphRAG result carries upstream metadata. Live query metrics appear in the pipeline cards above.
      </p>
      <BenchmarkCharts />
    </section>
  );
}
