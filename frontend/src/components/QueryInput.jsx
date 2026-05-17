import { useEffect, useRef, useState } from "react";
import { corpusInfo, sampleQuestionGroups } from "../data/benchmark";

export default function QueryInput({ question, setQuestion, onRun, onSampleSelect, loading, error }) {
  const [isPickerOpen, setIsPickerOpen] = useState(false);
  const pickerRef = useRef(null);

  useEffect(() => {
    if (!isPickerOpen) return undefined;

    const handlePointerDown = (event) => {
      if (pickerRef.current && !pickerRef.current.contains(event.target)) {
        setIsPickerOpen(false);
      }
    };

    document.addEventListener("pointerdown", handlePointerDown);
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, [isPickerOpen]);

  const selectQuestion = (sampleQuestion) => {
    setIsPickerOpen(false);
    onSampleSelect(sampleQuestion);
  };

  return (
    <section className="query-panel">
      <label htmlFor="query">Enter a legal question to run all 3 pipelines simultaneously</label>
      <div className="query-row">
        <input
          id="query"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") onRun();
          }}
        />
        <button className="run-button" type="button" onClick={() => onRun()} disabled={loading}>
          <span aria-hidden="true">{loading ? "●" : "▶"}</span>
          {loading ? "Running" : "Run all pipelines"}
        </button>
      </div>
      <div className="dataset-line">
        <span>{corpusInfo.source}</span>
        <span>{corpusInfo.cases} cases</span>
        <span>{corpusInfo.tokens} tokens</span>
        <span>{corpusInfo.model}</span>
        <div className="sample-picker" ref={pickerRef}>
          <button
            aria-expanded={isPickerOpen}
            className="sample-picker-trigger"
            onClick={() => setIsPickerOpen((open) => !open)}
            type="button"
          >
            Try a sample question
          </button>
          {isPickerOpen ? (
            <div className="sample-menu">
              {sampleQuestionGroups.map((group) => (
                <div className="sample-group" key={group.label}>
                  <h3>
                    <span className={`sample-dot sample-dot-${group.tone}`} />
                    {group.label}
                  </h3>
                  {group.questions.map((sampleQuestion) => (
                    <button
                      className="sample-question"
                      disabled={loading}
                      key={sampleQuestion}
                      onClick={() => selectQuestion(sampleQuestion)}
                      type="button"
                    >
                      {sampleQuestion}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          ) : null}
        </div>
      </div>
      {error ? <p className="error-line">{error}</p> : null}
    </section>
  );
}
