import { bonusCards } from "../data/benchmark";

export default function BonusPanel() {
  return (
    <section className="panel bonus-panel">
      <div className="section-heading">
        <h2>Bonus thresholds</h2>
      </div>
      <div className="bonus-grid">
        {bonusCards.map((item) => (
          <article className="bonus-card" key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value} ✓</strong>
          </article>
        ))}
      </div>
    </section>
  );
}
