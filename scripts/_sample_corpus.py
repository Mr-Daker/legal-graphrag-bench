import json, random
from pathlib import Path
random.seed(42)
lines = Path("data/raw/cold_cases/opinions.jsonl").read_text(encoding="utf-8").splitlines()
samples = random.sample(lines, 25)
for line in samples:
    rec = json.loads(line)
    m = rec.get("metadata", {})
    print(rec.get("id"), "|", m.get("case_name"), "|", m.get("citations"))
    print(" ", rec.get("text","")[:180])
    print()
