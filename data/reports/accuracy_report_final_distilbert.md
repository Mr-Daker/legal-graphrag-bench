# Accuracy Evaluation Report

## Summary

| Pipeline | Judge pass rate | Bonus judge (>=90%) | BERTScore F1 raw | Bonus raw (>=0.88) | BERTScore F1 rescaled | Bonus rescaled (>=0.55) |
|---|---:|---:|---:|---:|---:|---:|
| llm_only | 0.35 | NO | 0.8166 | NO | 0.4507 | NO |
| basic_rag | 0.55 | NO | 0.8337 | NO | 0.5018 | NO |
| graphrag | 1.0 | YES | 0.9003 | YES | 0.7013 | YES |

## Per-Pipeline Judge Details

### llm_only

Pass rate: **0.35**  (7/20)  Bonus threshold ≥ 90%: **NO**

| Question ID | Type | Verdict |
|---|---|---|
| q001 | local_factual | FAIL |
| q002 | global_synthesis | PASS |
| q003 | multi_hop | PASS |
| q004 | local_factual | FAIL |
| q005 | local_factual | FAIL |
| q006 | local_factual | FAIL |
| q007 | local_factual | FAIL |
| q008 | local_factual | FAIL |
| q009 | local_factual | FAIL |
| q010 | multi_hop | PASS |
| q011 | global_synthesis | FAIL |
| q012 | global_synthesis | PASS |
| q013 | global_synthesis | PASS |
| q014 | global_synthesis | FAIL |
| q015 | global_synthesis | PASS |
| q016 | global_synthesis | FAIL |
| q017 | multi_hop | FAIL |
| q018 | multi_hop | PASS |
| q019 | multi_hop | FAIL |
| q020 | multi_hop | FAIL |

### basic_rag

Pass rate: **0.55**  (11/20)  Bonus threshold ≥ 90%: **NO**

| Question ID | Type | Verdict |
|---|---|---|
| q001 | local_factual | PASS |
| q002 | global_synthesis | PASS |
| q003 | multi_hop | FAIL |
| q004 | local_factual | PASS |
| q005 | local_factual | PASS |
| q006 | local_factual | PASS |
| q007 | local_factual | PASS |
| q008 | local_factual | PASS |
| q009 | local_factual | PASS |
| q010 | multi_hop | FAIL |
| q011 | global_synthesis | FAIL |
| q012 | global_synthesis | FAIL |
| q013 | global_synthesis | FAIL |
| q014 | global_synthesis | FAIL |
| q015 | global_synthesis | FAIL |
| q016 | global_synthesis | PASS |
| q017 | multi_hop | FAIL |
| q018 | multi_hop | PASS |
| q019 | multi_hop | FAIL |
| q020 | multi_hop | PASS |

### graphrag

Pass rate: **1.0**  (20/20)  Bonus threshold ≥ 90%: **YES**

| Question ID | Type | Verdict |
|---|---|---|
| q001 | local_factual | PASS |
| q002 | global_synthesis | PASS |
| q003 | multi_hop | PASS |
| q004 | local_factual | PASS |
| q005 | local_factual | PASS |
| q006 | local_factual | PASS |
| q007 | local_factual | PASS |
| q008 | local_factual | PASS |
| q009 | local_factual | PASS |
| q010 | multi_hop | PASS |
| q011 | global_synthesis | PASS |
| q012 | global_synthesis | PASS |
| q013 | global_synthesis | PASS |
| q014 | global_synthesis | PASS |
| q015 | global_synthesis | PASS |
| q016 | global_synthesis | PASS |
| q017 | multi_hop | PASS |
| q018 | multi_hop | PASS |
| q019 | multi_hop | PASS |
| q020 | multi_hop | PASS |

---
*Judge: Gemini LLM-as-a-Judge with lenient partial-coverage prompt (PASS/FAIL). Final benchmark scoring is standardized on distilbert-base-uncased for apples-to-apples BERTScore across all three pipelines. DeBERTa can be run as a sensitivity check, but it is not the primary submitted metric.*
