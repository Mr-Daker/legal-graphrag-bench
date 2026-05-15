# Pipeline Comparison Report

## Summary

| Pipeline | Questions | Avg tokens | Avg prompt | Avg completion | Avg latency ms | Heuristic pass rate |
|---|---:|---:|---:|---:|---:|---:|
| llm_only | 20 | 143.3 | 54.65 | 88.65 | 1326.32 | 0.7 |
| basic_rag | 20 | 3746.85 | 3659.55 | 87.3 | 1318.0 | 0.75 |
| graphrag | 20 | 2375.1 | 2323.55 | 51.55 | 1312.45 | 0.95 |

## Official TigerGraph GraphRAG Base

- Built on: `https://github.com/tigergraph/graphrag`
- Commit: `f649f4197f3dc18bf5bd7dd2fb0a0e477a5a70b9`
- Metadata coverage: `20/20` GraphRAG rows
- All rows include metadata: `True`
- Customization: LegalGraphRAG customizes TigerGraph GraphRAG with a legal schema, EA-GraphRAG routing, CommunityReport retrieval, and PathRAG-light pruning.

## Accuracy Evaluation (LLM-as-a-Judge + BERTScore)

| Pipeline | Judge pass rate | Bonus judge (>=90%) | BERTScore F1 raw | Bonus raw (>=0.88) | BERTScore F1 rescaled | Bonus rescaled (>=0.55) |
|---|---:|---:|---:|---:|---:|---:|
| llm_only | 0.35 | NO | 0.8166 | NO | 0.4507 | NO |
| basic_rag | 0.55 | NO | 0.8337 | NO | 0.5018 | NO |
| graphrag | 1.0 | YES | 0.9003 | YES | 0.7013 | YES |

## GraphRAG vs Basic RAG

- `graphrag_vs_basic_rag_token_reduction_pct`: `36.61%`
- `graphrag_vs_basic_rag_latency_reduction_pct`: `0.42%`

## Per Question

### q001

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 62 | 1280.07 | 0.625 | True | The Supreme Court of North Carolina decided the case State v. Howerton. |
| basic_rag | 3745 | 1003.44 | 1.0 | True | The Court of Criminal Appeals of Oklahoma decided the case State v. Howerton. |
| graphrag | 2174 | 1684.97 | 1.0 | True | The Court of Criminal Appeals of Oklahoma decided the case State v. Howerton. |

### q002

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 246 | 2687.65 | 0.1928 | False | Common legal themes across criminal appeal opinions include:  *   **Sufficiency of evidence:** Whether the evidence presented at trial was enough to support the conviction. *   **E |
| basic_rag | 3842 | 2453.49 | 0.0976 | False | Common legal themes across criminal appeal opinions include:  *   **Appeals from convictions and sentences:** Many cases involve defendants appealing their convictions and sentence |
| graphrag | 2173 | 1578.69 | 0.8 | True | Common legal themes across criminal appeal opinions include standards of review, sufficiency of the evidence, ineffective assistance of counsel, and preservation of error. |

### q003

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 232 | 1446.77 | 0.1064 | False | Cited precedents influence a court's reasoning in criminal appeal opinions by providing a framework of established legal principles and past judicial decisions. Courts use preceden |
| basic_rag | 3436 | 1307.47 | 0.0 | False | The answer is not available in the retrieved context. |
| graphrag | 2255 | 1484.77 | 0.4375 | True | Cited precedents influence criminal appeal opinions by providing controlling rules, standards of review, and analogies or distinctions to similar facts. These precedents help court |

### q004

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 91 | 937.03 | 0.9231 | True | The citation for the case Bruce v. ICI Americas, Inc. is 931 F. Supp. 1066 (S.D. Ga. 1996). |
| basic_rag | 3733 | 922.81 | 0.9231 | True | The citation for the case Bruce v. ICI Americas, Inc. is 933 F. Supp. 781. |
| graphrag | 2173 | 1062.16 | 0.6 | True | The citation for the case *Bruce v. ICI Americas, Inc.* is 933 F. Supp. 781. This case was decided by the United States District Court on May 15, 1996. |

### q005

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 82 | 802.79 | 0.5 | True | The case *Bell v. Turner* was decided in **North Carolina**. It was heard by the **North Carolina Court of Appeals**. |
| basic_rag | 3651 | 1043.64 | 0.6 | True | The case Bell v. Turner was decided in Ohio, and the appellate district that heard it was the 4th District. |
| graphrag | 2549 | 871.41 | 0.7 | True | The case *Bell v. Turner* was decided in Ohio. The Fourth Appellate District heard the case. |

### q006

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 107 | 1089.08 | 0.9286 | True | The court that decided Rosetta Stone Ltd. v. Google, Inc. was the United States Court of Appeals for the Fourth Circuit. The citation is 675 F.3d 310 (4th Cir. 2012). |
| basic_rag | 3674 | 1067.77 | 1.0 | True | The United States Court of Appeals for the Fourth Circuit decided *Rosetta Stone Ltd. v. Google, Inc.*, with the citation 676 F.3d 144. |
| graphrag | 2382 | 966.96 | 0.9167 | True | The United States Court of Appeals decided *Rosetta Stone Ltd. v. Google, Inc.*, and the citation is 676 F.3d 144. |

### q007

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 110 | 1355.3 | 0.8571 | True | The court that decided *Commonwealth v. Jarabek* was the **Supreme Judicial Court of Massachusetts**.  The citations for the case are: *   384 Mass. 595 *   428 N.E.2d 1347 |
| basic_rag | 3709 | 973.21 | 0.8571 | True | The Supreme Judicial Court of Massachusetts decided Commonwealth v. Jarabek. The case carries the citations 424 N.E.2d 491 and 384 Mass. 293. |
| graphrag | 2158 | 863.77 | 0.8571 | True | The Supreme Judicial Court of Massachusetts decided Commonwealth v. Jarabek. The case carries the citations 424 N.E.2d 491 and 384 Mass. 293. |

### q008

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 82 | 756.27 | 0.6 | True | The Idaho Supreme Court decided State v. Jones in the 2003 Idaho case involving defendant Michael Wayne Jones. |
| basic_rag | 3792 | 1074.46 | 0.7 | True | The Court of Appeals of Idaho decided *State v. Jones* in the 2003 Idaho case involving defendant Michael Wayne Jones. |
| graphrag | 2342 | 805.45 | 0.75 | True | The Court of Appeals of Idaho decided State v. Jones, citing 77 P.3d 988 and 139 Idaho 299. |

### q009

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 100 | 1976.97 | 0.5 | True | The citation for Forgus v. Hodnett is 135 Cal. App. 4th 1178 (2006). The court that decided it was the California Court of Appeal, Fourth District. |
| basic_rag | 3459 | 1020.23 | 0.875 | True | The citation for Forgus v. Hodnett is 405 S.W.2d 337, and the court that decided it was the Supreme Court of Texas. |
| graphrag | 2116 | 1010.26 | 0.8235 | True | The citation for *Forgus v. Hodnett* is 405 S.W.2d 337. The Supreme Court of Texas decided this case. |

### q010

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 162 | 1159.49 | 0.6761 | True | The federal district court in *Sanders Ex Rel. Rayl v. Kansas Department of Social and Rehabilitation Services* analyzed the civil rights claims by determining whether the state ag |
| basic_rag | 3886 | 1405.63 | 0.4651 | True | The provided context does not contain information about how the federal district court in *Sanders Ex Rel. Rayl v. Kansas Department of Social and Rehabilitation Services* analyzed |
| graphrag | 2313 | 1703.19 | 0.6176 | True | The United States District Court for the District of Kansas analyzed the civil rights claims by considering due process and equal protection under the Constitution, as well as fede |

### q011

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 114 | 1148.25 | 0.5161 | True | The case corpus includes cases from the following types of courts:  *   **District Courts:** These are the trial courts of the federal court system. *   **Courts of Appeals:** Thes |
| basic_rag | 3854 | 1100.92 | 0.3902 | True | The case corpus represents the following types of courts: Supreme Court of Pennsylvania, Supreme Court of Vermont, United States Supreme Court, Georgia Supreme Court, North Carolin |
| graphrag | 2564 | 1073.0 | 0.4651 | True | The case corpus includes United States District Courts, United States Courts of Appeals, and state supreme courts. Specialized state criminal appellate courts are also represented. |

### q012

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 123 | 1074.23 | 0.6296 | True | The constitutional rights most frequently raised in the criminal appeal cases in this corpus are:  *   **Fourth Amendment:** This amendment protects against unreasonable searches a |
| basic_rag | 3843 | 1086.71 | 0.2174 | False | The provided context does not contain information about criminal appeal cases or the constitutional rights most frequently raised in them. Therefore, the answer is not available in |
| graphrag | 2780 | 1176.85 | 0.7042 | True | The most frequently raised constitutional rights in criminal appeal cases are due process, Fourth Amendment search and seizure protections, and Fourteenth Amendment equal protectio |

### q013

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 126 | 1043.05 | 0.2727 | False | Appellate courts in this corpus typically address claims of insufficient evidence by reviewing the evidence presented at trial to determine if it was legally sufficient to support  |
| basic_rag | 3770 | 1059.63 | 0.6111 | True | Appellate courts review the sufficiency of the evidence de novo. They examine the admitted evidence to determine if it would convince the average mind of guilt beyond a reasonable  |
| graphrag | 2606 | 1225.89 | 0.5484 | False | Appellate courts typically review sufficiency of the evidence claims by determining if a rational trier of fact could find the elements of the offense beyond a reasonable doubt, vi |

### q014

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 73 | 1444.22 | 0.2128 | False | The provided text does not contain information about patterns in civil cases involving government or state agencies as defendants. |
| basic_rag | 3784 | 934.84 | 0.2 | False | The provided context does not contain sufficient information to answer the question about patterns in civil cases involving government or state agencies as defendants. |
| graphrag | 2652 | 1256.37 | 0.4722 | True | Civil cases involving government or state agencies as defendants frequently involve issues of sovereign immunity and qualified immunity. These cases often center on due process and |

### q015

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 181 | 1134.67 | 0.1918 | False | The corpus contains civil disputes related to:  *   **Contract disputes:** This includes disagreements over the terms, performance, or breach of contracts. *   **Property disputes: |
| basic_rag | 3713 | 1247.83 | 0.2381 | True | The corpus contains civil disputes related to habeas corpus proceedings, as seen in *Cagle v. Davis*. Additionally, there are civil disputes concerning the placement of a child for |
| graphrag | 2100 | 1117.09 | 0.7429 | True | The corpus contains civil disputes alongside criminal cases, including contract disputes, civil rights claims, and intellectual property or trademark matters. |

### q016

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 268 | 1376.67 | 0.2128 | False | The most common procedural grounds on which courts deny or dismiss appeals in this corpus are:  *   **Lack of standing:** The appellant does not have a sufficient stake in the outc |
| basic_rag | 3933 | 1801.71 | 0.2308 | True | The provided context does not contain a comprehensive list of the most common procedural grounds on which courts deny or dismiss appeals. However, it does mention a few specific in |
| graphrag | 2180 | 1042.95 | 0.52 | True | Courts most commonly deny or dismiss appeals on procedural grounds such as waiver or failure to preserve error, lack of jurisdiction, and untimeliness. These grounds are often appl |

### q017

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 76 | 778.95 | 0.3137 | True | The provided text does not contain information on how courts distinguish between direct appeals and post-conviction relief petitions. |
| basic_rag | 3623 | 971.42 | 0.3019 | True | The provided context does not contain sufficient information to answer how courts distinguish between direct appeals and post-conviction relief petitions. |
| graphrag | 2240 | 2936.18 | 0.5867 | True | Courts distinguish between direct appeals and post-conviction relief petitions by noting that direct appeals challenge trial error soon after judgment. Post-conviction or habeas pe |

### q018

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 463 | 3327.7 | 0.3165 | True | The standard of review applied by an appellate court significantly affects the outcome of cases involving evidentiary rulings. Different standards of review give appellate courts v |
| basic_rag | 3934 | 2523.38 | 0.3137 | True | The standard of review applied by an appellate court affects the outcome of cases involving evidentiary rulings by determining the level of deference given to the trial court's dec |
| graphrag | 2712 | 1205.67 | 0.5 | True | Appellate courts review a district court's evidentiary rulings for an abuse of discretion, meaning they will only reverse if they are firmly convinced a mistake was made that affec |

### q019

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 80 | 901.95 | 0.2545 | True | I cannot answer this question. The provided text does not contain information about how federal circuit courts approach constitutional questions differently from state appellate co |
| basic_rag | 3744 | 1877.08 | 0.1951 | False | The provided context does not contain sufficient information to answer how federal circuit courts approach constitutional questions differently from state appellate courts. The exc |
| graphrag | 2393 | 1304.84 | 0.623 | True | Federal circuit courts review pure constitutional questions de novo, applying federal constitutional doctrine and Supreme Court precedent. State appellate courts, however, consider |

### q020

| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |
|---|---:|---:|---:|---|---|
| llm_only | 88 | 805.25 | 0.3214 | True | The corpus does not contain information on how courts apply the two-part Strickland test in criminal cases where ineffective assistance of counsel is claimed. |
| basic_rag | 3812 | 1484.27 | 0.2927 | True | In criminal cases where ineffective assistance of counsel is claimed, courts apply the two-part *Strickland* test. This test requires the defendant to show: (1) that counsel's perf |
| graphrag | 2640 | 1878.53 | 0.5333 | True | In criminal cases claiming ineffective assistance of counsel, courts apply the two-part Strickland test by first determining if counsel's performance was deficient, meaning it fell |

## Notes

- Heuristic pass and lexical F1 are lightweight local checks, not replacements for the required LLM-as-a-Judge and BERTScore evaluation.
- Costs are intentionally omitted here; add provider pricing at report time to avoid stale pricing assumptions.
