# CineScope — RAG Evaluation Report

## 1. Overview

CineScope is a multi-agent movie research and discovery assistant built using LangGraph, OpenAI, NVIDIA NIM, Chroma, SQLite, and Tavily Remote MCP.

This evaluation measures the quality of CineScope's Retrieval-Augmented Generation (RAG) pipeline, focusing on retrieval relevance, factual grounding, answer quality, and retrieval coverage.

The evaluation uses **RAGAS 0.4.3** and a fixed dataset of 15 movie research questions.

Following the Assignment 2 clarification, the evaluation isolates the vector-store RAG pipeline. The Research Agent's optional Tavily web search is disabled, and no SQLite or News Agent retrieval is included in the RAGAS metrics.

## 2. RAG Architecture

CineScope implements a standard single-pass RAG pipeline:

1. **Ingestion:** Movie research articles are collected from English Wikipedia.
2. **Chunking:** Documents are divided into overlapping text chunks.
3. **Embedding and storage:** Chunks are embedded using OpenAI and stored in Chroma.
4. **Retrieval:** The user query is embedded and the top five semantically similar chunks are retrieved.
5. **Generation:** Retrieved passages are injected into the LLM prompt to generate an answer with numbered source citations.

### Configuration

| Component | Configuration |
|---|---|
| Data source | English Wikipedia |
| Movie corpus | 75 films |
| Research documents | 75 |
| Vector database | Chroma |
| Total indexed chunks | 5,954 |
| Chunk size | 1,000 characters |
| Chunk overlap | 150 characters |
| Embedding model | `text-embedding-3-small` |
| Retrieval strategy | Single-pass vector similarity |
| Top-k | 5 |
| Generation model | `gpt-4.1-mini` |
| Evaluation framework | RAGAS 0.4.3 |
| Evaluation judge | `gpt-4.1-mini` |
| Evaluator embeddings | `text-embedding-3-small` |

The same fixed research corpus is used for all evaluation questions.

## 3. Evaluation Dataset

The evaluation dataset contains 15 predefined questions stored in `evals/rag_test_cases.json`.

| Category | Questions | Purpose |
|---|---:|---|
| Single-movie research | 6 | Evaluate thematic and conceptual analysis |
| Multi-movie comparisons | 4 | Evaluate retrieval and synthesis across multiple films |
| Factual questions | 3 | Evaluate straightforward fact retrieval |
| Insufficient evidence | 2 | Evaluate the system's behavior when evidence is unavailable |
| **Total** | **15** | |

Each test case includes a unique ID, category, question, reference answer, and expected TMDB movie IDs.

The reference answers provide evaluation targets rather than requiring exact string matches.

## 4. Evaluation Methodology

The evaluation uses `evals/evaluate_rag.py`.

For each question, the script:

1. Calls the standalone `answer_question()` function from `src/rag.py`.
2. Retrieves five passages from the Chroma vector store.
3. Generates an answer using the retrieved passages.
4. Evaluates the generated answer and retrieved contexts using four RAGAS metrics.
5. Computes movie-ID retrieval coverage and precision.
6. Records RAG pipeline latency.
7. Saves detailed results to JSON and CSV files.

No live web search is performed during evaluation.

### RAGAS Metrics

**Faithfulness:** Measures whether factual claims in the generated answer are supported by retrieved context.

**Answer Relevancy:** Measures how well the generated answer addresses the user's question.

**Context Precision:** Measures the relevance and ranking of retrieved passages against the reference answer.

**Context Recall:** Measures whether the retrieved context contains the information needed to support the reference answer.

### Additional Retrieval Metrics

Movie retrieval coverage is calculated as:

`|Expected Movie IDs ∩ Retrieved Movie IDs| / |Expected Movie IDs|`

Movie retrieval precision is calculated as:

`|Expected Movie IDs ∩ Retrieved Movie IDs| / |Retrieved Movie IDs|`

These metrics evaluate movie identity, not passage-level relevance. Retrieving the correct film does not necessarily mean retrieving the most relevant passage.

Latency measures retrieval and answer generation, excluding the time spent calculating RAGAS metrics.

## 5. Evaluation Results

The complete evaluation was executed on October 20, 2026.

### Aggregate Results

| Metric | Score |
|---|---:|
| Faithfulness | **0.949** |
| Answer Relevancy | **0.751** |
| Context Precision | **0.587** |
| Context Recall | **0.867** |
| Movie Retrieval Coverage | **1.000** |
| Movie Retrieval Precision | **0.967** |
| Average RAG Latency | **1.984 seconds** |

The results indicate strong factual grounding and movie-level retrieval coverage, with opportunities to improve passage relevance and ranking.

### Per-Question Results

| ID | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---:|---:|---:|---:|
| R01 | 0.929 | 0.979 | 0.325 | 1.000 |
| R02 | 0.900 | 0.971 | 1.000 | 1.000 |
| R03 | 1.000 | 0.948 | 0.887 | 1.000 |
| R04 | 1.000 | 0.966 | 0.478 | 1.000 |
| R05 | 0.706 | 0.858 | 1.000 | 0.500 |
| R06 | 0.800 | 0.000 | 0.000 | 0.000 |
| R07 | 1.000 | 0.891 | 1.000 | 1.000 |
| R08 | 0.944 | 0.893 | 0.000 | 1.000 |
| R09 | 0.955 | 0.934 | 1.000 | 1.000 |
| R10 | 1.000 | 0.921 | 0.200 | 1.000 |
| R11 | 1.000 | 1.000 | 0.833 | 1.000 |
| R12 | 1.000 | 0.903 | 0.833 | 1.000 |
| R13 | 1.000 | 1.000 | 0.887 | 1.000 |
| R14 | 1.000 | 0.000 | 0.000 | 1.000 |
| R15 | 1.000 | 0.000 | 0.367 | 0.500 |

## 6. Findings and Analysis

### 6.1 Strong Factual Grounding

The average faithfulness score of 0.949 indicates that most generated factual claims were supported by retrieved evidence.

Several questions achieved a perfect faithfulness score, including research on *Parasite*, *Ex Machina*, and cross-movie comparisons.

However, R05 (*Interstellar*) achieved 0.706, indicating that some generated claims may not have been fully supported by the retrieved passages.

This is an area for further investigation.

### 6.2 Strong Movie-Level Retrieval

Movie retrieval coverage was 1.000 across the evaluation dataset.

This means every question retrieved at least one passage from each expected movie.

Movie retrieval precision averaged 0.967, indicating that most distinct retrieved movie IDs matched the expected films.

However, movie-level accuracy alone does not establish passage relevance.

### 6.3 Passage Relevance Requires Improvement

Average context precision was 0.587.

Some questions received low scores despite retrieving the correct films:

- R01 — *The Matrix*: 0.325
- R04 — *Ex Machina*: 0.478
- R08 — *Arrival* versus *Interstellar*: 0.000
- R10 — *Ex Machina* versus *The Matrix*: 0.200

These results suggest that the top-five similarity search may retrieve passages that are related to the correct films but insufficiently relevant to the specific question.

Potential improvements include query-aware reranking, movie-specific filtering, and improved retrieval for comparisons.

These improvements have not yet been implemented or benchmarked.

### 6.4 Multi-Movie Comparisons

The four comparison questions achieved full movie retrieval coverage.

This demonstrates that the current global vector search can retrieve evidence from both requested films.

However, low context precision for R08 and R10 suggests that retrieving the correct movies is not always sufficient to provide the most useful evidence.

A future improvement could retrieve relevant passages independently for each explicitly requested movie.

### 6.5 Insufficient-Evidence Questions

R14 and R15 received answer relevancy scores of 0.000.

These questions deliberately request information that may not be supported by the fixed research corpus.

A low answer relevancy score does not necessarily indicate incorrect behavior in this category. A system that correctly refuses to invent unsupported facts may receive a low score against a conventional relevancy metric.

These cases require separate manual inspection of the generated answers to determine whether the system appropriately acknowledged insufficient evidence.

### 6.6 Arrival Evaluation Anomaly

R06, concerning the relationship between language and time in *Arrival*, received:

- Faithfulness: 0.800
- Answer relevancy: 0.000
- Context precision: 0.000
- Context recall: 0.000

Movie retrieval coverage and precision were both 1.000.

This discrepancy requires examination of the generated answer, reference answer, and retrieved passages. The current results alone do not establish whether the problem lies in retrieval, generation, or metric judgment.

## 7. Limitations

The evaluation has several limitations.

**Small test dataset:** Fifteen questions provide a useful baseline but do not establish performance across all possible movie research requests.

**Single evaluation run:** RAGAS uses LLM-based judgments, so scores may vary between runs.

**Fixed top-k retrieval:** Five globally retrieved chunks may not provide balanced evidence for every comparison question.

**Limited corpus:** The dataset covers 75 popular movies rather than the full range of cinema.

**LLM-as-judge limitations:** Automated metric scores are estimates of quality and should be supplemented with manual inspection.

**Abstention evaluation:** Standard answer relevancy metrics may not adequately assess questions where the correct response is to acknowledge missing evidence.

**No live-web evaluation:** Tavily results were deliberately excluded to preserve reproducibility, consistent with the instructor's clarification.

## 8. Potential Improvements

The following improvements could be evaluated in future iterations:

1. Movie-aware retrieval using TMDB IDs to restrict or balance relevant passages.
2. Reranking retrieved chunks according to question relevance.
3. Improved handling of multi-movie comparison questions.
4. Stronger validation of citation identifiers against retrieved evidence.
5. Separate abstention accuracy evaluation for unsupported questions.
6. A larger and more diverse evaluation dataset.

These are proposed improvements, not claims of implemented functionality.

## 9. Reproducibility

The evaluation can be reproduced using the isolated evaluation environment.

From the CineScope repository root:

```bash
source .venv-eval/bin/activate
PYTHONPATH=. python evals/evaluate_rag.py
```

The evaluation requires a configured OpenAI API key, the local Chroma index, and the movie research dataset.

To reproduce the full 15-question run, ensure `selected_cases = test_cases` in `evals/evaluate_rag.py`.

Results are saved under:

```text
evals/results/
```

Baseline output files:

```text
ragas_20261020_234001.json
ragas_20261020_234001.csv
```

For reproducibility, the final repository should also preserve the dependency versions used for evaluation and the configuration required to rebuild the Chroma index.

## 10. Conclusion

CineScope's single-pass RAG pipeline demonstrates strong factual grounding and movie-level retrieval coverage across the 15-question evaluation set.

The baseline achieved 0.949 faithfulness, 0.867 context recall, and 1.000 movie retrieval coverage.

The primary improvement opportunity is passage-level retrieval relevance, reflected in the context precision score of 0.587.

These findings provide a measurable baseline for future retrieval improvements while demonstrating the required ingestion, embedding, retrieval, grounded generation, citation, and evaluation stages of the Assignment 2 RAG pipeline.

## Manual Error Analysis

The generated answers and retrieved passages were manually inspected to investigate low-scoring evaluation cases.

### R06 — Arrival: Language and Time

The retriever returned five passages from the correct movie, but none directly explained the relationship between learning the alien language and perceiving time nonlinearly.

CineScope consequently acknowledged insufficient evidence rather than generating an unsupported explanation.

This is a retrieval failure rather than a hallucination.

### R08 — Arrival versus Interstellar

Both requested films were retrieved, but several passages focused on critical reception, filmmaking influences, or references instead of the philosophical concepts requested.

This demonstrates that movie-level retrieval coverage does not guarantee relevant passage selection.

### R10 — Ex Machina versus The Matrix

The retriever identified both movies, but some passages were only indirectly related to artificial intelligence.

The generated comparison remained relevant, but context precision was low.

### R11 — The Dark Knight

One of the five retrieved passages belonged to The Dark Knight Rises rather than The Dark Knight.

The correct director was nevertheless retrieved and reported with a valid citation.

This illustrates the limitations of global similarity search for films with closely related titles.

### R14 and R15 — Insufficient Evidence

Both generated responses correctly acknowledged that the retrieved corpus did not establish the requested information.

Manual abstention accuracy was 2/2 (100%) on these two test cases.

The RAGAS answer relevancy metric returned zero for both cases, so these scores should not be interpreted as evidence of hallucination.

### Overall Assessment

The evaluation indicates strong factual grounding and movie-level retrieval coverage, while highlighting opportunities to improve passage-level relevance.

The system also demonstrated appropriate behavior when the fixed corpus lacked sufficient evidence.

These results establish a baseline for future retrieval improvements without requiring a more complex RAG architecture.


---

## 11. Performance Benchmarking

### 11.1 Objective

CineScope was benchmarked to measure retrieval latency, LLM generation latency, time-to-first-token (TTFT), end-to-end response latency, and Cloud Run cold-start behavior.

Two separate benchmarking scripts were implemented:

- `benchmarks/benchmark_rag.py` — measures local Chroma retrieval and OpenAI RAG generation latency.
- `benchmarks/benchmark_cloud.py` — measures end-to-end latency and TTFT through CineScope's public Cloud Run SSE endpoint.

The benchmarks were conducted on October 21, 2026, using the deployed CineScope service in Google Cloud Run (`us-west1`).

### 11.2 Local RAG Pipeline Latency

Five movie research questions were evaluated using the standalone RAG pipeline.

| Question | Retrieval (s) | LLM (s) | Total (s) |
|---|---:|---:|---:|
| The Matrix — Simulated reality | 2.243 | 4.269 | 6.659 |
| WALL-E — Environmental responsibility | 0.846 | 2.205 | 3.052 |
| Parasite — Class inequality | 0.868 | 1.733 | 2.602 |
| Interstellar — Time dilation and gravity | 0.278 | 2.753 | 3.032 |
| The Matrix vs. Inception — Reality | 0.676 | 3.568 | 4.244 |
| **Average** | **0.982** | **2.906** | **3.918** |

Retrieval latency includes query embedding generation and Chroma vector similarity search. LLM latency includes the remote OpenAI generation request. Total latency also includes prompt construction and application overhead.

The results show that LLM generation accounted for approximately 74% of average RAG latency, while retrieval accounted for approximately 25%.

The first query had the highest retrieval latency, potentially reflecting initialization and cache effects. This explanation has not been independently verified.

### 11.3 Cloud Run End-to-End Performance

CineScope was evaluated through its publicly accessible Cloud Run endpoint:

`https://csci599-a2-cinescope-887999373674.us-west1.run.app/health`

Three representative queries were used:

1. Discovery — Find science-fiction movies directed by Christopher Nolan.
2. Research — How does The Matrix explore simulated reality?
3. News — What are the latest announcements about Dune Part Three?

Each benchmark invocation submitted the three questions sequentially. Three invocations were executed consecutively.

TTFT was measured from the beginning of the HTTP request until the first nonempty answer-token SSE event was received.

Total latency was measured from request initiation until completion of the SSE response.

Because CineScope streams the final synthesis output, TTFT includes supervisor routing, specialized agent execution, tool calls, and the delay before synthesis begins.

#### Results Across Three Runs

| Run | Average TTFT (s) | Average Total Latency (s) |
|---|---:|---:|
| Run 1 — Initial benchmark | 32.407 | 37.070 |
| Run 2 — Warm benchmark | 16.661 | 21.025 |
| Run 3 — Warm benchmark | 16.743 | 21.276 |

The first benchmark invocation was slower than the subsequent two. However, it was not independently verified as a cold-start run and is therefore classified as the initial benchmark rather than a confirmed cold start.

#### Warm Performance by Agent

The second and third benchmark invocations were used to calculate warm-performance averages.

| Agent | Mean TTFT (s) | Mean Total Latency (s) |
|---|---:|---:|
| Discovery | 15.456 | 17.303 |
| Research | 6.433 | 9.868 |
| News | 28.217 | 36.282 |
| **Overall** | **16.702** | **21.151** |

The Research Agent exhibited the lowest end-to-end latency among the three workflows.

The News Agent exhibited the highest latency. Its workflow includes remote Tavily MCP retrieval and LLM-based synthesis, which may contribute to the additional response time.

These measurements do not isolate the latency of individual LangGraph nodes or remote MCP calls.

### 11.4 Cloud Run Cold-Start Observations

CineScope was deployed with the following configuration:

| Setting | Value |
|---|---|
| Platform | Google Cloud Run |
| Region | `us-west1` |
| CPU | 1 vCPU |
| Memory | 2 GiB |
| Concurrency | 4 |
| Request timeout | 300 seconds |
| Minimum instances | 0 |
| Revision maximum instances | 1 |
| Runtime | Python 3.12 |
| Container image | Artifact Registry |

Cloud Run logs confirmed that instances were created automatically after periods of inactivity.

One observed startup sequence was:

| Event | Time (PDT) |
|---|---|
| Instance startup initiated | 15:05:15 |
| Uvicorn process started | 15:05:29 |
| Application startup completed | 15:05:29 |
| Application began serving requests | 15:05:29 |

The approximate interval between the Cloud Run instance-start event and application readiness was **14.4 seconds**.

A separate instance startup was also recorded at approximately 12:38 PDT, followed by application shutdown at approximately 12:53 PDT.

These logs confirm that CineScope can scale down and subsequently start a new instance.

However, the observed 14.4-second interval is an infrastructure/application startup measurement, not a complete cold-request response time.

The first Discovery request in the initial benchmark had:

- TTFT: 34.769 seconds
- Total latency: 36.147 seconds

This request is considered a potential cold-start candidate, but the available logs do not conclusively establish that it triggered a new Cloud Run instance.

Accordingly, a confirmed cold-request versus warm-request latency comparison remains a limitation of the current evaluation.

### 11.5 Performance Analysis

**Retrieval versus generation:** Local RAG generation took approximately three times as long as retrieval on average. This indicates that remote LLM generation was the larger contributor to standalone RAG latency.

**Agent-specific latency:** The Research workflow was faster than Discovery and News in the measured warm Cloud Run runs. Different agent workflows invoke different models and tools, so their end-to-end latencies are not directly interchangeable.

**First-run overhead:** The initial benchmark was substantially slower than subsequent runs. Possible contributors include instance initialization, connection setup, dependency initialization, remote API variability, and caching. The current measurements cannot isolate these factors.

**Streaming behavior:** CineScope streams the final synthesis response rather than the intermediate Research, Discovery, or News Agent output. Consequently, the observed TTFT includes substantial orchestration time before answer streaming begins.

**Scale-to-zero:** Cloud Run startup logs demonstrate that the application can initialize after an idle period. This supports cost-efficient deployment with zero minimum instances, at the expense of possible cold-start delays.

### 11.6 Benchmark Limitations

- Local RAG timings and Cloud Run timings were collected in different execution environments and should not be treated as equivalent.
- Only five local RAG questions and six warm Cloud Run requests were used.
- The initial benchmark was not conclusively identified as a cold-start run.
- Cloud Run TTFT measures the first streamed synthesis chunk, not the first token produced by any intermediate agent.
- Remote LLM and Tavily API latency can vary independently of application performance.
- Individual LangGraph node execution times were not separately instrumented in the deployed service.
- Provider token usage and per-query cost measurements are not included in this section; they will be reported separately.

### 11.7 Reproducibility

Run the local RAG benchmark:

```bash
PYTHONPATH=. python benchmarks/benchmark_rag.py
```

Run the Cloud Run benchmark:

```bash
python benchmarks/benchmark_cloud.py --mode warm --runs 3
```

Results are stored in `benchmarks/results/` as timestamped JSON files.

The following Cloud Run benchmark files correspond to the measurements reported above:

- `cloud_warm_20261021_221303.json`
- `cloud_warm_20261021_221429.json`
- `cloud_warm_20261021_221540.json`

All reported measurements are based on observed execution times rather than estimated values.

The OpenAI versus NVIDIA provider benchmark and cost analysis are documented in Section 12.

---

## 12. LLM Provider Performance and Cost Comparison

### 12.1 Methodology

CineScope uses two LLM providers: OpenAI GPT-4.1-mini and NVIDIA Nemotron 3.5 Lightning 30B A3B.

A controlled benchmark compared both models using three identical movie-related prompts covering factual retrieval, thematic analysis, and comparative synthesis.

Each model received the same system instructions and user questions. The benchmark measured request latency, reported input tokens, reported output tokens, and response availability.

The experiment used direct model API calls rather than the full LangGraph workflow, isolating model-level behavior from Chroma, SQLite, Tavily MCP, and orchestration overhead.

### 12.2 Benchmark Results

| Metric | OpenAI GPT-4.1-mini | NVIDIA Nemotron |
|---|---:|---:|
| Successful requests | 3/3 | 3/3 |
| Mean latency | 1.400s | 7.636s |
| Median latency | 1.383s | 6.867s |
| Mean input tokens | 64.0 | 71.3 |
| Mean output tokens | 85.7 | 1,072.0 |

#### Per-Prompt Latency

| Task | OpenAI | NVIDIA |
|---|---:|---:|
| Factual question | 1.383s | 3.618s |
| Thematic research | 1.322s | 6.867s |
| Comparative synthesis | 1.495s | 12.424s |

OpenAI was approximately 5.45 times faster on average in this small benchmark.

This finding is specific to the tested models, API endpoints, prompts, and execution conditions. It should not be generalized to all OpenAI or NVIDIA models.

### 12.3 NVIDIA Reasoning Token Analysis

NVIDIA Nemotron reported substantially more output tokens than OpenAI.

An additional API inspection confirmed that Nemotron includes internal reasoning tokens in its reported completion token usage.

For a short factual question, the NVIDIA API reported:

| Token category | Count |
|---|---:|
| Input tokens | 27 |
| Output tokens | 185 |
| Reasoning tokens | 174 |
| Non-reasoning output tokens | 11 |

Reasoning tokens represented approximately 94.1% of the reported output tokens in this example.

This helps explain why NVIDIA can report hundreds of output tokens for a short visible response.

The exact reasoning-token proportion was not measured for every benchmark question.

### 12.4 Cost Comparison

OpenAI GPT-4.1-mini standard API pricing was recorded as:

- Input: $0.40 per million tokens.
- Output: $1.60 per million tokens.

Using the benchmark's reported token counts:

| Prompt | Estimated OpenAI cost |
|---|---:|
| M01 | $0.0000476 |
| M02 | $0.0001996 |
| M03 | $0.0002408 |
| **Mean** | **$0.0001627** |

These estimates exclude caching discounts and other infrastructure costs.

NVIDIA Nemotron was accessed through NVIDIA Build's free prototype API endpoint. No direct NVIDIA API charge was incurred under the user's current access arrangement.

Consequently, the observed NVIDIA API access cost was $0.00, but no commercial per-token NVIDIA-hosted production price was established.

The comparison therefore distinguishes paid OpenAI token pricing from NVIDIA's free prototype access rather than claiming that NVIDIA inference has zero underlying computational cost.

### 12.5 Model Routing Implications

CineScope currently uses NVIDIA for selected straightforward tasks and OpenAI for more complex reasoning or structured-output tasks.

The benchmark demonstrates a tradeoff:

- OpenAI delivered lower response latency across the three tested prompts.
- NVIDIA provided free prototype API access but exhibited higher latency and substantially greater reported output-token usage.
- NVIDIA's reported output usage includes internal reasoning tokens, which should be considered when analyzing inference efficiency.

The results provide a quantitative basis for CineScope's provider-routing decisions, but do not establish that either provider is universally more accurate or cost-efficient.

### 12.6 Limitations

The benchmark used only three prompts per provider and one repetition per prompt.

No formal answer-quality scoring was performed for the provider comparison.

The experiment did not measure full multi-agent execution costs, Chroma embedding costs, Tavily usage costs, or Cloud Run infrastructure charges.

NVIDIA's free prototype access may have different rate limits, availability, and billing conditions from a production deployment.

Future evaluations could repeat each prompt, compare answer quality, and measure full-workflow cost per user request.

### 12.7 Reproducibility

Run the benchmark from the CineScope repository root:

```bash
PYTHONPATH=. python benchmarks/benchmark_models.py --runs 1
```

Baseline results:

`benchmarks/results/model_comparison_20261021_222318.json`

The results preserve the prompts, generated answers, reported token usage, and response latencies for inspection.
