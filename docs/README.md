# Documentation index

Start here:

| Document | What it answers |
|---|---|
| [`design.md`](design.md) | What the platform guarantees, what it refuses to do, and why orchestration is hand-written |
| [`architecture.md`](architecture.md) | Components, state machine and failure model as built |
| [`data-contract.md`](data-contract.md) | Canonical sample contract, quality policy shape, dataset manifest |
| [`implementation-status.md`](implementation-status.md) | Per-milestone DONE / PARTIAL / BLOCKED / NOT_RUN with evidence |

Operating it:

| Document | What it answers |
|---|---|
| [`ops-runbook.md`](ops-runbook.md) | Recovery operations: stuck tasks, expired leases, object GC |
| [`benchmark.md`](benchmark.md) | How the performance numbers are produced and how to re-run them |
| [`linux-demo-plan.md`](linux-demo-plan.md) | One-shot Linux + GPU demo checklist |
| [`evidence/`](evidence/) | Measured evidence that ships with the repo |

Decisions and provenance:

| Document | What it answers |
|---|---|
| [`ADR/`](ADR/) | Binding decisions and documented deviations (start at 0001, 0007, 0010) |
| [`licenses.md`](licenses.md) | Third-party licences and the obligations that follow |
| [`model-card.md`](model-card.md) | YOLO11n usage, weights SHA-256, what it is and is not used for |
| [`public-dataset-provenance.md`](public-dataset-provenance.md) | Licence and revision of any public dataset used for demonstration |
| [`test-report.md`](test-report.md) | Acceptance-test evidence for the V1 milestone |
| [`api-openapi.json`](api-openapi.json), [`api-openapi-v2.json`](api-openapi-v2.json) | Generated OpenAPI 3.1 documents |

Not shipped, by design: the internal specification the implementation was written against,
job-search notes, and the raw experiment archive. Evidence behind the numbers in
`benchmark.md` is summarised in [`evidence/`](evidence/); anything not measured on a
reproducible host is not claimed.
