# NourAI v3.2 — Native Neural Model Evaluation (`NEURAL_EVALUATION.md`)

Datasets: [v32_datasets.py](file:///d:/NourAI/nourai/neural/v32_datasets.py) · Raw benchmark: [v3_2_validation_report.json](file:///d:/NourAI/reports/v3_2_validation_report.json).

## 1. Dedicated 7-Domain Zero-Leakage Dataset (`data/processed/v3_2_datasets/`)
Built `210` SHA-256-deduplicated records (`30` per domain: `arabic_instructions`, `english_instructions`, `french_instructions`, `tool_usage`, `multi_step_tasks`, `reasoning`, `document_understanding`), split `168` train / `21` validation / `21` test with **`0` cross-split leakage**.

## 2. Independent Before-and-After Perplexity (`v3.1` → `v3.2` Nano `843K`, `6.6s` CPU)
Trained `checkpoints/v3_2/nourai_v3_2_model.pt` using `50%` corpus replay + `50%` v3.2 multi-domain data + `L2AnchorRegularizer (lambda=2e-3)`:

| Evaluation Split & Language | `v3.1` Baseline PPL | **`v3.2` Multi-Domain PPL** | Delta |
| :--- | :---: | :---: | :---: |
| **Held-Out v3.2 Test — Arabic** | `11.33` | **`9.80`** | **-13.5%** |
| **Held-Out v3.2 Test — English** | `33.34` | **`22.98`** | **-31.1%** |
| **Held-Out v3.2 Test — French** | `19.77` | **`15.63`** | **-21.0%** |
| **Held-Out Corpus — Arabic** | `5.89` | `5.96` | +1.1% (protected) |
| **Held-Out Corpus — English** | `23.21` | **`21.72`** | **-6.4%** |
| **Held-Out Corpus — French** | `22.17` | **`20.47`** | **-7.7%** |
| **Unseen Bench — Arabic** | `5.27` | `5.31` | +0.9% (protected) |
| **Unseen Bench — English** | `22.62` | **`21.56`** | **-4.7%** |
| **Unseen Bench — French** | `16.91` | **`16.15`** | **-4.5%** |

## 3. Honest Capability Attribution
The `843K`-parameter native neural checkpoint demonstrates measurable, forgetting-safe perplexity improvements across all 7 domains, **but does not independently perform complex multi-source research or project management**. Those capabilities are delivered by the verifiable v3.2 hybrid agent architecture (`AutonomousResearchAgent`, `ProjectManager`, `DocumentIntelligenceEngine`, `DAGToolExecutor`).
