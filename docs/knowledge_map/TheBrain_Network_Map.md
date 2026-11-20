---
title: "NourAI — تصور شبكة المعرفة بأسلوب TheBrain وObsidian Graph"
tags:
  - nourai
  - thebrain
  - knowledge-graph
  - obsidian-graph
parent: "[[NourAI_Knowledge_Map]]"
---

# 🕸️ تصور شبكة المعرفة بأسلوب TheBrain (`Parent · Child · Jump`)

في فلسفة **TheBrain**، لا تُحبس الفكرة داخل مجلد واحد؛ بل تقع كل فكرة (`Thought`) في مركز شبكة ديناميكية لها ثلاثة أنواع من الروابط:
- **⬆️ Parent (الأصل):** المفهوم الأعلى الذي تنبثق منه الفكرة.
- **⬇️ Child (الفروع):** المكونات التفصيلية التابعة لها.
- **↔️ Jump (الروابط الترابطية الجانبية):** علاقات تأثير، اعتماد، معالجة، أو تكامل مع أفكار من فروع أخرى تماماً.

---

## 1. الخريطة الشبكية الشاملة (Interactive Mermaid Knowledge Graph)

```mermaid
graph TB
    %% الأنماط اللونية
    classDef core fill:#1E40AF,stroke:#1E3A8A,color:#FFFFFF,stroke-width:2px
    classDef goal fill:#6D28D9,stroke:#5B21B6,color:#FFFFFF
    classDef project fill:#0284C7,stroke:#0369A1,color:#FFFFFF
    classDef gap fill:#DC2626,stroke:#991B1B,color:#FFFFFF,stroke-dasharray: 5 5
    classDef idea fill:#D97706,stroke:#B45309,color:#FFFFFF
    classDef security fill:#0F766E,stroke:#115E59,color:#FFFFFF

    CENTER["🧠 NourAI v1.2.0<br/>Local-First AR·EN·FR"]:::core

    %% الأهداف
    G1["🎯 Local-First Sovereignty"]:::goal
    G2["🎯 Multilingual Parity (AR/EN/FR)"]:::goal
    G3["🎯 Engineering Honesty"]:::goal

    %% المحاور الأساسية
    M1["🧠 Neural Model<br/>Pre-LN Transformer"]:::core
    M2["⚙️ 8-Layer Cognitive Engine"]:::core
    M3["📚 Data & Corpus<br/>SHA-256 & 0-Leakage"]:::project
    M4["🚀 Training & SFT Pipeline"]:::project
    M5["⚡ Hybrid Inference<br/>Neural + Fallback"]:::project
    M6["🌐 API & RTL/LTR Web UI"]:::project
    M7["🛡️ Security & Production<br/>TLS 1.3 · WAL · Rollback"]:::security
    M8["📊 Evaluation & QA<br/>24 Phases + Pytest"]:::project

    %% الثغرات المكتشفة
    GAP1["⚠️ Nano 843K Coherency<br/>& AR Perplexity Regression"]:::gap
    GAP2["⚠️ Arabic Byte-Level<br/>Context Tax (128B)"]:::gap
    GAP3["⚠️ Small SFT (72 Train)<br/>& CPU-Only Bottleneck"]:::gap

    %% الابتكارات والحلول
    IDEA1["💡 Byte-Level BPE<br/>(8K–32K Vocab)"]:::idea
    IDEA2["💡 Cloud GPU Scaling<br/>(Base 25.6M + AMP)"]:::idea
    IDEA3["💡 Hybrid Local RAG<br/>+ INT8/KV-Cache"]:::idea

    %% علاقات هرمية (Parent -> Child)
    CENTER --> G1 & G2 & G3
    CENTER --> M1 & M2 & M3 & M4 & M5 & M6 & M7 & M8

    %% علاقات ترابطية شبكية (TheBrain Jump Links)
    M3 -.->|"يغذي بالتوكنات المتوازنة"| M4
    M4 -.->|"ينتج أوزان"| M1
    M1 -.->|"يشغّل الاستدلال"| M5
    M2 -.->|"ينسّق RAG والذاكرة والبحث"| M5
    M5 -.->|"يخدم الواجهة بوضوح المصدر"| M6
    M7 -.->|"يحمي ويُشغّل"| M6
    M7 -.->|"يوثّق ويسترجع"| M3
    M8 -.->|"يدقق ويقيس"| M1 & M3 & M4

    %% روابط الثغرات والابتكارات
    M1 -.->|"يعاني من"| GAP1
    M1 -.->|"قيّد السياق بـ"| GAP2
    M4 -.->|"مقيّد بـ"| GAP3
    M8 -.->|"اكتشف"| GAP1

    IDEA1 ==>|"يعالج"| GAP2
    IDEA2 ==>|"يعالج"| GAP1 & GAP3
    IDEA3 ==>|"يعزّز"| M2 & M5
```

---

## 2. بطاقات الأفكار المحورية بأسلوب TheBrain (Thought Plex Cards)

### 🧠 Thought 1: `[[Pre-LN Decoder-Only Transformer]]`
- **⬆️ Parent:** `[[NourAI v1.2.0]]`
- **⬇️ Children:** `[[ByteTokenizer]]`, `[[Model Presets (Nano/Small/Base)]]`, `[[Weight Tying]]`, `[[Checkpoint Format]]`
- **↔️ Jumps:**
  - `← trained_by` `[[Training Engine]]`
  - `→ served_by` `[[Inference Engine]]`
  - `← audited_by` `[[Multilingual Evaluation]]`
  - `⚠️ constrained_by` `[[Gap: Arabic Byte-Level Context Tax]]`, `[[Gap: Nano Model Coherency]]`
  - `💡 upgraded_by` `[[Byte-Level BPE Multilingual Tokenizer]]`, `[[INT8/INT4 CPU Quantization & KV-Cache]]`

### ⚙️ Thought 2: `[[8-Layer Cognitive Architecture]]`
- **⬆️ Parent:** `[[NourAI v1.2.0]]`
- **⬇️ Children:** `[[FoundationModelEngine]]`, `[[ReasoningEngine]]`, `[[MemoryEngine]]`, `[[KnowledgeEngine]]`, `[[CodingEngine]]`, `[[EvaluationEngine]]`, `[[AgentEngine]]`, `[[NourAIPlatform]]`
- **↔️ Jumps:**
  - `→ uses_storage` `[[Local Memory & Document RAG]]`
  - `→ uses_tool` `[[Live Web Search]]`
  - `← protected_by` `[[Security Architecture]]` (AST Safe Math + SSRF Guard + Zip Bomb Limit)
  - `💡 evolved_by` `[[Graph-Aware Long-Term Memory]]`, `[[Hybrid Local RAG (BM25 + Tiny Embeddings)]]`

### 📚 Thought 3: `[[Data & Corpus Engineering]]`
- **⬆️ Parent:** `[[NourAI v1.2.0]]`
- **⬇️ Children:** `[[Corpus Acquisition & Adapters]]`, `[[Data Provenance & Leakage Audit]]`, `[[Balanced Corpus v1.2]]`, `[[SFT Dataset v1.2]]`
- **↔️ Jumps:**
  - `→ guarantees_goal` `[[Multilingual Parity AR-EN-FR]]`
  - `→ feeds` `[[Pretraining & Response-Masked SFT]]`
  - `⚠️ gap` `[[Gap: Small SFT Dataset Size]]`
  - `← verified_by` `[[Test Suite & Phase Audits]]`

### ⚡ Thought 4: `[[Inference & Hybrid Capabilities]]`
- **⬆️ Parent:** `[[NourAI v1.2.0]]`
- **⬇️ Children:** `[[Inference Engine]]`, `[[Fallback Assistant]]`, `[[Live Web Search]]`, `[[Local Memory & Document RAG]]`
- **↔️ Jumps:**
  - `→ enforces_goal` `[[Engineering Honesty & Telemetry]]`
  - `→ powers` `[[API Backend]]` & `[[Web UI & Canvas]]`
  - `⚠️ gap` `[[Gap: Arabic PDF Parsing]]`

### 🛡️ Thought 5: `[[Security, Production & DevOps]]`
- **⬆️ Parent:** `[[NourAI v1.2.0]]`
- **⬇️ Children:** `[[Security Architecture]]`, `[[Production Deployment]]`, `[[Release Lifecycle]]`
- **↔️ Jumps:**
  - `→ enforces_goal` `[[Local-First Sovereignty]]`, `[[Production Readiness & Reproducibility]]`
  - `→ wraps` `[[API Backend]]`, `[[Live Web Search]]`, `[[Local Memory & Document RAG]]`
  - `⚠️ gap` `[[Gap: Outdated API Reference Doc]]`

### 🤖 Thought 6 (v3.0.0): `[[Autonomous Agent Core v3.0]]`
- **⬆️ Parent:** `[[NourAI v3.0.0]]`
- **⬇️ Children:** `[[PerceptionEngine]]`, `[[ReasoningEngine]]`, `[[PlanningEngine]]`, `[[TaskManager]]`, `[[CriticEngine]]`, `[[RecoveryEngine]]`, `[[AutonomousAgentOrchestrator]]`
- **↔️ Jumps:**
  - `→ uses_memory` `[[Four-Tier Advanced Memory]]` & `[[Integrated Knowledge Graph]]`
  - `→ invokes` `[[Standardized Tool Engine]]` (9 sandboxed tools, least-privilege + approval gate)
  - `→ verified_by` `[[V3 Evaluation & Regression Gate]]`
  - `→ exposed_via` `[[V3APIRouter (/api/v3/*)]]`

### ⚡ Thought 7 (v3.1.0): `[[NourAI v3.1 Real-World Autonomy & Control Center]]`
- **⬆️ Parent:** `[[NourAI v3.1.0]]`
- **⬇️ Children:** `[[Neural Forensic Audit & v3.1 Checkpoint]]`, `[[DAG Parallel Tool Orchestrator]]`, `[[105-Task Real-World Agent Benchmark]]`, `[[Self-Evaluation Engine]]`, `[[NourAI Control Center Dashboard]]`
- **↔️ Jumps:**
  - `→ resolves` `[[Gap: Broken v1.2.0 Small Checkpoint Perplexity]]`
  - `→ upgrades` `[[Four-Tier Advanced Memory]]` (AR/FR normalization + preference superseding + stopword filtering)
  - `→ monitors` `[[Autonomous Agent Core v3.0]]` & `[[Standardized Tool Engine]]`

### 🔬 Thought 8 (v3.2.0): `[[NourAI v3.2 Research, Projects & Multimodal Intelligence]]`
- **⬆️ Parent:** `[[NourAI v3.2.0]]`
- **⬇️ Children:** `[[Autonomous Research Agent]]`, `[[Persistent Long-Term Project Manager]]`, `[[Advanced Reasoning Engine v3.2]]`, `[[Multimodal Document Intelligence]]`, `[[Control Center 2.0]]`, `[[205-Task Real-World Benchmark]]`
- **↔️ Jumps:**
  - `→ retrieves_citations_from` `[[Multimodal Document Intelligence]]`, `[[Four-Tier Advanced Memory]]`, `[[Integrated Knowledge Graph]]`
  - `→ persists_across_restarts` `[[SQLite WAL Persistence]]`
  - `→ monitored_by` `[[Self-Evaluation Engine]]` (9 KPIs, zero auto-unverified training)

---

## 3. خريطة تدفق القيمة والبيانات (End-to-End Dependency Flow)

```mermaid
flowchart LR
    RAW["مصادر خام<br/>QNL · WTO · Tatoeba"] --> PROV["تدقيق SHA-256<br/>وتراخيص ومنع تسرب"]
    PROV --> BAL["مدونة متوازنة AR/EN/FR<br/>+ بيانات SFT"]
    BAL --> TRAIN["محرك التدريب<br/>CPU محلي / GPU سحابي"]
    TRAIN --> CKPT["نقطة حفظ موثّقة<br/>Release v1.2.0"]
    CKPT --> INF["محرك الاستدلال<br/>+ 8 طبقات معرفية"]
    INF --> API["خادم REST + TLS 1.3<br/>PBKDF2 · Rate Limit"]
    API --> UI["واجهة RTL/LTR + Canvas<br/>+ CLI"]
    UI -.->|"تغذية راجعة وتقييم"| EVAL["تقييم متعدد اللغات<br/>Loss · PPL · BPB"]
    EVAL -.->|"تحسين مستمر Phase 24"| BAL
```
