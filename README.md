# 🌐 TruthLens AI — AI-Powered Global News Verification System

> *"Don't trust the headline. Verify the claim."*

**TruthLens AI** is an enterprise-grade, multi-lingual AI & Natural Language Processing (NLP) verification platform designed to detect, cross-examine, and fact-check news content, social media assertions, and viral claims.

By uniting **classical statistical NLP** with **fine-grained proposition entailment**, **live public multi-source corroboration**, and **source tier governance**, TruthLens AI bridges the critical gap between stylistic pattern recognition and real-world factual truth.

---

## 🏗️ System Architecture & Verification Flow

TruthLens AI operates a **6-Stage Hybrid AI & Evidence Verification Architecture**:

```mermaid
flowchart TD
    A[Input Modes: News Text / Web URL / Plain Claim / OCR Image] --> B[Content Normalization & Language Detection]
    B --> C[ML Stylistic Classifier: TF-IDF + Logistic Regression]
    B --> D[Proposition Extraction: Subject, Action, Object, Location, Date, Quantities]
    D --> E[Live Evidence Corroboration: Google News RSS & Wikipedia]
    E --> F[Claim-Level Entailment & Tier Governance: Tier 1-4 & Reference]
    C --> G[Deterministic Evidence Fusion Engine]
    F --> G
    G --> H[Multi-State Verdict: VERIFIED | MISLEADING | LIKELY FAKE | UNCERTAIN]
    H --> I[Plain-Language Explainability Dashboard & PDF Export]
```

---

## ⚡ Core Upgraded Capabilities

### 1. 🔍 Fine-Grained Proposition Entailment
Traditional systems rely on keyword search, producing critical false positives when sources share keywords but describe entirely different events.
- **Proposition Extraction:** Claims are decomposed into semantic tuples:
  $$\text{Claim} = \{\text{Subject}, \text{Actions}, \text{Objects}, \text{Locations}, \text{Quantities}, \text{Dates}\}$$
- **Conflict & Alignment Resolution:**
  - *Location Discrepancy:* A claim regarding `Mars` will **never** be corroborated by an article regarding the `Moon`, even if both mention `NASA`.
  - *Action Mismatch:* An article describing `corrosion of modules` will **never** corroborate a claim about `building a permanent city`.
  - *Entailment Stance:* Each external report is deterministically categorized into `SUPPORTING`, `CONTRADICTING`, `CONTEXTUAL`, or `IRRELEVANT`.

### 2. 🏛️ Standardized 4-Tier Source Governance
- **Tier 1 (Government & Peer-Reviewed Science):** Official agencies (`.gov`, `.gov.in`, ISRO, NASA, ESA, WHO) and scientific publishers (`Nature`, `Science`). Classified as *Authoritative Sources*.
- **Tier 2 (Established International News):** Major news bureaus with editorial oversight (`Reuters`, `Associated Press`, `BBC`, `The Hindu`, `AFP`).
- **Tier 3 (General Public Publishers):** Identifiable independent digital publications and web portals.
- **Tier 4 (Unverified / Self-Published):** Anonymous submissions, social media posts, and unverified direct inputs.
- **Reference Source (Wikipedia):** Classified strictly as a *Reference Source* — never treated as an authoritative primary source.

### 3. 🌐 Global Multi-Lingual Intelligence
Automatic language detection and normalization across **6 major world languages**:
- **English (EN)**
- **Hindi (HI)** (Devanagari script analysis)
- **Spanish (ES)**
- **French (FR)** (Elision & grammatical marker detection)
- **German (DE)** (Compound token analysis)
- **Arabic (AR)** (Arabic script analysis)

### 4. 🛡️ Enterprise Security & SSRF Protection
- **Server-Side Request Forgery (SSRF) Defense:** URL ingestion actively validates and rejects loopback, RFC 1918 private subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), AWS metadata services (`169.254.169.254`), and non-HTTP protocols.
- **Ephemeral Processing:** Text submissions and user queries are processed in memory without unauthorized persistent storage.

### 5. 🤖 Transparent Explainability ("Why this result?")
Every verification report provides a structured, plain-language 3-tier rationale:
1. **ML Linguistic Analysis:** Evaluates vocabulary tone, emotional keywords, and statistical classifier confidence. Explains that model confidence reflects linguistic patterns rather than physical reality.
2. **Live Evidence Corroboration:** Explicit count of supporting and refuting reports, publisher domains, and source tier rankings.
3. **Verdict Fusion Rationale:** Exact deterministic decision rule applied to reach the final verdict.

---

## 📊 Machine Learning Model Specifications

| Parameter | Details |
| :--- | :--- |
| **Algorithm** | Logistic Regression (`max_iter=1000`, $L_2$ regularization) |
| **Vectorization** | TF-IDF (Term Frequency - Inverse Document Frequency) |
| **Training Dataset** | 44,898 Full-Length Articles (23,481 Fake, 21,417 Real) |
| **Primary Domain** | Reuters News & International Political/Scientific Affairs |
| **Classifier Accuracy** | 98.63% |
| **Serialized Artifacts** | `fake_news_model.pkl` (1.6 MB), `vectorizer.pkl` (5.3 MB) |

### ⚠️ Model Confidence vs. Factual Truth
A foundational design principle in TruthLens AI:
- **Model Confidence** measures pattern similarity to the training set (e.g., formal journalistic vocabulary vs sensationalist phrasing).
- **Factual Truth** requires independent empirical corroboration.
- **The Core Rule:** When supporting evidence $= 0$ and contradicting evidence $= 0$, the final verdict **MUST BE `UNCERTAIN`**. A 99% ML confidence alone can **NEVER** produce `LIKELY FAKE` in the absence of evidence.

---

## 📁 Repository Structure

```text
Fake-News-Detection-System/
│
├── app.py                     # Main TruthLens AI Streamlit Application
├── login.py                   # Secure multi-user authentication portal
├── main.py                    # ML model training and evaluation script
├── fake_news_model.pkl        # Serialized Logistic Regression model
├── vectorizer.pkl             # Serialized TF-IDF vectorizer
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive architectural documentation
│
├── services/                  # Core verification services
│   ├── __init__.py
│   ├── content_extractor.py   # Safe URL scraping (SSRF protected) & normalization
│   ├── ocr_service.py         # Tesseract OCR with confidence calculation
│   ├── evidence_service.py    # Proposition extraction & semantic entailment
│   ├── ai_reasoning.py        # Structured 3-part plain-language explainability
│   └── verdict_engine.py      # Deterministic evidence fusion engine
│
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── text_cleaner.py        # Preprocessing, 6-language detection & sensationalism
│   ├── source_analyzer.py     # 4-tier source taxonomy & SSRF safety check
│   └── pdf_generator.py       # Publication-grade PDF dossier generator
│
└── tests/                     # Comprehensive test suites
    ├── test_contradiction_detection.py  # Proposition-level ranking, quantity & temporal contradictions
    ├── test_final_validation.py         # Mandatory 4-case verification suite
    ├── test_truthlens_suite.py          # Master TruthLens AI test suite
    ├── test_chandrayaan.py              # Dedicated Chandrayaan-3 corroboration suite
    ├── test_regression_propositions.py  # Mars vs Moon proposition entailment regression
    ├── test_core_cases.py               # Core operational scenarios
    ├── test_full_scenarios.py           # 10 comprehensive verification tests
    ├── test_pipeline.py                 # Full end-to-end integration test
    └── test_ui_simulation.py            # Streamlit UI simulation tests
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.10 to 3.14
- Tesseract OCR engine installed:
  - **Windows:** Download from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
  - **Ubuntu / Debian:** `sudo apt-get install tesseract-ocr`
  - **macOS:** `brew install tesseract`

### 2. Clone Repository
```bash
git clone https://github.com/vk980159-dot/Fake-News-Detection-System.git
cd Fake-News-Detection-System
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Test Suites
Verify all components, proposition entailment, contradiction detection, and regression checks:
```bash
# 1. Run Proposition-Level Contradiction Suite (Ranking, Quantity, Temporal, Modifiers)
python tests/test_contradiction_detection.py

# 2. Run Mandatory 4-Case Verification Suite
python tests/test_final_validation.py

# 3. Run Dedicated Chandrayaan-3 Corroboration Suite
python tests/test_chandrayaan.py

# 4. Run Proposition Entailment Regression Suite
python tests/test_regression_propositions.py

# 5. Run Master TruthLens AI Suite
python tests/test_truthlens_suite.py

# 6. Run Full End-to-End Pipeline & Scenarios Suite
python tests/test_pipeline.py
python tests/test_full_scenarios.py
python tests/test_ui_simulation.py
```

### 5. Launch TruthLens AI
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

*(Optional)* Launch authentication portal:
```bash
streamlit run login.py
```
*Default Demo Credentials:* `vivek@gmail.com` / `1234`

---

## ⚠️ Known Limitations & Edge Cases

1. **Live Public Evidence Dependence:**
   - Real-time fact verification relies on publicly accessible Google News RSS feeds and Wikipedia public search APIs.
   - If an event is entirely novel, hyperlocal, or offline, available evidence may be insufficient, strictly resulting in an `UNCERTAIN` verdict.
2. **OCR Image Quality:**
   - Optical Character Recognition accuracy depends on image resolution, contrast, font clarity, and local Tesseract installation. Degraded or handwritten images may yield low OCR confidence warnings.
3. **Optional Generative AI Synthesis:**
   - Deep reasoning synthesis via Google Gemini requires an optional `GEMINI_API_KEY`. If unconfigured, the system automatically and safely falls back to deterministic rule-based explainability without error.

---

## ⚖️ Responsible AI & Disclaimer

TruthLens AI provides probabilistic factual assessments combining statistical natural language processing with live corroborating evidence. This platform does not constitute absolute legal or journalistic proof. Critical decisions should always involve independent verification by certified fact-checkers and primary sources.

---

## 👨‍💻 Author & Credits

- **Developer:** Vivek Kumar
- **Project:** TruthLens AI — AI-Powered Global News Verification System
- **Core Stack:** Python, Streamlit, Scikit-Learn, Tesseract OCR, ReportLab, Google News RSS
