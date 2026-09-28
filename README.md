```markdown
# ⚡ NEXUS // CareerPulse AI
### *Enterprise-Grade Student Placement Readiness & Compensation Band Engine*

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-00f2fe?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-a855f7?style=for-the-badge)](LICENSE)

---

## 📌 Executive Summary

**NEXUS CareerPulse AI** is an end-to-end predictive analytics platform designed to evaluate candidate placement readiness and project compensation tiers during institutional campus recruitment drives.

Standard classification algorithms tend to output extreme binary step probabilities (0.0 or 1.0) and overlook mandatory, non-negotiable enterprise placement criteria. NEXUS bridges this gap by integrating:
1. **Calibrated Ensemble Inference**: 150-tree Random Forest classifier regularized with leaf smoothing (`min_samples_leaf=15`) to output smooth, continuous probability estimates.
2. **Corporate Eligibility Logic**: Strict Zero-Active-Backlog enforcement and 60.0% secondary board aggregate thresholds (10th and 12th Grade).
3. **5-Pillar Modular Rubric**: Granular evaluation spanning Language ($L_x$), Aptitude ($A_x$), Core Engineering ($C_x$), Programming ($P_x$), and Soft Skills ($S_x$).
4. **Interactive High-Performance UI**: Dark-mode cybernetic dashboard featuring inline vector rendering (zero external font dependencies), candidate auto-population by USN/Name, and glassmorphic navigation routing.

---

## 🏗️ System Architecture

```text
                                 [ Candidate Record Input ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ USN Directory Lookup ]                     [ Manual Profile Entry ]
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                               [ Corporate Policy Filters ]
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             Active Backlogs > 0?                       10th or 12th Board < 60%?
              ├── YES ──► [ DISQUALIFIED (0.0%) ]        ├── YES ──► [ SEVERE RISK CRITICAL (4-12%) ]
              └── NO                                     └── NO
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                                 [ Scikit-Learn Pipeline ]
                                  ├── One-Hot Encoding
                                  ├── Robust Scaler
                                  └── Calibrated Random Forest
                                             │
                                             ▼
                                 [ Continuous Probability ]
                                 [   & CGPA Elasticity    ]
                                             │
                                             ▼
                                [ Tier Band Recommendation ]
                       ┌─────────────────────┼─────────────────────┐
                       ▼                     ▼                     ▼
                  Mass Recruiter       Mid-Tier Product       Tier-1 MNC
                  (4.0 - 5.0 LPA)      (8.0 - 14.0 LPA)    (14.0 - 20.0 LPA)

```

---

## 📊 5-Pillar Departmental Assessment Rubric

Candidates are evaluated across 5 modular clearance categories. To qualify for corporate placement drives, candidates must clear a minimum Level 3 threshold across all domains:

| Module Code | Skill Domain | Score Levels | Corporate Cutoff Benchmark |
| --- | --- | --- | --- |
| **$L_x$** | Language & Professional Fluency | $L_0 - L_4$ | Minimum $L_3$ clearance |
| **$A_x$** | Quantitative & Logical Aptitude | $L_0 - L_4$ | Minimum $L_3$ clearance |
| **$C_x$** | Core Engineering Fundamentals | $L_0, L_2 - L_5$ | Minimum $L_3$ clearance |
| **$P_x$** | Algorithmic Coding & Development | $L_{0.0} - L_{5.0}$ | Minimum $L_{3.0}$ clearance |
| **$S_x$** | Soft Skills & Executive Presence | $L_0 - L_4$ | Minimum $L_3$ clearance |

---

## 📈 Model Performance & Benchmarks

The model was evaluated against multiple supervised baselines on test partitions:

| Model Architecture | Precision | Recall | F1-Score | Status |
| --- | --- | --- | --- | --- |
| **Random Forest (Calibrated Ensemble)** | **100.0%** | **100.0%** | **1.0000** | **Production Deployed** |
| Gradient Boosting Classifier (GBM) | 100.0% | 100.0% | 1.0000 | Benchmarked |
| K-Nearest Neighbors ($k=5$) | 91.04% | 97.95% | 0.9437 | Baseline |
| Logistic Regression | 90.52% | 93.13% | 0.9181 | Baseline |

> **Data Integrity & Leakage Prevention**: Non-predictive identifiers (`student_id`, `student_name`, `usn`) and composite score targets (`Overall_Level_Score`) were strictly isolated and removed prior to model training to prevent target leakage.

---

## 📂 Project Repository Structure

```text
placement-prediction/
├── data/
│   └── student_placement_data.csv        # Integrated candidate records & metrics
├── models/
│   └── best_pipeline.pkl                 # Serialized Scikit-Learn pipeline
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb         # Feature analysis & distribution curves
│   └── 02_model_training.ipynb           # Model selection, tuning & serialization
├── app.py                                # Streamlit dashboard application
├── requirements.txt                      # Production runtime dependencies
├── .gitignore                            # Virtual environment & cache filters
└── README.md                             # Technical project documentation

```

---

## ⚙️ Business Rules & Diagnostic Engine

1. **Mandatory Zero-Backlog Policy**:
* If `backlogs > 0`, the candidate is immediately marked `DISQUALIFIED` with `0.0%` probability, regardless of test tier clearances or CGPA standing.


2. **60.0% Secondary Board Filter**:
* If either 10th or 12th board marks fall below `60.0%`, placement probability is constrained to `2.0% - 14.0%` due to automated enterprise applicant tracking system (ATS) filters.


3. **CGPA Elasticity**:
* High CGPA ($\ge 8.5$) unlocks Tier-1 product roles (14.0 – 20.0 LPA).
* Mid CGPA (7.0 – 7.4) scales placement probability downward to mirror real-world market competition.
* CGPA below 6.0 triggers first-class cutoff exclusion alerts.



---

## 🚀 Quickstart & Local Installation

### Prerequisites

* Python 3.10 or 3.11
* Git CLI

### 1. Clone the Repository

```bash
git clone [https://github.com/stutikatiyar/placement-prediction.git](https://github.com/stutikatiyar/placement-prediction.git)
cd placement-prediction

```

### 2. Set Up Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Runtime Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### 4. Run the Web Application

```bash
streamlit run app.py

```

Open your browser at `http://localhost:8501`.

---

## 👩‍💻 Author & Contact

* **Developers**:  Sujal Ranjan & Stuti Katiyar
* **Repository**: [https://github.com/stutikatiyar/placement-prediction](https://github.com/stutikatiyar/placement-prediction?utm_source=gemini)

```

```