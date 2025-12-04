# Depression-Detection-Using-Psychomotor-Dynamics
# 💡 Depression Detection using Psychomotor Dynamics and Bagging SVM

## Overview
This project presents a novel, objective system for depression screening by analyzing **unconscious psychomotor dynamics** (head and eye movements) extracted from video interviews. Our method bypasses the limitations of subjective self-reporting by treating physical movement patterns as quantifiable **objective biomarkers**.

The core innovation is the **Temporal Splitting (Fatigue Hypothesis)**, which measures the decline in behavioral energy during the course of the interview. Using a robust Bagging Support Vector Machine (SVM) ensemble, the model achieves **90.0% accuracy** and a high **93% Sensitivity** (Recall) on the DAIC-WOZ dataset.

## 🌉 Research Gap & Innovation

| Aspect | Conventional Methods | Our Innovation |
| :--- | :--- | :--- |
| **Biomarker Focus** | Subjective Self-Report (PHQ-8) or basic gaze features. | **Objective Psychomotor Dynamics** (unconscious head/gaze movement). |
| **Key Discovery** | Failed to capture temporal decline in energy. | **Temporal Splitting (Fatigue Hypothesis):** Measures the diagnostic drop in motor energy from the video start to end. |
| **Core Features** | Not designed for clinical correlation. | **Mathematically defined features** (e.g., Head Idle Ratio/'Statue Metric' for Rigidity, Spectral Entropy for Monotony). |

## 🛠️ Methodology Pipeline

The system follows a three-phase pipeline:

1.  **Input & Pre-processing:** Video input (Gaze and Head Pose) from the DAIC-WOZ dataset is filtered using a Confidence Filter to ensure data quality.
2.  **Feature Engineering (Rigidity Engineering):** The filtered signals are used to calculate eight elite psychomotor features:
    * **Temporal Splitting:** Captures Fatigue (e.g., `end_head_power_low`).
    * **Time-Domain:** Captures Rigidity (`glo_head_idle_ratio`) and Sluggishness (`glo_head_mean_vel`).
    * **Frequency-Domain:** Captures Monotony (using Welch's Method for Spectral Entropy).
3.  **Classification:** The features are scaled using **RobustScaler** and classified using a **Bagging SVM Ensemble** (50 independent models) for enhanced stability and prediction robustness.

## 📊 Results and Performance

The model's performance on the balanced DAIC-WOZ dataset ($N=60$) confirms its efficacy as a screening tool.

| Metric | Score | Insight |
| :--- | :--- | :--- |
| **Accuracy** | 90.00% | High overall reliability for depression screening. |
| **Precision** | 93.00% | When 'Depressed' is predicted, it is 93% likely correct (low false alarm rate). |
| **Recall (Sensitivity)** | **87.00%** | **Highly effective at finding truly depressed patients (minimizes False Negatives).** |
| **F1-Score** | 0.90 | Strong, balanced performance across both healthy and depressed classes. |

*Note: The model achieved **87% Recall** for the Depressed class and **93% Recall** for the Healthy class, averaging to a strong $\text{macro avg recall}$ of $90\%$ (as per your Classification Report).*

## 🛑 Limitations and Future Work

| Challenge | Future Solution (Multi-Modal Fusion) |
| :--- | :--- |
| **Small Dataset ($N=60$)** | Validate on much larger, more diverse datasets. |
| **No Severity Score** | Transition model to predict the **continuous PHQ-8 severity score** (regression). |
| **'Smiling Depression'** | Integrate **Audio** (speech prosody) and **Text** (sentiment analysis) to detect masked symptoms. |

## 🚀 Getting Started

### Prerequisites

* Python 3.x
* (List required libraries like `numpy`, `pandas`, `scipy`, `scikit-learn`, etc.)

### Installation and Run

```bash
# Clone the repository
git clone [YOUR_REPO_URL]
cd [PROJECT_DIRECTORY]

# Install dependencies
pip install -r requirements.txt

# Run the full feature extraction and training pipeline
python main_runner.py
