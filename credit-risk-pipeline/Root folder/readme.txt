# Credit Default Risk Detection: Ensemble vs. Deep Learning Architectures

**Module:** MSIN0097 Predictive Analytics  
**Institution:** UCL School of Management  

## Project Overview
This repository contains an end-to-end predictive machine learning pipeline designed to identify retail credit card defaults. The project evaluates two distinct algorithmic architectures to address the asymmetric financial costs of credit misclassification: a supervised Random Forest ensemble and a semi-supervised deep learning Autoencoder. 

The primary business objective is to minimize direct capital loss resulting from missed defaults (False Negatives) while transparently managing the operational friction associated with falsely declining solvent customers (False Positives). Standard accuracy is discarded as an evaluation metric in favor of Recall and the F1-score.

## Repository Architecture
To ensure strict reproducibility, the repository is structured as follows:

* `data/`: Contains the data ingestion scripts or instructions for the UCI dataset. (Note: Data is fetched dynamically via API in the notebook).
* `notebooks/`: Contains the primary execution file `Predictive_analytics_indi_assignment.ipynb`.
* `reports/`: Contains the generated figures, confusion matrices, and the final 2,000-word business report.
* `requirements.txt`: Lists all Python dependencies required to execute the environment.
* `README.md`: Project documentation and execution instructions.

## Methodology and Pipeline Safety

### 1. Data Provenance
The pipeline utilizes the UCI Default of Credit Card Clients dataset (ID 350), comprising 30,000 observations of Taiwanese credit card holders across 23 continuous and categorical financial features.

### 2. Leakage Prevention
The dataset exhibits a severe 22 percent target class imbalance. To prevent data leakage during synthetic oversampling, the pipeline enforces a strict isolation protocol. The data is partitioned using a stratified 80/20 train/test split. Subsequently, standard scaling and the Synthetic Minority Over-sampling Technique (SMOTE) are encapsulated strictly within an `imblearn.pipeline.Pipeline` object. This guarantees that synthetic data generation occurs exclusively on the training folds during cross-validation.

### 3. Model Architectures
* **Baseline (Random Forest):** A tree-based ensemble instantiated with `class_weight='balanced'`. This model natively handles non-linear tabular data and neutralizes feature multicollinearity via random node subsampling. It serves as the highly interpretable, high-sensitivity operational baseline.
* **Modern Approach (Keras Autoencoder):** A semi-supervised dense neural network. The Autoencoder is trained exclusively on the 'No Default' training subset to learn the latent manifold of solvent financial behavior. Defaults are flagged as anomalies based on a 95th-percentile Mean Squared Error (MSE) reconstruction threshold.

## Reproducibility and Quickstart

To reproduce the results, models, and figures discussed in the final report, execute the following commands in your terminal:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/chandakmukund14-hue/credit-risk-pipeline.git]
   cd credit-risk-pipeline

Model Card Summary
Intended Use: Institutional risk management and macro-portfolio monitoring for retail credit facilities.

Out of Scope Use: Automated individual credit line assignment without human underwriter oversight. Not calibrated for macroeconomic stress testing.

Constraints: The models assume historical stationarity based on a six-month rolling window from 2005. The pipeline will exhibit temporal and geographical bias if applied to modern Western banking populations without complete retraining.

Evaluation Caveat: The operational Random Forest model is strategically optimized for Recall to protect institutional capital. Stakeholders must explicitly accept the resulting operational friction of a higher False Positive rate.