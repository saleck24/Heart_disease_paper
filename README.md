# Advancing Heart Disease Diagnosis and ECG Classification Using Deep Learning and Explainable AI

This repository contains the improved code, data processing scripts, and the LaTeX manuscript for the research paper extending the previous work on Heart Disease Diagnosis and ECG Classification.

## 📁 Structure of the Repository

- `paper_improved.tex`: The full LaTeX source code of the improved article, formatted for the Journal of Theoretical and Applied Information Technology (JATIT).
- `figures/`: Contains all the figures (1 to 16) generated and used in the paper.
- `src/`: Contains the source code for the deep learning models (CNN-BiLSTM-Attention, TabNet) and training routines.
- `main_notebook.ipynb`: Jupyter notebook containing the main experiments, training loops, and evaluation metrics.
- `prepare_mitdb.py`: Script to download and prepare the raw MIT-BIH Arrhythmia Database from PhysioNet using a Leave-One-Patient-Out (LOPO) strategy.
- `generate_figures.py`: Script used to generate all the analytical and architectural figures present in the paper. Proves the reproducibility of the visual results.

*Note: The `data/` folder is ignored via `.gitignore` to prevent uploading heavy datasets (e.g., 900MB+ for MIT-BIH) to GitHub. Please run `prepare_mitdb.py` to regenerate the data locally.*

## 🚀 Key Improvements

This version introduces several major improvements over the baseline study:
1.  **Methodological Rigor**: Implementation of Leave-One-Patient-Out Cross-Validation (LOPO-CV) and cross-dataset testing to prevent data leakage.
2.  **Advanced Deep Learning**: Replacement of classical ML baselines with physiology-aware deep learning architectures (CNN-BiLSTM-Attention for ECG, TabNet for tabular data).
3.  **Explainable AI (XAI)**: Integration of SHAP for tabular features, attention heatmaps for ECG waveforms, and Monte Carlo Dropout for uncertainty estimation, transforming the approach into clinical-grade AI.

## 🛠️ How to use

1.  **Data Preparation**: Run `python prepare_mitdb.py` to fetch and format the patient-level ECG data.
2.  **Experiments**: Open `main_notebook.ipynb` to explore the models, training, and evaluation.
3.  **Reproduce Figures**: Run `python generate_figures.py` to automatically recreate all 16 figures used in the scientific paper.
4.  **Paper Compilation**: The LaTeX paper `paper_improved.tex` is ready to be compiled using standard LaTeX engines (e.g., pdfLaTeX) or uploaded directly to Overleaf along with the `figures/` directory.
