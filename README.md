# Asset Pricing via Machine Learning (MSc Dissertation)

This repository contains the full implementation of my MSc dissertation project at King’s College London.  
The project focuses on **high-frequency asset pricing** using machine learning and advanced factor models.

---

## 📌 Project Overview

The aim of this dissertation is to explore **cross-sectional asset pricing** in the Chinese stock market by leveraging modern machine learning techniques and high-dimensional factor models.  

Key highlights:
- **Data scale**:  
  - **CSI 300 constituents**, covering **2005–2025//2022-2025**  
  - **5-minute frequency intraday data**: ~8 million observations  
  - **Daily frequency data**: ~120,000 observations  
  - This large-scale panel dataset is substantially bigger than most prior studies in the asset pricing literature, ensuring robust empirical evaluation and realistic testing environments.  

- **Methodological contribution**:  
  - Incorporates **Instrumented PCA (IPCA)** as proposed by Kelly et al. (2019), using the **original IPCA code implementation** to extract interpretable latent factors.  
  - Benchmarks traditional and modern methods:  
    - **Lasso regression** (sparse linear models)  
    - **XGBoost** (nonlinear tree-based ensemble)  
    - **Deep Neural Networks (DNN)**  
    - **IPCA** (economically interpretable latent factor model)  

---

## 📂 Repository Structure

```

├── 818_PROJECT2.0/ # Early project version
├── 818_PROJECT3.0/ # Refined project version
├── 818 5min.py # Core high-frequency pipeline (CSI 300, 5-min data)
├── 818 dnn+blc.py # Deep Neural Network with baseline comparison
├── 818-data.py # Data preprocessing and factor construction
├── ipca_runner.py # Integration of original IPCA implementation
└── pycache/ # Cache files
```
---

## ⚙️ Features

- **Data preprocessing**: batch merging of raw stock CSV files + index series.  
- **Factor construction**: technical and microstructure features standardized via cross-sectional Z-scores.  
- **Target variable**: next-period excess return = stock return – market return.  
- **Rolling training/testing**: ensures realistic out-of-sample evaluation.  
- **Evaluation metrics**: $R^2$, MSE, IC, plus visualizations (distribution plots, factor importance, learning curves).  

---

## 📖 References

- Kelly, B., Pruitt, S., & Su, Y. (2019). *Characteristics are covariances: A unified model of risk and return*. Journal of Financial Economics.  
- Gu, S., Kelly, B., & Xiu, D. (2018). *Empirical asset pricing via machine learning*. The Review of Financial Studies.  
- Feng, G., He, J., & Polson, N. (2018). *Deep learning in asset pricing*.  

---

## 🚀 Usage

1. Prepare raw CSI 300 constituent data (daily or 5-min frequency).  
2. Run preprocessing scripts (`818-data.py`).  
3. Choose and run model pipeline:  
   - `818 5min.py` (main experiment)  
   - `818 dnn+blc.py` (deep learning comparison)  
   - `ipca_runner.py` (IPCA factor extraction)  
4. Results (JSON + CSV + plots) will be saved in the output folder.  

---

## 📌 Notes

- This project emphasizes **scalability with large datasets** and **factor interpretability through IPCA**.  
- The workflow is modular, allowing adaptation to other markets (e.g., Hong Kong, Japan, Korea, India).  
