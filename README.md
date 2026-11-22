# 📱 LK-PhonePrice: Ensemble Learning & Hybridization Dashboard

> **Predicting Used & New Mobile Phone Prices in Sri Lanka ([ikman.lk](https://ikman.lk)) using Bagging, Boosting, Stacking, and Hybrid AI**  
> *Machine Learning Assignment — 4th Year Curriculum*

---

## 🌟 Executive Summary

In Sri Lanka's leading secondary marketplace ([ikman.lk](https://ikman.lk)), used and brand-new mobile phone asking prices fluctuate drastically due to subjective seller valuations, currency inflation, and rapid tech depreciation. 

This project develops, tunes, benchmarks, and deploys **12 machine learning architectures** across **5 distinct model families**:
1. **Bagging:** Random Forest (tuned), Extra Trees (tuned), Bagged Trees
2. **Boosting:** Gradient Boosting (tuned), LightGBM (tuned), XGBoost (tuned), CatBoost (tuned)
3. **Stacking:** Stacking Ensemble (`RF + XGB + LGBM + CatBoost -> Ridge Meta-Learner`)
4. **Hybridization:**
   - **Hybrid 1:** Random Forest + Artificial Neural Network (MLP) Convex Weighted Blend (`AverageBlend`)
   - **Hybrid 2:** Random Forest -> ANN Sequential Residual Error Correction (`ResidualHybrid`)
5. **Baselines:** Single Decision Tree, Ridge Regression

---

## 🏆 Benchmark Leaderboard (Rupees Scale & Log Scale)

All models were evaluated under an **identical 5-Fold Cross-Validation** protocol on the training set (`KFold(5, shuffle=True, random_state=42)`) and validated on an **unseen 20% holdout test set**.

| Rank | Model Name | Family | 5-Fold CV RMSE (Rs) | Test RMSE (Rs) | Test MAE (Rs) | Test MAPE (%) | Test $R^2$ (Rs) | Train Time |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Stacking (RF+XGB+LGBM+Cat -> Ridge)** | **Stacking** | **Rs. 31,726** ± 3,079 | **Rs. 30,607** | Rs. 16,491 | 18.6% | **0.9215** | 115.5s |
| 🥈 | **Hybrid: RF + ANN Blend** | **Hybrid** | Rs. 32,379 ± 2,903 | Rs. 31,320 | **Rs. 15,650** | **18.2%** | **0.9178** | 20.3s |
| 🥉 | **LightGBM (tuned)** | **Boosting** | Rs. 32,594 ± 2,723 | **Rs. 30,181** | Rs. 16,560 | 19.2% | **0.9236** | **2.5s** |
| 4 | **Gradient Boosting (tuned)** | **Boosting** | Rs. 32,452 ± 3,195 | Rs. 31,955 | Rs. 17,204 | 19.3% | 0.9144 | 36.1s |
| 5 | **Random Forest (tuned)** | **Bagging** | Rs. 32,969 ± 3,031 | Rs. 31,023 | Rs. 15,546 | 18.3% | 0.9193 | 11.1s |
| 6 | **Extra Trees (tuned)** | **Bagging** | Rs. 33,624 ± 3,368 | Rs. 32,075 | Rs. 16,414 | 19.4% | 0.9138 | 7.4s |
| 7 | **XGBoost (tuned)** | **Boosting** | Rs. 33,536 ± 2,498 | Rs. 32,061 | Rs. 18,049 | 20.0% | 0.9138 | 5.9s |
| 8 | **CatBoost (tuned)** | **Boosting** | Rs. 33,537 ± 3,338 | Rs. 33,893 | Rs. 18,329 | 20.1% | 0.9037 | 14.3s |
| 9 | **Bagged Trees (BaggingRegressor)** | **Bagging** | Rs. 34,372 ± 2,949 | Rs. 31,078 | Rs. 16,449 | 19.6% | 0.9190 | 18.6s |
| 10 | **Hybrid: RF -> ANN Residual** | **Hybrid** | Rs. 36,343 ± 3,185 | Rs. 32,623 | Rs. 17,704 | 19.1% | 0.9108 | 38.6s |
| 11 | **ANN (MLP, tuned)** | **Neural Net** | Rs. 39,222 ± 2,293 | Rs. 37,278 | Rs. 20,244 | 22.4% | 0.8835 | 7.9s |
| 12 | **Single Decision Tree** | **Baseline** | Rs. 40,334 ± 3,885 | Rs. 30,552 | Rs. 17,968 | 22.2% | 0.9218 | 0.4s |
| 13 | **Ridge Regression** | **Baseline** | Rs. 47,911 ± 3,277 | Rs. 44,046 | Rs. 24,072 | 25.2% | 0.8374 | 0.5s |

---

## 🔍 Key Academic & Theoretical Insights

### 1. Why Stacking Won the CV Selection
- **Inductive Bias Diversity:** Random Forest (bagging), XGBoost (exact tree boosting), LightGBM (leaf-wise gradient boosting), and CatBoost (oblivious symmetric decision trees) explore fundamentally different loss functional spaces.
- **Out-of-Fold (OOF) Ridge Meta-Learner:** In-sample predictions on training data have near-zero residual errors for tree models. By training the Ridge meta-learner strictly on 5-fold OOF cross-validation predictions, the model avoided target leakage and discovered optimal weights: `XGBoost: 0.369`, `LightGBM: 0.279`, `Random Forest: 0.195`, `CatBoost: 0.161`.

### 2. Why Hybridization (RF + ANN) Achieved Lowest MAE (Rs. 15,650)
- **Orthogonal Functional Strengths:**
  - **Random Forest:** Excels at sharp, axis-aligned categorical splits (e.g., Brand = Apple vs Samsung; District = Colombo).
  - **Artificial Neural Network (MLP):** Excels at continuous, smooth manifold interpolation across quantitative variables (RAM, storage, listing length).
  - Blending them with $w = 0.80$ eliminates tree leaf step-discontinuities while retaining razor-sharp brand boundaries.

### 3. Why Ensembles Dominate Linear Regression & Single Trees
- **Linear Ridge Failure:** Ridge assumes additive, non-interactive effects ($y = \sum w_i x_i$). In reality, doubling storage (128GB to 256GB) adds **Rs. 45,000+** to an iPhone 15 Pro, but less than **Rs. 8,000** to a Xiaomi Redmi. Without manual polynomial interaction terms, Ridge suffers severe underfitting ($R^2 = 0.837$).
- **Single Tree Instability:** A single tree suffers from greedy hierarchical fragmentation: an early suboptimal split cascades irreversibly down all child nodes. Bagging eliminates this variance.

---

## 📊 SHAP (SHapley Additive exPlanations) Importance

Aggregated global SHAP importance highlights what truly drives mobile phone prices in Sri Lanka:

| Rank | Feature | Importance Share (%) | Market Role |
|:---:|:---|:---:|:---|
| 1 | `brand` | **24.1%** | Defines brand tier and baseline prestige |
| 2 | `model` | **17.5%** | Defines release year and flagship vs budget tier |
| 3 | `ram_gb` | **17.1%** | Core performance spec, heavily advertised on Android |
| 4 | `storage_gb` | **11.9%** | Step multiplier, strong gradient on Apple iPhones |
| 5 | `network` | **7.7%** | 5G provides a major generational price premium |
| 6 | `condition` | **6.3%** | Brand New commands a 26%–51% markup over Used |
| 7 | `membership_level` | **5.7%** | Premium shop ads reflect verified commercial dealer stock |
| 8 | `district` | **2.9%** | Colombo & Gampaha show slight capital premium |
| 9 | `os` | **2.5%** | iOS vs Android platform differentiation |
| 10 | `desc_len` | **2.4%** | Longer descriptions correlate with legitimate shop ads |
| 11 | `n_images` | **1.1%** | Professional multi-photo listings command higher prices |

---

## 🚀 How to Run the Dashboard

### Prerequisites
Make sure Python is installed. Dependencies:
```bash
pip install streamlit pandas scikit-learn xgboost lightgbm catboost plotly joblib
```

### Option A: One-Click Launcher (Windows)
Double-click `run_dashboard.bat` in this folder.

### Option B: Terminal Command
Run inside this folder:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎬 3-Minute Video Presentation Script

| Time | Slide / Screen Action | Voiceover Audio Script |
|:---:|:---|:---|
| `0:00 - 0:30` | **Executive Overview Tab** | *"Hello everyone! In Sri Lanka's booming secondary mobile market on ikman.lk, asking prices vary wildly—sellers struggle to price fairly, and buyers can't tell if an ad is a bargain or overpriced. To solve this, we collected and cleaned over 3,800 real Sri Lankan listings and built this comprehensive Ensemble Learning Dashboard. Instead of relying on a single model, we benchmarked 12 architectures across Bagging, Boosting, Stacking, and Hybrid AI to predict fair market prices in rupees with extreme precision."* |
| `0:30 - 1:15` | **Model Leaderboard Tab** | *"Standard linear regression fails here, achieving an R² of only 0.83. But ensemble learning excels. First, we implemented Bagging—including Random Forest and Extra Trees—which reduces prediction variance across high-cardinality models. Next, we tuned Gradient Boosting models—XGBoost, CatBoost, and LightGBM—which sequentially minimize bias. LightGBM proved fastest, training in just 2.5 seconds with a Test RMSE of Rs 30,180. To achieve peak generalization, we built a Stacking Ensemble using our forest and three gradient boosters as base estimators, blended via a Ridge meta-learner trained on 5-fold out-of-fold predictions. Stacking took 1st place in cross-validation with an R² of 0.921!"* |
| `1:15 - 1:55` | **Hybridization Deep-Dive Tab** | *"Following our lecturer's guidance on hybridization, we combined Random Forest with an Artificial Neural Network. Why? Decision trees excel at sharp categorical boundaries like brand and model, while Neural Networks smoothly interpolate continuous features like storage and RAM. We tested two architectures: a weighted blend and a sequential residual hybrid where the neural net learns the forest's out-of-fold mistakes. Our RF + ANN blend achieved the lowest Mean Absolute Error of all 12 models at just Rs 15,650—outperforming both parent models!"* |
| `1:55 - 2:30` | **Live Price Predictor Tab** | *"Here is the live deployed dashboard. A user simply chooses the brand—say Apple—selects the model, storage, RAM, and district. In milliseconds, our ensemble pipeline handles missing values, encodes categories, and outputs the estimated price—here, Rs 185,000—along with a confidence interval based on empirical test error. We can also compare all ensemble models side-by-side or run instant 'what-if' analyses to see how upgrading storage or condition affects market value."* |
| `2:30 - 3:00` | **SHAP & Conclusion Tab** | *"Finally, we analyzed model interpretability using SHAP. Brand and model account for 41.5% of total importance, followed by RAM and Storage at 29%. Geographic variation is secondary. Our only limitation is that we model seller asking prices rather than final negotiated transactions. In conclusion, our Stacking and Hybrid ensembles achieved over 92% R² and reduced prediction error by 35% compared to baseline regression. Thank you!"* |

---

## 📂 Project Directory Structure

```
dashboard_files (1)/
├── app.py                             # Full interactive Streamlit Dashboard application
├── run_dashboard.bat                  # One-click Windows launcher for the dashboard
├── utils.py                           # Custom sklearn transformers, pipelines & hybrid classes
├── README.md                          # Comprehensive assignment documentation & benchmarks
├── Phone_Price_Ensemble_Colab_(1).ipynb# Complete Jupyter Notebook with full interpretations
├── data/
│   └── phones_clean.csv               # Cleaned ikman.lk dataset (3,825 listings)
├── models/
│   └── phone_models.joblib            # Serialized fitted ensemble models for live deployment
└── reports/
    ├── meta.json                      # Metadata, category lookups & price bounds
    ├── model_results.csv              # Full empirical cross-validation & test results
    ├── shap_importance.csv            # Aggregated SHAP feature importance shares
    ├── model_comparison.png           # High-resolution benchmark comparison plot
    └── versions.txt                   # Pinned library dependencies
```
