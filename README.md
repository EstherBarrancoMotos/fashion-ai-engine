# Fashion AI Engine

> A Machine Learning decision engine that helps fashion brands reduce overproduction, returns, and unsold inventory through demand forecasting, return prediction, and scenario simulation.

---

## 🎯 The Problem

The fashion industry destroys or heavily discounts an estimated **30% of its production** every year due to:

- Inaccurate demand forecasts
- High return rates (15–40% in online fashion, depending on category)
- Poor assortment decisions
- Lack of insight into what styles, colors, and categories actually perform

This translates into billions in lost margin and a massive sustainability problem.

## 💡 The Solution

**Fashion AI Engine** is a modular ML system that acts as a decision engine for fashion brands. Given historical sales and product metadata, it:

1. **Predicts demand** per product (regression).
2. **Predicts return probability** per product (classification).
3. **Analyzes style intelligence** — which colors, categories, and price points perform best or worst.
4. **Simulates business decisions** — what would happen to margin, inventory, and returns if we cut production of risky items?

The output is not just a model — it's a **Streamlit app** that any merchandiser could use to make better decisions, with quantified business impact.

---

## 📊 Business Impact (Target Metrics)

The project is evaluated not only on technical metrics but on **business outcomes** measured via simulation on a temporal holdout:

- % reduction in excess inventory
- % reduction in returns
- Estimated € margin uplift
- Sensitivity analysis on key economic assumptions

Economic assumptions (gross margin, return cost, markdown rate) are sourced from public industry reports (H&M and Inditex annual reports, Narvar, Optoro) — never invented.

---

## 🧱 Architecture

```
┌─────────────────────┐
│   H&M dataset       │
│   (Kaggle, real)    │
└──────────┬──────────┘
           │
   ┌───────┴────────┐
   │ Feature Store  │
   └───────┬────────┘
           │
   ┌───────┼─────────────────────────────┐
   ▼       ▼              ▼              ▼
┌──────┐ ┌──────┐  ┌──────────────┐ ┌──────────┐
│Demand│ │Return│  │    Style     │ │ Customer │
│Model │ │Model │  │ Intelligence │ │ Behavior │
└───┬──┘ └───┬──┘  └──────┬───────┘ └─────┬────┘
    │       │             │               │
    └───────┴──────┬──────┴───────────────┘
                   ▼
         ┌──────────────────┐
         │ Decision         │
         │ Simulator        │  ← business logic + economic model
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ Streamlit App    │
         │ (SaaS-style UI)  │
         └──────────────────┘
```

---

## 📁 Project Structure

```
fashion-ai-engine/
├── data/                # raw, interim, processed, external (gitignored)
├── notebooks/           # exploratory and modelling notebooks
├── src/                 # production code
│   ├── data/            # loaders & splits
│   ├── features/        # feature engineering
│   ├── models/          # demand, returns, style
│   ├── simulator/       # economic model + decision policies
│   ├── evaluation/      # technical & business metrics
│   └── utils/
├── app/                 # Streamlit app (multi-page)
├── tests/               # unit tests
├── reports/             # figures & final results
├── configs/             # YAML configs
└── mlruns/              # MLflow tracking
```

---

## 🚀 Quickstart

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd fashion-ai-engine

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the H&M dataset (requires Kaggle API token)
#    Accept competition rules at:
#    https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules
mkdir -p data/raw && cd data/raw
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f articles.csv
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f customers.csv
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f transactions_train.csv
unzip "*.zip" && rm *.zip
cd ../..

# 5. Open the first notebook
jupyter notebook notebooks/00_first_look.ipynb

# 6. Run the Streamlit app (once models are trained)
streamlit run app/streamlit_app.py
```

---

## 🗺 Roadmap

- [x] Project setup and structure
- [ ] Exploratory data analysis (EDA)
- [ ] Demand prediction baseline (LightGBM)
- [ ] Return prediction model
- [ ] Style intelligence module
- [ ] Decision simulator (core of the project)
- [ ] Streamlit MVP
- [ ] Business impact analysis & sensitivity
- [ ] Final demo video

---

## 📚 Data Sources

- **H&M Personalized Fashion Recommendations** — Kaggle competition dataset (real transactions, 2018–2020).
- **Industry benchmarks** — return rates by category sourced from publicly available reports (Narvar, Optoro, Statista).

---

## 🛠 Tech Stack

- **Modelling:** scikit-learn, LightGBM, CatBoost
- **Experiment tracking:** MLflow
- **App:** Streamlit
- **Data:** pandas, polars (for large transactions file)
- **Visualization:** Plotly, matplotlib, seaborn

---

## 📝 License

This project is for educational and portfolio purposes. The H&M dataset is subject to its own competition terms.

---

## 👤 Author

Built by [your name] as a portfolio project demonstrating end-to-end ML for business impact.
Connect on [LinkedIn](https://linkedin.com/in/yourprofile).
