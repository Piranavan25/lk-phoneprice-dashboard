"""
LK-PhonePrice: Ensemble Learning & Hybridization Dashboard
Predicting Used & New Mobile Phone Prices in Sri Lanka (ikman.lk)
Using Bagging, Boosting, Stacking, and Hybrid AI (Random Forest + ANN)
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Plotly for interactive charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Import custom feature definitions and transformers
import utils

# Backward-compatibility patch for scikit-learn unpickling if needed
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="LK-PhonePrice | Ensemble Learning Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact, modern academic styling
st.markdown("""
<style>
    /* Metric Cards */
    .model-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.12);
        transition: transform 0.15s ease-in-out, border-color 0.15s ease-in-out;
    }
    .model-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .model-card-highlight {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid #818cf8;
        box-shadow: 0 0 15px rgba(129, 140, 248, 0.25);
    }
    .model-card-best-mae {
        background: linear-gradient(135deg, rgba(46, 16, 101, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid #c084fc;
        box-shadow: 0 0 15px rgba(192, 132, 252, 0.25);
    }
    .model-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .model-pred-price {
        font-size: 1.65rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 6px 0;
    }
    .model-stats-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: #94a3b8;
        padding-top: 6px;
        border-top: 1px solid rgba(148, 163, 184, 0.15);
    }
    .model-stats-item {
        text-align: center;
    }
    .model-stats-val {
        font-weight: 600;
        color: #e2e8f0;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-stacking { background-color: rgba(236, 72, 153, 0.2); color: #f472b6; border: 1px solid #db2777; }
    .badge-hybrid   { background-color: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid #9333ea; }
    .badge-boosting { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; }
    .badge-bagging  { background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #2563eb; }
    .badge-tag      { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }

    /* Summary Bar */
    .summary-banner {
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 10px;
        padding: 12px 18px;
        margin: 14px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Deployed Models Metadata & Metric Mapping
# ---------------------------------------------------------
# The 7 deployed models saved in models/phone_models.joblib
DEPLOYED_MODELS = [
    {
        "key": "Stacking ensemble",
        "display_name": "Stacking Ensemble",
        "csv_name": "Stacking (RF+XGB+LGBM+Cat -> Ridge)",
        "family": "Stacking",
        "badge_class": "badge-stacking",
        "icon": "🏆",
        "tag": "Selected (Lowest CV RMSE)",
        "card_class": "model-card-highlight",
        "color": "#ec4899"
    },
    {
        "key": "Hybrid RF + ANN (blend)",
        "display_name": "Hybrid RF + ANN Blend",
        "csv_name": "Hybrid: RF + ANN blend",
        "family": "Hybrid",
        "badge_class": "badge-hybrid",
        "icon": "🧬",
        "tag": "Lowest Test MAE",
        "card_class": "model-card-best-mae",
        "color": "#a855f7"
    },
    {
        "key": "Random Forest",
        "display_name": "Random Forest",
        "csv_name": "Random Forest (tuned)",
        "family": "Bagging",
        "badge_class": "badge-bagging",
        "icon": "🌲",
        "tag": "Tuned Bagging",
        "card_class": "",
        "color": "#3b82f6"
    },
    {
        "key": "LightGBM",
        "display_name": "LightGBM",
        "csv_name": "LightGBM (tuned)",
        "family": "Boosting",
        "badge_class": "badge-boosting",
        "icon": "💡",
        "tag": "Fast Boosting",
        "card_class": "",
        "color": "#10b981"
    },
    {
        "key": "XGBoost",
        "display_name": "XGBoost",
        "csv_name": "XGBoost (tuned)",
        "family": "Boosting",
        "badge_class": "badge-boosting",
        "icon": "⚡",
        "tag": "Hist Gradient Boosting",
        "card_class": "",
        "color": "#f59e0b"
    },
    {
        "key": "CatBoost",
        "display_name": "CatBoost",
        "csv_name": "CatBoost (tuned)",
        "family": "Boosting",
        "badge_class": "badge-boosting",
        "icon": "🐱",
        "tag": "Symmetric Trees",
        "card_class": "",
        "color": "#eab308"
    },
    {
        "key": "Hybrid RF -> ANN (residual)",
        "display_name": "Hybrid RF -> ANN Residual",
        "csv_name": "Hybrid: RF -> ANN residual",
        "family": "Hybrid",
        "badge_class": "badge-hybrid",
        "icon": "🔬",
        "tag": "Residual Correction",
        "card_class": "",
        "color": "#8b5cf6"
    }
]

# ---------------------------------------------------------
# Data & Model Loaders (Cached)
# ---------------------------------------------------------
@st.cache_data
def load_meta():
    meta_path = "reports/meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_results():
    results_path = "reports/model_results.csv"
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    return pd.DataFrame()

@st.cache_resource
def load_trained_models():
    """Loads all 7 fitted models from the Joblib file."""
    model_path = "models/phone_models.joblib"
    if os.path.exists(model_path):
        try:
            loaded = joblib.load(model_path)
            if isinstance(loaded, dict):
                return loaded, None
            return None, "Saved joblib file did not contain a model dictionary."
        except Exception as e:
            return None, str(e)
    return None, f"Model file not found at {model_path}."

meta = load_meta()
results_df = load_results()
models_dict, model_load_err = load_trained_models()

# Build evaluation metrics lookup from model_results.csv
def get_metrics_dict():
    lookup = {}
    if not results_df.empty:
        for _, row in results_df.iterrows():
            m_name = row["model"]
            lookup[m_name] = {
                "test_mae": float(row.get("test_mae", 0.0)),
                "test_rmse": float(row.get("test_rmse", 0.0)),
                "test_r2": float(row.get("test_r2_rs", row.get("test_r2_log", 0.0))),
                "cv_rmse": float(row.get("cv_rmse", 0.0)),
                "family": str(row.get("family", ""))
            }
    return lookup

metrics_lookup = get_metrics_dict()

# ---------------------------------------------------------
# Sidebar Navigation (Cleaned & Simplified to 3 Pages)
# ---------------------------------------------------------
st.sidebar.title("📱 LK-PhonePrice")
st.sidebar.caption("Ensemble & Hybrid Pricing Engine (Sri Lanka)")
st.sidebar.markdown("---")

nav_page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🔮 Live Price Predictor",
        "🏠 Executive Overview",
        "📊 Model Leaderboard & Comparison"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project Details:**
- **Data:** [ikman.lk](https://ikman.lk) (3,825 Cleaned Ads)
- **Deployed Models:** 7 Ensemble & Hybrid Models
- **Selected Model:** Stacking Ensemble (Lowest CV RMSE)
- **Lowest MAE:** Hybrid RF + ANN Blend (Rs. 15,650)
""")

# =========================================================
# PAGE 1: LIVE PRICE PREDICTOR (Main Interactive Page)
# =========================================================
if nav_page == "🔮 Live Price Predictor":
    st.subheader("🔮 Live Price Predictor: All-Model Prediction & Comparison")
    st.markdown("Enter phone specifications once below and click **Predict & Compare All Models** to evaluate across all 7 trained ensemble architectures simultaneously.")

    # Model status check
    if model_load_err:
        st.error(f"⚠️ Error loading models: {model_load_err}. Please ensure `models/phone_models.joblib` is present and compatible.")

    # -----------------------------------------------------
    # STEP 1: Enter Phone Details (Three Clear Sections)
    # -----------------------------------------------------
    with st.container():
        col_id, col_hw, col_attr = st.columns([1.1, 1.1, 1.1])

        # Section A: Device Identity
        with col_id:
            st.markdown("##### A. Device Identity")
            brand_models = meta.get("brand_models", {})
            brands_list = sorted(list(brand_models.keys())) if brand_models else ["Apple", "Samsung", "Xiaomi", "Google"]
            default_brand_idx = brands_list.index("Apple") if "Apple" in brands_list else 0
            sel_brand = st.selectbox("Brand", brands_list, index=default_brand_idx, key="input_brand")

            avail_models = brand_models.get(sel_brand, ["Other model"])
            default_model_idx = 0
            if "iPhone 15 Pro" in avail_models:
                default_model_idx = avail_models.index("iPhone 15 Pro")
            elif "Galaxy S24" in avail_models:
                default_model_idx = avail_models.index("Galaxy S24")
            sel_model = st.selectbox("Model", avail_models, index=default_model_idx, key="input_model")

            sel_condition = st.radio("Condition", ["Used", "Brand New"], horizontal=True, key="input_condition")

            brand_os_dict = meta.get("brand_os", {})
            auto_os = brand_os_dict.get(sel_brand, "Android")
            os_options = ["iOS", "Android", "Unknown"]
            default_os_idx = 0 if auto_os == "iOS" else (1 if auto_os == "Android" else 2)
            sel_os = st.selectbox("Operating System", os_options, index=default_os_idx, key="input_os")

        # Section B: Hardware Specifications
        with col_hw:
            st.markdown("##### B. Hardware Specifications")
            storage_opts = meta.get("storages", [16, 32, 64, 128, 256, 512, 1024])
            default_sto_idx = storage_opts.index(128) if 128 in storage_opts else 0
            sel_storage = st.selectbox("Storage Capacity (GB)", storage_opts, index=default_sto_idx, key="input_storage")

            ram_opts = meta.get("rams", [2, 3, 4, 6, 8, 12, 16])
            default_ram_idx = ram_opts.index(6) if 6 in ram_opts else (ram_opts.index(8) if 8 in ram_opts else 0)
            sel_ram = st.selectbox("RAM (GB)", ram_opts, index=default_ram_idx, key="input_ram")

            net_opts = meta.get("networks", ["4G", "5G", "VoLTE", "Unknown"])
            default_net_idx = net_opts.index("5G") if "5G" in net_opts else 0
            sel_network = st.selectbox("Network Connectivity", net_opts, index=default_net_idx, key="input_network")

        # Section C: Listing Attributes
        with col_attr:
            st.markdown("##### C. Listing Attributes")
            dist_opts = meta.get("districts", ["Colombo", "Gampaha", "Kandy", "Galle", "Kurunegala"])
            default_dist_idx = dist_opts.index("Colombo") if "Colombo" in dist_opts else 0
            sel_district = st.selectbox("District (Sri Lanka)", dist_opts, index=default_dist_idx, key="input_district")

            mem_opts = meta.get("membership", ["free", "plus", "premium"])
            sel_membership = st.selectbox("Seller Membership Level", mem_opts, index=0, key="input_membership")

            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                sel_images = st.number_input(
                    "Number of Photos",
                    min_value=1,
                    max_value=15,
                    value=int(meta.get("median_n_images", 2)),
                    key="input_images"
                )
            with col_sub2:
                sel_desc_len = st.number_input(
                    "Description Length",
                    min_value=10,
                    max_value=2500,
                    value=int(meta.get("median_desc_len", 553)),
                    step=50,
                    key="input_desc_len"
                )

    # -----------------------------------------------------
    # STEP 2: Predict Button
    # -----------------------------------------------------
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Predict & Compare All Models", type="primary", use_container_width=True)

    # Preserve prediction state so results stay visible
    if predict_clicked or "has_predicted" not in st.session_state:
        st.session_state["has_predicted"] = True

    # -----------------------------------------------------
    # STEP 3: Display ALL Model Predictions Together
    # -----------------------------------------------------
    if st.session_state.get("has_predicted", False):
        # Prepare feature payload using exact pipeline features from utils.py
        input_data = {
            "brand": sel_brand,
            "model": sel_model,
            "condition": sel_condition,
            "os": sel_os,
            "network": sel_network,
            "district": sel_district,
            "membership_level": sel_membership,
            "storage_gb": float(sel_storage),
            "ram_gb": float(sel_ram),
            "storage_missing": 0,
            "ram_missing": 0,
            "n_images": int(sel_images),
            "desc_len": int(sel_desc_len)
        }
        input_df = pd.DataFrame([input_data])[utils.FEATURES]

        # Execute predictions on all available deployed models
        predictions_results = []
        valid_predictions = {}

        if models_dict is not None:
            for item in DEPLOYED_MODELS:
                k = item["key"]
                metrics = metrics_lookup.get(item["csv_name"], {})
                test_mae = metrics.get("test_mae", 0.0)
                test_rmse = metrics.get("test_rmse", 0.0)
                test_r2 = metrics.get("test_r2", 0.0)

                if k in models_dict:
                    try:
                        model_obj = models_dict[k]
                        pred_log = model_obj.predict(input_df)[0]
                        pred_lkr = float(np.exp(pred_log))
                        pred_lkr = max(5000.0, pred_lkr)

                        predictions_results.append({
                            "key": k,
                            "display_name": item["display_name"],
                            "family": item["family"],
                            "badge_class": item["badge_class"],
                            "icon": item["icon"],
                            "tag": item["tag"],
                            "card_class": item["card_class"],
                            "color": item["color"],
                            "pred_lkr": pred_lkr,
                            "test_mae": test_mae,
                            "test_rmse": test_rmse,
                            "test_r2": test_r2,
                            "error": None
                        })
                        valid_predictions[item["display_name"]] = pred_lkr
                    except Exception as err:
                        predictions_results.append({
                            "key": k,
                            "display_name": item["display_name"],
                            "family": item["family"],
                            "badge_class": item["badge_class"],
                            "icon": item["icon"],
                            "tag": item["tag"],
                            "card_class": item["card_class"],
                            "color": item["color"],
                            "pred_lkr": None,
                            "test_mae": test_mae,
                            "test_rmse": test_rmse,
                            "test_r2": test_r2,
                            "error": str(err)
                        })
                else:
                    predictions_results.append({
                        "key": k,
                        "display_name": item["display_name"],
                        "family": item["family"],
                        "badge_class": item["badge_class"],
                        "icon": item["icon"],
                        "tag": item["tag"],
                        "card_class": item["card_class"],
                        "color": item["color"],
                        "pred_lkr": None,
                        "test_mae": test_mae,
                        "test_rmse": test_rmse,
                        "test_r2": test_r2,
                        "error": "Model key not found in loaded joblib"
                    })

        # Summary consensus bar
        if valid_predictions:
            vals = list(valid_predictions.values())
            avg_p = np.mean(vals)
            min_p = np.min(vals)
            max_p = np.max(vals)

            st.markdown(f"""
            <div class="summary-banner">
                <div>
                    <span style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">Target Phone:</span>
                    <span style="font-weight: 700; color: #f8fafc; font-size: 1.05rem; margin-left: 6px;">{sel_brand} {sel_model}</span>
                    <span style="color: #cbd5e1; font-size: 0.85rem; margin-left: 6px;">({sel_condition}, {sel_storage}GB, {sel_ram}GB RAM, {sel_district})</span>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.85rem;">Ensemble Consensus:</span>
                    <span style="font-size: 1.15rem; font-weight: 800; color: #38bdf8; margin-left: 6px;">Rs. {avg_p:,.0f} LKR</span>
                    <span style="color: #94a3b8; font-size: 0.8rem; margin-left: 10px;">(Spread: Rs. {min_p:,.0f} – Rs. {max_p:,.0f})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 1. Individual Model Prediction Cards")

        # Top Row: 4 Models (Stacking, Hybrid Blend, Random Forest, LightGBM)
        row1_cols = st.columns(4)
        for i, item in enumerate(predictions_results[:4]):
            with row1_cols[i]:
                card_cls = f"model-card {item['card_class']}".strip()
                tag_html = f"<span class='badge badge-tag'>{item['tag']}</span>" if item['tag'] else ""
                fam_badge = f"<span class='badge {item['badge_class']}'>{item['family']}</span>"

                if item["error"] is None:
                    price_str = f"Rs. {item['pred_lkr']:,.0f}"
                    mae_str = f"Rs. {item['test_mae']:,.0f}"
                    rmse_str = f"Rs. {item['test_rmse']:,.0f}"
                    r2_str = f"{item['test_r2']:.4f}"
                else:
                    price_str = "<span style='font-size: 1rem; color: #ef4444;'>Prediction Error</span>"
                    mae_str = "N/A"
                    rmse_str = "N/A"
                    r2_str = "N/A"

                st.markdown(f"""
                <div class="{card_cls}">
                    <div class="model-title">
                        <span>{item['icon']} {item['display_name']}</span>
                        {fam_badge}
                    </div>
                    <div style="margin-bottom: 4px;">{tag_html}</div>
                    <div class="model-pred-price">{price_str}</div>
                    <div class="model-stats-row">
                        <div class="model-stats-item">
                            <div>Test MAE</div>
                            <div class="model-stats-val">{mae_str}</div>
                        </div>
                        <div class="model-stats-item">
                            <div>Test RMSE</div>
                            <div class="model-stats-val">{rmse_str}</div>
                        </div>
                        <div class="model-stats-item">
                            <div>Test R²</div>
                            <div class="model-stats-val">{r2_str}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Bottom Row: 3 Models (XGBoost, CatBoost, Hybrid Residual)
        row2_cols = st.columns(3)
        for i, item in enumerate(predictions_results[4:7]):
            with row2_cols[i]:
                card_cls = f"model-card {item['card_class']}".strip()
                tag_html = f"<span class='badge badge-tag'>{item['tag']}</span>" if item['tag'] else ""
                fam_badge = f"<span class='badge {item['badge_class']}'>{item['family']}</span>"

                if item["error"] is None:
                    price_str = f"Rs. {item['pred_lkr']:,.0f}"
                    mae_str = f"Rs. {item['test_mae']:,.0f}"
                    rmse_str = f"Rs. {item['test_rmse']:,.0f}"
                    r2_str = f"{item['test_r2']:.4f}"
                else:
                    price_str = "<span style='font-size: 1rem; color: #ef4444;'>Prediction Error</span>"
                    mae_str = "N/A"
                    rmse_str = "N/A"
                    r2_str = "N/A"

                st.markdown(f"""
                <div class="{card_cls}">
                    <div class="model-title">
                        <span>{item['icon']} {item['display_name']}</span>
                        {fam_badge}
                    </div>
                    <div style="margin-bottom: 4px;">{tag_html}</div>
                    <div class="model-pred-price">{price_str}</div>
                    <div class="model-stats-row">
                        <div class="model-stats-item">
                            <div>Test MAE</div>
                            <div class="model-stats-val">{mae_str}</div>
                        </div>
                        <div class="model-stats-item">
                            <div>Test RMSE</div>
                            <div class="model-stats-val">{rmse_str}</div>
                        </div>
                        <div class="model-stats-item">
                            <div>Test R²</div>
                            <div class="model-stats-val">{r2_str}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # -------------------------------------------------
        # STEP 4 & 5: Comparison Table & Comparison Chart
        # -------------------------------------------------
        c_table, c_chart = st.columns([1.1, 1.2])

        with c_table:
            st.markdown("#### 2. Model Comparison Table")

            table_rows = []
            for item in predictions_results:
                table_rows.append({
                    "Model": item["display_name"],
                    "Family": item["family"],
                    "Predicted Price (LKR)": f"Rs. {item['pred_lkr']:,.0f}" if item["pred_lkr"] else "Error",
                    "Test MAE (LKR)": f"Rs. {item['test_mae']:,.0f}",
                    "Test RMSE (LKR)": f"Rs. {item['test_rmse']:,.0f}",
                    "Test R²": f"{item['test_r2']:.4f}",
                    "Status": "Selected by CV" if "Stacking" in item["display_name"] else ("Lowest Test MAE" if "Blend" in item["display_name"] else "Deployed")
                })

            df_comparison = pd.DataFrame(table_rows)
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)

            st.caption("""
            💡 **Evaluation Note:** The **Stacking Ensemble** was selected based on 5-fold cross-validation RMSE (Rs. 31,726).
            The **Hybrid RF + ANN Blend** achieved the lowest held-out Test MAE (Rs. 15,650), successfully bridging discrete tree splits with smooth neural manifold interpolation.
            """)

        with c_chart:
            st.markdown("#### 3. Model Prediction Comparison Chart")
            if HAS_PLOTLY and valid_predictions:
                chart_df = pd.DataFrame([
                    {
                        "Model": item["display_name"],
                        "Price": item["pred_lkr"],
                        "Family": item["family"],
                        "Color": item["color"]
                    }
                    for item in predictions_results if item["pred_lkr"] is not None
                ])

                fig = px.bar(
                    chart_df,
                    x="Model",
                    y="Price",
                    color="Family",
                    text="Price",
                    labels={"Price": "Predicted Price (LKR)", "Model": "Ensemble Model"},
                    color_discrete_map={
                        "Stacking": "#ec4899",
                        "Hybrid": "#a855f7",
                        "Bagging": "#3b82f6",
                        "Boosting": "#f59e0b"
                    }
                )
                fig.update_traces(
                    texttemplate='Rs. %{text:,.0f}',
                    textposition='outside',
                    marker_line_color='rgba(255,255,255,0.2)',
                    marker_line_width=1
                )
                fig.update_layout(
                    height=340,
                    margin=dict(l=15, r=15, t=25, b=15),
                    yaxis=dict(title="Price in LKR (Rs.)", tickprefix="Rs. "),
                    xaxis=dict(title=""),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            elif not valid_predictions:
                st.info("No predictions to display in chart.")

# =========================================================
# PAGE 2: EXECUTIVE OVERVIEW
# =========================================================
elif nav_page == "🏠 Executive Overview":
    st.subheader("🏠 Executive Overview: Project & Architecture Summary")

    st.markdown("""
    This project addresses the real-world machine learning challenge of predicting **used and brand-new mobile phone prices**
    in the Sri Lankan secondary marketplace ([ikman.lk](https://ikman.lk)).
    Because individual sellers establish subjective asking prices, standard linear regressions suffer from high variance,
    categorical sparsity, and strong non-linear interactions across hardware specifications and geographic districts.

    We benchmarked and deployed **12 machine learning architectures** across **5 model families**:
    **Bagging**, **Boosting**, **Stacking**, **Hybridization (Tree + Neural Network)**, and **Baselines**.
    """)

    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Cleaned Dataset", "3,825 Listings", "ikman.lk (Sri Lanka)")
    with kpi2:
        st.metric("Selected Model (CV)", "Stacking Ensemble", "5-Fold CV RMSE Rs. 31,726")
    with kpi3:
        st.metric("Lowest Test MAE", "Rs. 15,650", "Hybrid RF + ANN Blend")
    with kpi4:
        st.metric("Ensemble Test R²", "0.9215", "+8.4% over Ridge Baseline")

    st.markdown("---")

    col_sum1, col_sum2 = st.columns([1.2, 1])

    with col_sum1:
        st.markdown("""
        #### 📌 Key Findings & Inductive Bias Analysis
        1. **Stacking Ensemble Wins Overall Generalization:**
           - Stacking combines Random Forest (bagging), XGBoost (exact tree boosting), LightGBM (leaf-wise boosting), and CatBoost (symmetric oblivious trees) with a Ridge regression meta-learner.
           - Evaluated with strictly leak-free 5-Fold Out-of-Fold (OOF) predictions, it achieved the lowest **CV RMSE (Rs. 31,726)** and a **Test RMSE of Rs. 30,607** ($R^2 = 0.9215$).
        2. **Hybridization Solves Categorical vs Continuous Trade-offs:**
           - The **Random Forest + ANN weighted blend** achieved the lowest Mean Absolute Error (**Rs. 15,650**).
           - Random Forest handles discrete categorical splits (brand, model, district), while the Artificial Neural Network (MLP) provides continuous manifold smoothing across storage and RAM.
        3. **Failure of Linear Baselines:**
           - Ridge regression achieved an $R^2$ of only 0.8374 with an MAE of Rs. 24,072.
           - Storage doubling (128GB to 256GB) adds ~Rs. 45,000 on an Apple iPhone, but less than Rs. 8,000 on a budget phone. Linear models cannot capture this interaction without explicit polynomial expansion.
        """)

    with col_sum2:
        st.markdown("#### 🌲 Model Families Overview")
        st.markdown("""
        - **🌲 Bagging (Variance Reduction):**
          Parallel decision trees trained on bootstrap samples with random feature subsampling (*Random Forest, Extra Trees*).
        - **⚡ Boosting (Bias Reduction):**
          Sequential additive trees minimizing gradient pseudo-residuals (*XGBoost, LightGBM, CatBoost*).
        - **🏆 Stacking (Meta-Learning):**
          Meta-learner discovers optimal linear weighting of base estimators from out-of-fold predictions.
        - **🧬 Hybridization (Cross-Paradigm):**
          Integrates orthogonal tree inductive biases with deep neural representation learning (*RF + ANN*).
        """)

# =========================================================
# PAGE 3: MODEL LEADERBOARD & COMPARISON
# =========================================================
elif nav_page == "📊 Model Leaderboard & Comparison":
    st.subheader("📊 Rigorous Model Comparison & Benchmarks")
    st.markdown("""
    All models were evaluated under an **identical 5-Fold Cross-Validation protocol** on the training set
    (`KFold(5, shuffle=True, random_state=42)`) and evaluated on an **unseen holdout test set (20%)**.
    Metrics are computed directly in **Sri Lankan Rupees (LKR)**.
    """)

    if not results_df.empty:
        # Leaderboard Table
        display_df = results_df.copy()
        display_df["CV RMSE (5-Fold)"] = display_df.apply(lambda r: f"Rs. {r['cv_rmse']:,.0f} ± {r['cv_rmse_std']:,.0f}", axis=1)
        display_df["Test RMSE (LKR)"] = display_df["test_rmse"].apply(lambda v: f"Rs. {v:,.0f}")
        display_df["Test MAE (LKR)"] = display_df["test_mae"].apply(lambda v: f"Rs. {v:,.0f}")
        display_df["Test MAPE"] = display_df["test_mape"].apply(lambda v: f"{v:.1f}%")
        display_df["Test R²"] = display_df["test_r2_rs"].apply(lambda v: f"{v:.4f}")
        display_df["Train Time"] = display_df["seconds"].apply(lambda v: f"{v:.1f}s")

        cols_to_show = [
            "model", "family", "CV RMSE (5-Fold)", "Test RMSE (LKR)",
            "Test MAE (LKR)", "Test MAPE", "Test R²", "Train Time"
        ]
        rename_map = {"model": "Model Architecture", "family": "Family"}

        st.dataframe(
            display_df[cols_to_show].rename(columns=rename_map),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # Visualizations
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("##### 📉 Test RMSE by Model Architecture (Lower is Better)")
            if HAS_PLOTLY:
                fig_rmse = px.bar(
                    results_df.sort_values("test_rmse"),
                    x="test_rmse",
                    y="model",
                    orientation="h",
                    color="family",
                    labels={"test_rmse": "Test RMSE (LKR)", "model": "Architecture"},
                    color_discrete_map={
                        "Stacking": "#ec4899",
                        "Hybrid": "#a855f7",
                        "Boosting": "#f59e0b",
                        "Bagging": "#3b82f6",
                        "Baseline": "#64748b",
                        "Neural network": "#10b981"
                    }
                )
                fig_rmse.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_rmse, use_container_width=True)

        with col_c2:
            st.markdown("##### ⏱️ Test MAE vs Training Latency (Pareto Frontier)")
            if HAS_PLOTLY:
                fig_scatter = px.scatter(
                    results_df,
                    x="seconds",
                    y="test_mae",
                    color="family",
                    text="model",
                    size=[14]*len(results_df),
                    labels={"seconds": "Training Time (Seconds)", "test_mae": "Test MAE (LKR)"},
                    color_discrete_map={
                        "Stacking": "#ec4899",
                        "Hybrid": "#a855f7",
                        "Boosting": "#f59e0b",
                        "Bagging": "#3b82f6",
                        "Baseline": "#64748b",
                        "Neural network": "#10b981"
                    }
                )
                fig_scatter.update_traces(textposition="top center")
                fig_scatter.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
st.caption("LK-PhonePrice Ensemble Learning Dashboard | Machine Learning Assignment | Sri Lankan Used Market Context (ikman.lk)")
