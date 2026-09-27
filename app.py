import json
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

from src.data_loader import load_data, quality_audit
from src.data_cleaning import clean_data
from src.feature_engineering import add_features
from src.scenario_simulation import simulate_product
from src.recommendation_engine import recommend
from config.factories import FACTORIES

st.set_page_config(page_title="Factory Reallocation & Shipping Optimization",
                   page_icon="🍫", layout="wide")

DATA_PATH = Path("data/raw/dataset.csv")
MODEL_PATH = Path("models/shipping_model.pkl")

@st.cache_data
def get_data():
    return add_features(clean_data(load_data(DATA_PATH)))

@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)

st.title("Factory Reallocation & Shipping Optimization")
st.caption("Nassau Candy Distributor — predictive analytics, what-if simulation and decision support")

if not DATA_PATH.exists():
    st.error("Dataset not found. Put the CSV at data/raw/dataset.csv.")
    st.stop()

df = get_data()
audit = quality_audit(df)

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", [
        "Executive Overview", "Data Quality", "Exploratory Analysis",
        "Factory Optimization Simulator", "What-If Scenario",
        "Recommendation Dashboard", "Model Performance", "Factory Map"
    ])

if page == "Executive Overview":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Orders", df["Order ID"].nunique())
    c2.metric("Sales", f"${df['Sales'].sum():,.2f}")
    c3.metric("Gross Profit", f"${df['Gross Profit'].sum():,.2f}")
    c4.metric("Median Lead Time", f"{df['Shipping Lead Time'].median():,.0f} days")
    st.warning(
        "The supplied dates produce lead times of roughly 904–1,642 days. "
        "Review the date interpretation before treating these as real operational shipping durations."
    )
    fig = px.histogram(df, x="Shipping Lead Time", nbins=40, title="Shipping Lead Time Distribution")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Data Quality":
    st.subheader("Data Quality Audit")
    st.json(audit)
    st.dataframe(df[df["Data Quality Flag"] != ""].head(500), use_container_width=True)

elif page == "Exploratory Analysis":
    st.subheader("Exploratory Analysis")
    metric = st.selectbox("Metric", ["Sales","Gross Profit","Units","Shipping Lead Time"])
    group = st.selectbox("Group by", ["Product Name","Region","Ship Mode","Division","Current Factory"])
    agg = df.groupby(group)[metric].mean().reset_index().sort_values(metric, ascending=False)
    fig = px.bar(agg, x=group, y=metric, title=f"Average {metric} by {group}")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(df, x="Sales", y="Gross Profit", size="Units",
                      color="Division", hover_name="Product Name",
                      title="Sales vs Gross Profit")
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Factory Optimization Simulator":
    st.subheader("Factory Optimization Simulator")
    product = st.selectbox("Product", sorted(df["Product Name"].dropna().unique()))
    if MODEL_PATH.exists():
        model = get_model()
        sim = simulate_product(product, df, model)
        sim = recommend(sim)
        st.dataframe(sim, use_container_width=True)
        fig = px.bar(sim, x="Scenario Factory", y="Predicted Lead Time",
                     color="Risk", title=f"Predicted Lead Time — {product}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train the model first with: python train_model.py")

elif page == "What-If Scenario":
    st.subheader("What-If Scenario Analysis")
    product = st.selectbox("Product", sorted(df["Product Name"].dropna().unique()))
    factory = st.selectbox("Alternative Factory", list(FACTORIES.keys()))
    units = st.number_input("Units", min_value=1.0, value=10.0)
    speed = st.slider("Optimization priority: Speed → Profit", 0, 100, 50)
    st.info(f"Scenario: {product} → {factory}, Units = {units:,.0f}, Speed weight = {speed}%")
    st.write("Scenario financial values require configurable shipping/handling assumptions.")
    if MODEL_PATH.exists():
        model = get_model()
        sim = simulate_product(product, df, model)
        chosen = sim[sim["Scenario Factory"] == factory].copy()
        current = sim[sim["Scenario Factory"] == sim["Current Factory"].iloc[0]].copy()
        if not chosen.empty and not current.empty:
            st.metric("Predicted Lead Time", f"{chosen['Predicted Lead Time'].iloc[0]:.1f} days")
            st.metric("Lead Time Reduction",
                      f"{chosen['Lead Time Reduction %'].iloc[0]:.1f}%")

elif page == "Recommendation Dashboard":
    st.subheader("Recommendation Dashboard")
    if not MODEL_PATH.exists():
        st.warning("Run python train_model.py first.")
    else:
        model = get_model()
        product = st.selectbox("Product filter", ["All"] + sorted(df["Product Name"].unique()))
        speed = st.slider("Speed priority", 0, 100, 50)
        products = sorted(df["Product Name"].unique()) if product == "All" else [product]
        all_results = []
        for p in products:
            sim = simulate_product(p, df, model)
            if not sim.empty:
                all_results.append(recommend(sim, speed/100, 1-speed/100))
        if all_results:
            rec = pd.concat(all_results, ignore_index=True)
            st.dataframe(rec, use_container_width=True)
            st.download_button("Download recommendations CSV",
                               rec.to_csv(index=False).encode("utf-8"),
                               "factory_recommendations.csv", "text/csv")

elif page == "Model Performance":
    st.subheader("Model Performance")
    path = Path("models/model_comparison.csv")
    if path.exists():
        metrics = pd.read_csv(path)
        st.dataframe(metrics, use_container_width=True)
        fig = px.bar(metrics, x="Model", y="RMSE", title="RMSE Comparison")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run python train_model.py first.")

elif page == "Factory Map":
    st.subheader("Factory Locations")
    rows = [{"Factory": k, "Latitude": v["latitude"], "Longitude": v["longitude"]}
            for k,v in FACTORIES.items()]
    fmap = pd.DataFrame(rows)
    fig = px.scatter_geo(fmap, lat="Latitude", lon="Longitude", text="Factory",
                         scope="usa", title="Nassau Candy Factory Locations")
    st.plotly_chart(fig, use_container_width=True)
