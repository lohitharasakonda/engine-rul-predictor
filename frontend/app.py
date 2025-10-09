import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
DEMO_DIR = os.environ.get("DEMO_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "demo"))

st.set_page_config(page_title="Engine Health Monitor", layout="wide")
st.title("Aircraft Engine Health Monitor")

# sidebar with demo files
with st.sidebar:
    st.header("demo data")
    demos = [
        ("healthy engine", "healthy_engine.csv"),
        ("warning engine", "warning_engine.csv"),
        ("critical engine", "critical_engine.csv"),
    ]
    for label, fname in demos:
        fpath = os.path.join(DEMO_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                st.download_button(label, f, file_name=fname)

    st.divider()
    st.write("random forest, 300 trees")
    st.write("test RMSE: 19.70 cycles")

uploaded_file = st.file_uploader("upload engine sensor CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    with st.expander("data preview"):
        st.dataframe(df.head(10))
        st.caption(f"{len(df)} rows")

    if st.button("predict RUL", type="primary"):
        with st.spinner("running prediction..."):
            uploaded_file.seek(0)
            files = {"file": ("data.csv", uploaded_file, "text/csv")}
            try:
                resp = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=60)
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("cant connect to backend, is it running?")
                st.stop()
            except Exception as e:
                st.error(f"prediction failed: {e}")
                st.stop()

        latest = result["latest"]
        rul = latest["rul"]
        health = latest["health_score"]
        status = latest["status"]

        if status == "Healthy":
            st.success(f"status: {status}")
        elif status == "Warning":
            st.warning(f"status: {status} - schedule maintenance")
        else:
            st.error(f"status: {status} - immediate maintenance needed!")

        col1, col2, col3 = st.columns(3)
        col1.metric("predicted RUL", f"{rul} cycles")
        col2.metric("health score", f"{health}%")
        col3.metric("rows analyzed", result["num_rows"])

        # health gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health,
            number={"suffix": "%"},
            title={"text": "engine health"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2c3e50"},
                "steps": [
                    {"range": [0, 30], "color": "#e74c3c"},
                    {"range": [30, 60], "color": "#f39c12"},
                    {"range": [60, 100], "color": "#2ecc71"},
                ],
            },
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # RUL trend
        result_df = pd.DataFrame({
            "cycle": result["cycles"],
            "predicted_rul": result["predicted_rul"],
            "health_score": result["health_scores"],
        })

        fig_rul = go.Figure()
        fig_rul.add_trace(go.Scatter(
            x=result_df["cycle"], y=result_df["predicted_rul"],
            mode="lines", line=dict(color="#3498db", width=2),
            name="predicted RUL",
        ))
        fig_rul.update_layout(
            title="RUL over cycles",
            xaxis_title="cycle",
            yaxis_title="RUL",
            height=400,
        )
        st.plotly_chart(fig_rul, use_container_width=True)

        # health trend
        fig_health = go.Figure()
        fig_health.add_trace(go.Scatter(
            x=result_df["cycle"], y=result_df["health_score"],
            fill="tozeroy", mode="lines",
            line={"color": "#2ecc71"},
        ))
        fig_health.add_hline(y=60, line_dash="dash", line_color="#f39c12",
                             annotation_text="warning (60%)")
        fig_health.add_hline(y=30, line_dash="dash", line_color="#e74c3c",
                             annotation_text="critical (30%)")
        fig_health.update_layout(
            title="health score over cycles",
            height=400,
            yaxis_range=[0, 105],
        )
        st.plotly_chart(fig_health, use_container_width=True)

        # sensor trends if raw data
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        if sensor_cols and "cycle" in df.columns:
            st.subheader("sensor trends")
            selected = st.multiselect("pick sensors", sensor_cols, default=sensor_cols[:4])
            if selected:
                fig_sensors = px.line(df, x="cycle", y=selected)
                st.plotly_chart(fig_sensors, use_container_width=True)

        with st.expander("full prediction table"):
            st.dataframe(result_df)

else:
    st.info("upload a CSV to get started, or download demo data from the sidebar")