import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Optimizador de Cartera MedTech", layout="wide")

st.title("🏥 Optimizador de Cartera MedTech")
st.subheader("Encuentra la cartera de inversión óptima entre las principales empresas de tecnología médica del mundo")

empresas = {
    "MDT": "Medtronic", "BSX": "Boston Scientific", "SYK": "Stryker",
    "ABT": "Abbott", "BDX": "Becton Dickinson", "ZBH": "Zimmer Biomet",
    "BAX": "Baxter", "COO": "Cooper Companies", "DGX": "Quest Diagnostics",
    "DXCM": "Dexcom", "EVH": "Evolent Health", "EW": "Edwards Lifesciences",
    "GMED": "Globus Medical", "HSIC": "Henry Schein", "IART": "Integra LifeSciences",
    "ICUI": "ICU Medical", "INGN": "Inogen", "INSP": "Inspire Medical",
    "IRTC": "iRhythm", "ISRG": "Intuitive Surgical", "LH": "Labcorp",
    "LIVN": "LivaNova", "LMAT": "LeMaitre Vascular", "LNTH": "Lantheus",
    "MMSI": "Merit Medical", "NEOG": "Neogen", "NTRA": "Natera",
    "NVCR": "NovoCure", "OFIX": "Orthofix", "OMCL": "Omnicell",
    "PHG": "Philips", "PODD": "Insulet", "RGEN": "Repligen",
    "RMD": "ResMed", "SIE.DE": "Siemens Healthineers", "STE": "Steris",
    "STIM": "Neuronetics", "TMDX": "TransMedics", "TNDM": "Tandem Diabetes",
    "ATEC": "Alphatec", "ATRC": "AtriCure", "ALGN": "Align Technology"
}

st.sidebar.header("⚙️ Configuración")

selected = st.sidebar.multiselect(
    "Selecciona las empresas a analizar:",
    options=list(empresas.keys()),
    format_func=lambda x: f"{x} — {empresas.get(x, x)}",
    default=["MDT", "BSX", "SYK", "ISRG", "DXCM", "EW", "RMD"]
)

years = st.sidebar.slider("Años de datos históricos:", 2, 10, 5)
num_portfolios = st.sidebar.select_slider(
    "Precisión de simulación:",
    options=[1000, 5000, 10000, 25000, 50000],
    value=10000
)

st.sidebar.markdown("---")
st.sidebar.markdown("**¿Qué es esto?**")
st.sidebar.markdown("Esta herramienta aplica la **Teoría de Carteras de Markowitz** para encontrar la combinación óptima de empresas MedTech que maximiza la rentabilidad minimizando el riesgo.")

if len(selected) < 3:
    st.warning("Selecciona al menos 3 empresas para ejecutar el análisis.")
else:
    if st.button("🚀 Calcular Cartera Óptima", type="primary"):
        with st.spinner("Descargando datos reales y ejecutando simulación..."):

            end = pd.Timestamp.today().strftime("%Y-%m-%d")
            start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
            data = yf.download(selected, start=start, end=end, progress=False)["Close"]
            data = data.ffill().dropna()

            returns = data.pct_change().dropna()
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252

            results = np.zeros((3, num_portfolios))
            weights_record = []

            for i in range(num_portfolios):
                w = np.random.random(len(selected))
                w /= np.sum(w)
                weights_record.append(w)
                ret = np.sum(mean_returns * w)
                risk = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                results[0,i] = ret
                results[1,i] = risk
                results[2,i] = ret / risk

            max_idx = np.argmax(results[2])
            optimal_w = weights_record[max_idx]

            st.success("✅ Análisis completado")

            col1, col2, col3 = st.columns(3)
            col1.metric("📈 Rentabilidad Anual Esperada", f"{results[0,max_idx]*100:.1f}%")
            col2.metric("⚡ Riesgo (Volatilidad)", f"{results[1,max_idx]*100:.1f}%")
            col3.metric("⭐ Ratio de Sharpe", f"{results[2,max_idx]:.2f}")

            st.subheader("Frontera Eficiente de Markowitz")
            st.caption("Cada punto representa una combinación diferente de inversión. La estrella roja es la cartera óptima.")
            fig, ax = plt.subplots(figsize=(12, 7))
            scatter = ax.scatter(results[1,:], results[0,:],
                               c=results[2,:], cmap="viridis",
                               alpha=0.4, s=8)
            plt.colorbar(scatter, ax=ax, label="Ratio de Sharpe")
            ax.scatter(results[1,max_idx], results[0,max_idx],
                      marker="*", color="red", s=800,
                      label="Cartera Óptima", zorder=5)
            ax.set_xlabel("Riesgo (Volatilidad Anual)")
            ax.set_ylabel("Rentabilidad Anual Esperada")
            ax.set_title("Frontera Eficiente — Empresas MedTech")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Distribución de la Cartera Óptima")
            df_result = pd.DataFrame({
                "Empresa": [empresas.get(t, t) for t in selected],
                "Ticker": selected,
                "Asignación (%)": (optimal_w * 100).round(2)
            }).sort_values("Asignación (%)", ascending=False)
            df_result = df_result[df_result["Asignación (%)"] > 0.5]
            st.dataframe(df_result, hide_index=True, use_container_width=True)

            st.subheader("Rendimiento Individual por Empresa")
            df_ranking = pd.DataFrame({
                "Empresa": [empresas.get(t, t) for t in selected],
                "Ticker": selected,
                "Rentabilidad Anual (%)": (mean_returns * 100).round(1),
                "Riesgo (%)": (np.sqrt(np.diag(cov_matrix)) * 100).round(1)
            }).sort_values("Rentabilidad Anual (%)", ascending=False)
            st.dataframe(df_ranking, hide_index=True, use_container_width=True)

st.markdown("---")
st.caption("Desarrollado por Nicolás Alfonso Escuredo · Ingeniería Biomédica y Emprendimiento · UFV Madrid · 2025")
