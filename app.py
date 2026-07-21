import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="MedTech Portfolio Optimizer", layout="wide")

st.title("🏥 MedTech Portfolio Optimizer")
st.subheader("Find the optimal investment portfolio among the world's leading medical technology companies")

# Diccionario de empresas con nombres completos
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
    "STIM": "Neuronetics", "SYK": "Stryker", "TMDX": "TransMedics",
    "TNDM": "Tandem Diabetes", "ATEC": "Alphatec", "ATRC": "AtriCure",
    "ALGN": "Align Technology"
}

# Sidebar con opciones
st.sidebar.header("⚙️ Configuration")

selected = st.sidebar.multiselect(
    "Select companies to analyse:",
    options=list(empresas.keys()),
    format_func=lambda x: f"{x} — {empresas.get(x, x)}",
    default=["MDT", "BSX", "SYK", "ISRG", "DXCM", "EW", "RMD"]
)

years = st.sidebar.slider("Years of historical data:", 2, 10, 5)
num_portfolios = st.sidebar.select_slider(
    "Simulation precision:",
    options=[1000, 5000, 10000, 25000, 50000],
    value=10000
)

if len(selected) < 3:
    st.warning("Please select at least 3 companies to run the analysis.")
else:
    if st.button("🚀 Calculate Optimal Portfolio", type="primary"):
        with st.spinner("Downloading data and running simulation..."):
            
            # Descargar datos
            end = pd.Timestamp.today().strftime("%Y-%m-%d")
            start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
            data = yf.download(selected, start=start, end=end)["Close"]
            data = data.ffill().dropna()
            
            # Calculos
            returns = data.pct_change().dropna()
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            # Monte Carlo
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
            
            # Resultados en columnas
            col1, col2, col3 = st.columns(3)
            col1.metric("📈 Expected Annual Return", f"{results[0,max_idx]*100:.1f}%")
            col2.metric("⚡ Risk (Volatility)", f"{results[1,max_idx]*100:.1f}%")
            col3.metric("⭐ Sharpe Ratio", f"{results[2,max_idx]:.2f}")
            
            # Gráfica Frontera Eficiente
            st.subheader("Efficient Frontier")
            fig, ax = plt.subplots(figsize=(12, 7))
            scatter = ax.scatter(results[1,:], results[0,:], 
                               c=results[2,:], cmap="viridis", 
                               alpha=0.4, s=8)
            plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")
            ax.scatter(results[1,max_idx], results[0,max_idx],
                      marker="*", color="red", s=800, 
                      label="Optimal Portfolio", zorder=5)
            ax.set_xlabel("Risk (Annual Volatility)")
            ax.set_ylabel("Expected Annual Return")
            ax.set_title("MedTech Efficient Frontier")
            ax.legend()
            st.pyplot(fig)
            
            # Tabla cartera óptima
            st.subheader("Optimal Portfolio Allocation")
            df_result = pd.DataFrame({
                "Company": [empresas.get(t, t) for t in selected],
                "Ticker": selected,
                "Allocation (%)": (optimal_w * 100).round(2)
            }).sort_values("Allocation (%)", ascending=False)
            df_result = df_result[df_result["Allocation (%)"] > 0.5]
            st.dataframe(df_result, hide_index=True, use_container_width=True)
            
            # Ranking individual
            st.subheader("Individual Company Performance")
            df_ranking = pd.DataFrame({
                "Company": [empresas.get(t, t) for t in selected],
                "Ticker": selected,
                "Annual Return (%)": (mean_returns * 100).round(1),
                "Risk (%)": (np.sqrt(np.diag(cov_matrix)) * 100).round(1)
            }).sort_values("Annual Return (%)", ascending=False)
            st.dataframe(df_ranking, hide_index=True, use_container_width=True)

st.markdown("---")
st.caption("Built by Nicolás Alfonso Escuredo · Biomedical Engineering & Entrepreneurship · UFV Madrid · 2025")
