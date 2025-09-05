import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.express as px
from io import StringIO

st.set_page_config(page_title="Statistics Assistant", layout = "centered")
st.title("Statistics Assistant")

## Upload Data
option = st.radio("Choose input method", ["Upload CSV", "Paste data"])
df = None

# Upload CSV File
if option == "Upload CSV":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

elif option == "Paste data":
    raw_text = st.text_area("Paste your table here (comma or tab separated)", height=200)
    if raw_text:
        try:
            df = pd.read_csv(StringIO(raw_text), sep=None, engine="python")
        except Exception as e:
            st.error(f"Could not parse data: {e}")



## Data
if df is not None:
    st.subheader("Data Preview")
    st.dataframe(df.head(20))

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if len(num_cols) < 2:
        st.error("Need at least two numeric columns.")
        st.stop()   

    ## Select columns
    col_a = st.selectbox("Column A (numeric)", num_cols)
    col_b = st.selectbox("Column B (numeric)", [c for c in num_cols if c != col_a])


    A = df[col_a].astype(float).to_numpy()
    B = df[col_b].astype(float).to_numpy()

    ## Paired t-test
    tstat, pval = stats.ttest_rel(A,B, nan_policy = "omit")
    diff = A - B
    n = int(np.count_nonzero(~np.isnan(diff)))
    mean_A, mean_B = float(np.nanmean(A)), float(np.nanmean(B))

    # 95% confidence interval
    def confidence_interval(x: np.ndarray):
        x = x[~np.isnan(x)]      
        n = len(x)               
        if n < 2:
            return (np.nan, np.nan, np.nan)
        mean = float(np.mean(x))    
        se = stats.sem(x, nan_policy="omit")   
        crit_val = stats.t.ppf(0.975, df=n-1)     
        ci_low, ci_high = mean - crit_val*se, mean + crit_val*se
        return mean, ci_low, ci_high

    mean_A, loA, hiA = confidence_interval(A)
    mean_B, loB, hiB = confidence_interval(B)

    # Report
    st.markdown(f"""
    ### Results
    **Test**: Paired t‑test between `{col_a}` and `{col_b}` (n={n}).
    **Means (95% CI)**: {col_a} = {mean_A:.3g} [{loA:.3g}, {hiA:.3g}] • {col_b} = {mean_B:.3g} [{loB:.3g}, {hiB:.3g}]
    **t** = {tstat:.3g}, **p** = {pval:.3g}
    """)


    # Summary
    summary = (f"Paired t-test: {col_a} (mean={mean_A:.3g}) vs {col_b} (mean={mean_B:.3g}), n={n}; t={tstat:.3g}, p={pval:.3g}.")
    st.text_area("Copy summary", value=summary, height=80)
    
    # Plot
    plot_type = st.radio("Choose plot type",["Boxplot", "Bar Chart with CI"])
    df_long = pd.melt(df[[col_a, col_b]], value_vars=[col_a, col_b], var_name="Measure", value_name="Value")
    if plot_type == "Boxplot":
        fig = px.box(df_long, x="Measure", y="Value", points="all", title=f"{col_a} vs {col_b}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Bar Chart with CI":
        summary_df = (
            df_long.groupby("Measure")["Value"]
            .apply(lambda g: pd.Series(confidence_interval(g), index=["mean", "ci_low", "ci_high"])).reset_index()
        )

        st.write(summary_df)
        st.write(summary_df.columns.tolist())


        fig = px.bar(
            summary_df,
            x="Measure",
            y="mean",
            error_y=summary_df["ci_high"] - summary_df["mean"],
            error_y_minus=summary_df["mean"] - summary_df["ci_low"],
            title="Bar chart with 95% CI"
        )
        st.plotly_chart(fig, use_container_width=True)



    