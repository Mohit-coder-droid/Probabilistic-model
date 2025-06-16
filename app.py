import streamlit as st
from probabilistic_models import *
from utils import *
import pandas as pd
# from scipy.stats import weibull_min, lognorm
import numpy as np
# import scipy.stats as stats
import matplotlib.pyplot as plt
# from scipy.special import erf, erfinv

st.set_page_config(layout = "wide")
st.title("Probabilistic model fitting")

# File uploader
uploaded_files = st.file_uploader("Upload your .xlsx file", type=["xlsx"], accept_multiple_files=True)

def plot_different_cdf(model,cdf=[0.5,0.9,0.1,0.99,0.01]):
    temperature_values = np.linspace(10, 600, 100)
    fig, ax = plt.subplots(figsize=(8,6))

    for (name,df) in selected_files.items():
        ax.scatter(df["Temperature"], df["Mpa"], edgecolors='black', alpha=0.7, s=30, label=f"{name}")

    if st.checkbox(f"✔️ Show Different CDF values", value=True, key=f"{model.name}_cdf_check"):
        for i in range(len(cdf)):
            ys_predicted_cdf = model.predict(cdf[i],temperature_values)
            ax.plot(temperature_values, ys_predicted_cdf, linestyle="-", linewidth=1, label=f"Predicted YS (CDF={cdf[i]})")

    var_cdf = st.slider("Select CDF value", min_value=0.01, max_value=0.99, value=0.5, step=0.01, key=f"{model.name}_slider")
    ys_predicted_cdf = model.predict(var_cdf, temperature_values)
    ax.plot(temperature_values, ys_predicted_cdf, linestyle="-", linewidth=2, label=f"Predicted YS (Selected CDF={var_cdf})")

    ax.set_xlabel("Temperature (°C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Yield Stress (YS)", fontsize=12, fontweight="bold")
    ax.set_title("Yield Stress vs. Temperature Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    st.pyplot(fig)    
    
def line_fit_plot(model,df_dict):
    st.markdown(f"<h2 style='text-align: center;'>{model.name}</h2>", unsafe_allow_html=True)
    fig,ax = plt.subplots(figsize=(10, 6))

    for temp in df_dict.keys():
        data = df_dict[temp]["Mpa"].values
        data = np.sort(data)

        sigma_values, ln_sigma_values,sigma_fit_log, y_fit = model.transform(data)

        ax.scatter(sigma_values, ln_sigma_values, label=f"Temp {temp}")
        ax.plot(sigma_fit_log, y_fit, linestyle='-')

    ax.set_title("Probability Plot with Fitted Line", fontsize=14, fontweight="bold")
    ax.set_xlabel("ln(Data)", fontsize=12)
    ax.set_ylabel(model.transform_y_label, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)
    # ax.tight_layout()
    st.pyplot(fig)

if uploaded_files is not None:
    try:
        global selected_files
        st.subheader("Select files to process:")
        selected_files = {}

        for uploaded_file in uploaded_files:
            if st.checkbox(f"✔️ {uploaded_file.name}"):
                selected_files[f"{uploaded_file.name}"] = pd.read_excel(uploaded_file)

        if (len(selected_files)):
            data = pd.concat(list(selected_files.values()), ignore_index=True)

            # Check if 'Temperature' column exists
            if "Temperature" in data.columns:
                df,df_dict = df_processor(data)
                X_values = df['Inverse_Temp'].values
                Y_values = df['Mpa'].values
                # st.write("It came here")
                weibull = WeibullModel(X_values, Y_values)
                lognormal = LognormalModel(X_values, Y_values)
                normal = NormalModel(X_values, Y_values)

                st.header("Compare Various Models")
                row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns(
                    (0.1, 1, 0.1, 1, 0.1)
                )

                with row1_1:
                    line_fit_plot(weibull, df_dict)

                with row1_2:
                    line_fit_plot(lognormal, df_dict)

                wb, lnm, nm = st.tabs(["Weibull", "LogNormal", "Normal"])

                # st.header("Weibull Model")
                with wb:
                    row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns(
                        (0.1, 1, 0.1, 1.6, 0.1)
                    )
                    with row2_1:
                        weibull.st_description

                    with row2_2:
                        plot_different_cdf(weibull)

                # st.header("LogNormal Model")
                with lnm:
                    row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
                        (0.1, 1, 0.1, 1.6, 0.1)
                    )
                    with row3_1:
                        lognormal.st_description

                    with row3_2:
                        plot_different_cdf(lognormal)

                # st.header("Normal Model")
                with nm:
                    row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns(
                        (0.1, 1, 0.1, 1.6, 0.1)
                    )
                    with row4_1:
                        normal.st_description

                    with row4_2:
                        plot_different_cdf(normal)

            else:
                st.error("The uploaded file does not contain a 'Temperature' column.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a .xlsx file to get started.")
