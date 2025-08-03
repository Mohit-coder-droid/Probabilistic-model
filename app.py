import streamlit as st
from probabilistic_models import *
from utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout = "wide")
st.title("Probabilistic model fitting")

# File uploader
uploaded_files = st.file_uploader("Upload your .xlsx file", type=["xlsx"], accept_multiple_files=True)

def plot_different_cdf(model,cdf=[0.5,0.9,0.1,0.99,0.01]):
    temperature_values = np.linspace(10, 600, 100)
    fig, ax = plt.subplots(figsize=(10,6))

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
    fig,ax = plt.subplots(figsize=(10, 6))

    for temp in df_dict.keys():
        data = df_dict[temp]["Mpa"].values
        data = np.sort(data)

        try:
            sigma_values, ln_sigma_values,sigma_fit_log, y_fit = model.transform(data)
        except:
            sigma_values, ln_sigma_values,sigma_fit_log, y_fit = model.transform(data, temp)

        ax.scatter(sigma_values, ln_sigma_values, label=f"Temp {temp}")
        ax.plot(sigma_fit_log, y_fit, linestyle='-')

    ax.set_title("Probability Plot with Fitted Line", fontsize=14, fontweight="bold")
    ax.set_xlabel("ln(Data)", fontsize=12)
    ax.set_ylabel(model.transform_y_label, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)
    st.pyplot(fig)

def plot_different_cdf_two_var(predict, params,temperature,data,cdf=[0.5,0.9,0.1,0.99,0.01]):
    fig, ax = plt.subplots(figsize=(8,6))
    
    strain_values = np.linspace(0.002,0.020, 100)
    temperature_values = np.ones_like(strain_values) * temperature
    data = data[data['Temperature']==temperature]

    ax.scatter(data["Unnamed: 2"] ,data['Strain amplitude'], edgecolors='black', alpha=0.7, s=30, label=f"Vendor 1")

    for i in range(len(cdf)):
        ys_predicted_cdf = predict(cdf[i],temperature_values, strain_values, params)
        ax.plot( ys_predicted_cdf,strain_values, linestyle="-", linewidth=1, label=f"Predicted YS (CDF={cdf[i]})")

    ax.set_xscale('log')
    ax.set_xlabel("Total Strain Amplitude", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized Failure Cycle", fontsize=12, fontweight="bold")
    ax.set_title("For Temperature {}".format(temperature), fontsize=14, fontweight="bold")
    ax.set_xlim(1e-3,1)
    ax.legend()
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
            if "Temperature" in data.columns and "Mpa" in data.columns:
                df,df_dict = df_processor(data)
                X_values = df['Inverse_Temp'].values
                Y_values = df['Mpa'].values

                # Fit various models
                weibull = WeibullModel(X_values, Y_values)
                lognormal = LognormalModel(X_values, Y_values)
                weibull_p = WeibullModel(np.log(df['Temperature'].values), Y_values, power_law=True)
                lognormal_p = LognormalModel(np.log(df['Temperature'].values), Y_values, power_law=True)
                normal = NormalModel(X_values, Y_values)
                weibull3 = WeibullModel3(X_values, Y_values)
                lognormal3 = LognormalModel3(X_values, Y_values)
                gumbell = Gumbell(X_values, Y_values)
                expo = Exponential(X_values, Y_values)
                gamma = Gamma(X_values, Y_values)

                models = [weibull, lognormal, weibull_p, lognormal_p, normal, weibull3, lognormal3, gumbell, expo, gamma]

                st.header("Various Models")

                tab_models = st.tabs([m.tab_name for m in models])

                for i in range(len(tab_models)):
                    with tab_models[i]:
                        st.subheader(models[i].name)
                        plots = st.tabs(["Probability Line Fit Plot", "Yield Stress vs Temperature"])

                        with plots[0]:
                            row_space1 = st.columns(
                               (0.1, 0.7, 0.1)
                            )
                            with row_space1[1]:
                                line_fit_plot(models[i], df_dict)

                        with plots[1]:
                            row_space1 = st.columns(
                               (0.1, 0.7, 0.1)
                            )
                            with row_space1[1]:
                                plot_different_cdf(models[i])

                        st.markdown("<div style='margin-top: 80px;'></div>", unsafe_allow_html=True)

                        row_space1 = st.columns(
                            (0.1, 0.7, 0.1)
                        )
                        with row_space1[1]:
                            with st.container(border=True):
                                models[i].st_description

            # When there are two parameters to be fitted
            elif "Temperature" in data.columns and "Strain amplitude" in data.columns:
                data['Inverse_Temp'] = 11604.53 / (data['Temperature'] + 273.16)
                data['Ln_Strain'] = np.log(data['Strain amplitude'])
                diff_temp = data['Temperature'].unique()

                # Fit various different models
                weibull = WeibullModel(data['Inverse_Temp'], data['Failure cycle'], data['Ln_Strain'])
                lognormal = LognormalModel(data['Inverse_Temp'], data['Failure cycle'], data['Ln_Strain'])
                normal = NormalModel(data['Inverse_Temp'], data['Failure cycle'], data['Strain amplitude']) 
                expo = Exponential(data['Inverse_Temp'], data['Failure cycle'], data['Ln_Strain'])
                gumbell = Gumbell(data['Inverse_Temp'], data['Failure cycle'], data['Ln_Strain'])
                gamma = Gamma(data['Inverse_Temp'], data['Failure cycle'], data['Ln_Strain'])

                models = [weibull, lognormal, normal, expo, gumbell, gamma]
                # models = [weibull, lognormal]

                st.header("Various Models")

                tab_models = st.tabs([m.tab_name for m in models])

                for i in range(len(tab_models)):
                    with tab_models[i]:
                        st.subheader(models[i].name)
                        plots = st.tabs([f"Temperature {diff_temp[j]}" for j in range(len(diff_temp))])

                        for j in range(len(diff_temp)):
                            with plots[j]:
                                row_space1 = st.columns(
                                (0.1, 0.7, 0.1)
                                )
                                with row_space1[1]:
                                    temp_data = data[data['Temperature']==diff_temp[j]]
                                    params = models[i].minimize(models[i].bounds, (temp_data['Inverse_Temp'], temp_data['Failure cycle'], temp_data['Ln_Strain']))
                                    # st.write("Upto here all things are ok")
                                    plot_different_cdf_two_var(models[i].two_var_predict,params,diff_temp[j], data)

                        st.markdown("<div style='margin-top: 80px;'></div>", unsafe_allow_html=True)

                        row_space1 = st.columns(
                            (0.1, 0.7, 0.1)
                        )
                        with row_space1[1]:
                            with st.container(border=True):
                                models[i].st_description
            else:
                st.error("The uploaded file does not contain a 'Temperature' column.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a .xlsx file to get started.")
