import streamlit as st
import probabilistic_models
import utils
# import pandas as pd
# from scipy.stats import weibull_min, lognorm
# import numpy as np
# import scipy.stats as stats
# import matplotlib.pyplot as plt
# from scipy.special import erf, erfinv

st.set_page_config(layout = "wide")
st.title("Probabilistic model fitting")

# File uploader
uploaded_files = st.file_uploader("Upload your .xlsx file", type=["xlsx"], accept_multiple_files=True)

def weibull_cdf(df_4):
    return 1 - np.exp(- (df_4["YS"] / df_4["Scale"]) ** df_4["Shape"]) 

def lognormal_cdf(df_4):
    return 0.5 + 0.5 * erf((np.log(df_4["YS"]) - df_4["Shape"]) / (np.sqrt(2) * df_4["Scale"]))

def calculate_params(df_dict,fit_formula,cdf,plot_cdf,name, model_name):
    # st.heading(model_name)
    st.header(model_name)
    weibull_params = {}

    for temp, df in df_dict.items():
        # Fit Weibull distribution to Yield Stress data
        shape, loc, scale = fit_formula.fit(df["YS"], floc=0)  # Fix location at 0 for stability
        weibull_params[temp] = {"Shape": shape, "Loc": loc, "Scale": scale}
        
    weibull_df = pd.DataFrame.from_dict(weibull_params, orient="index").reset_index()

    weibull_df.rename(columns={"index": "Temperature"}, inplace=True)

    weibull_df.drop(columns=["Loc"], inplace=True)

    st.subheader("Parameters for different temperature")
    st.dataframe(weibull_df,width=600)

    df_4 = df_3.merge(weibull_df, on="Temperature", how="left")
    df_4["CDF"] = cdf(df_4)

    unique_scales = df_4["Scale"].unique()
    unique_temperatures = df_4["Temperature"].unique()

    # Create new DataFrame from these unique values
    df_5 = pd.DataFrame({"Temperature": unique_temperatures, "Scale": unique_scales})

    df_5["ln_Scale"] = np.log(df_5["Scale"])
    df_5["inv_Temperature"] = 1 / (df_5["Temperature"] + 273.15)

    u, w, r_value, p_value, std_err = stats.linregress(df_5["inv_Temperature"], df_5["ln_Scale"])

    st.write("In the Arrhenius equation: ")
    st.latex(r'''\ln(\sigma_m) = U_t + \frac{W_t}{T}''')
    st.write(f"Intercept: {w}")
    st.write(f"Slope: {u}")
    

    st.subheader("Yield Stress vs. Temperature Comparison")
    plot_different_cdf(df_4,[0.5,0.9,0.1,0.99,0.01],u,w,df,name,plot_cdf, model_name)

def weibull_plot_cdf(temperature_values,shape,u,w,cdf):
    return np.exp(
            (w + (u / (temperature_values + 273.15))) +
            ((1 / shape) * np.log(np.log(1 / (1 - cdf))))
        )

def lognormal_plot_cdf(temperature_values,shape,u,w,cdf):
    return np.exp(
        (w + u / (temperature_values + 273.15)) +
        (np.sqrt(2) * shape * erfinv(2 * cdf - 1))
    )

def plot_different_cdf(df_4,cdf,u,w, df,name, cdf_formula, model_name):
    shape = np.mean(df_4["Shape"])
    temperature_values = np.linspace(10, 600, 100)

    fig, ax = plt.subplots(figsize=(8,6))

    for (name,df) in selected_files.items():
        ax.scatter(df["Temperature"], df["YS"], edgecolors='black', alpha=0.7, s=30, label=f"{name}")

    if st.checkbox(f"✔️ Show Different CDF values", value=True, key=f"{model_name}_cdf_check"):
        for i in range(len(cdf)):
            ys_predicted_cdf = cdf_formula(temperature_values, shape, u, w, cdf[i])
            ax.plot(temperature_values, ys_predicted_cdf, linestyle="-", linewidth=1, label=f"Predicted YS (CDF={cdf[i]})")

    var_cdf = st.slider("Select CDF value", min_value=0.01, max_value=0.99, value=0.5, step=0.01, key=f"{model_name}_slider")
    ys_predicted_cdf = cdf_formula(temperature_values, shape, u, w, var_cdf)
    ax.plot(temperature_values, ys_predicted_cdf, linestyle="-", linewidth=2, label=f"Predicted YS (Selected CDF={var_cdf})")

    ax.set_xlabel("Temperature (°C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Yield Stress (YS)", fontsize=12, fontweight="bold")
    ax.set_title("Yield Stress vs. Temperature Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    st.pyplot(fig)

def preprocessing(df,name):
    global df_4, df_3
    data_sorted = df.sort_values(by="Temperature", ascending=True)

    df_3 = data_sorted.reset_index(drop=True)
    df_dict = {temp: df_3[df_3["Temperature"] == temp].reset_index(drop=True) for temp in df_3["Temperature"].unique()}

    # Make Weibull model
    calculate_params(df_dict,weibull_min,weibull_cdf, weibull_plot_cdf,name, "Weibull Model")

    # Make Lognormal model
    calculate_params(df_dict,lognorm,lognormal_cdf, lognormal_plot_cdf, name, "Lognormal Model")
    
    
if uploaded_files is not None:
    try:
        global selected_files
        st.subheader("Select files to process:")
        selected_files = {}

        for uploaded_file in uploaded_files:
            if st.checkbox(f"✔️ {uploaded_file.name}"):
                selected_files[f"{uploaded_file.name}"] = pd.read_excel(uploaded_file)


        if (len(selected_files)):
            data = pd.concat(list(selected_files.values()))

            # Check if 'Temperature' column exists
            if "Temperature" in data.columns:
                df = df_processor
                preprocessing(data,uploaded_file.name)
            else:
                st.error("The uploaded file does not contain a 'Temperature' column.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a .xlsx file to get started.")
