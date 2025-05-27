import streamlit as st
import pandas as pd
from scipy.stats import weibull_min
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.title("Probabilistic model fitting")

# File uploader
uploaded_files = st.file_uploader("Upload your .xlsx file", type=["xlsx"], accept_multiple_files=True)

def calculate_params(df_dict):
    weibull_params = {}

    for temp, df in df_dict.items():
        # Fit Weibull distribution to Yield Stress data
        shape, loc, scale = weibull_min.fit(df["YS"], floc=0)  # Fix location at 0 for stability
        weibull_params[temp] = {"Shape": shape, "Loc": loc, "Scale": scale}
        
    weibull_df = pd.DataFrame.from_dict(weibull_params, orient="index").reset_index()

    weibull_df.rename(columns={"index": "Temperature"}, inplace=True)

    weibull_df.drop(columns=["Loc"], inplace=True)

    st.subheader("Weibull parameters for different temperature")
    st.dataframe(weibull_df,width=600)

    return weibull_df

def plot_different_cdf(cdf,u,w, df,name):
    global df_4
    shape = np.mean(df_4["Shape"])
    temperature_values = np.linspace(10, 600, 100)

    fig, ax = plt.subplots(figsize=(8,6))

    for (name,df) in selected_files.items():
        ax.scatter(df["Temperature"], df["YS"], edgecolors='black', alpha=0.7, s=30, label=f"{name}")

    for i in range(len(cdf)):
        ys_predicted_cdf = np.exp(
            (w + (u / (temperature_values + 273.15))) +
            ((1 / shape) * np.log(np.log(1 / (1 - cdf[i]))))
        )
        ax.plot(temperature_values, ys_predicted_cdf, linestyle="-", linewidth=1, label=f"Predicted YS (CDF={cdf[i]})")

    ax.set_xlabel("Temperature (°C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Yield Stress (YS)", fontsize=12, fontweight="bold")
    ax.set_title("Yield Stress vs. Temperature Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    st.pyplot(fig)

def preprocessing(df,name):
    global df_4
    data_sorted = df.sort_values(by="Temperature", ascending=True)

    df_3 = data_sorted.reset_index(drop=True)
    df_dict = {temp: df_3[df_3["Temperature"] == temp].reset_index(drop=True) for temp in df_3["Temperature"].unique()}

    weibull_df = calculate_params(df_dict)
    
    df_4 = df_3.merge(weibull_df, on="Temperature", how="left")
    df_4["CDF"] = 1 - np.exp(- (df_4["YS"] / df_4["Scale"]) ** df_4["Shape"]) 

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
    plot_different_cdf([0.5,0.9,0.1,0.99,0.01],u,w,df,name)
    

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
                preprocessing(data,uploaded_file.name)
                # Sort by Temperature
                # data_sorted = data.sort_values(by="Temperature", ascending=True)

                # st.success("File successfully processed and sorted by Temperature.")
                # st.dataframe(data_sorted)
            else:
                st.error("The uploaded file does not contain a 'Temperature' column.")

        # data = pd.concat([df_1, df_2], ignore_index=True)
        # data_sorted = data.sort_values(by="Temperature", ascending=True)
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a .xlsx file to get started.")
