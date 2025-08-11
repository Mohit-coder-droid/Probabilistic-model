import streamlit as st
from .probabilistic_models import * 
from .utils import * 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Probabilistic Model Fitter")

# --- Main App Interface ---
st.title("Probabilistic Model Fitting ⚙️")
st.markdown("Upload your data, select the analysis type, map your data columns, and explore various probabilistic models.")

# --- Sidebar for All User Controls ---
with st.sidebar:
    st.header("Controls")

    global selected_files
    selected_files = {}
    
    # 1. File Uploader
    uploaded_files = st.file_uploader(
        "Upload .xlsx or .csv files", 
        type=["xlsx", "csv"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if st.checkbox(f"✔️ {uploaded_file.name}"):
                selected_files[f"{uploaded_file.name}"] = pd.read_excel(uploaded_file)

# --- Main Panel Logic ---
if not uploaded_files:
    st.info("👈 Please upload one or more data files to get started.")
    st.stop()

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


# --- Process Files and Get User Input ---
try:
    # Read and combine all uploaded files into a single DataFrame
    all_files = [pd.read_excel(file) if file.name.endswith('xlsx') else pd.read_csv(file) for file in uploaded_files]
    data = pd.concat(all_files, ignore_index=True)
    
    # Get available column names for dropdowns
    options = data.columns.tolist()

    with st.sidebar:
        # 2. Data Preview
        with st.expander("Preview Uploaded Data"):
            st.dataframe(data.head())

        # 3. Select Analysis Type
        analysis_type = st.radio(
            "Select Analysis Type:",
            ("Yield Stress vs. Temperature", "Fatigue Life Analysis", "Fatigue Crack growth Analysis"),
            help="Choose the type of model you want to fit based on your data."
        )
        
        # --- Column Selectors ---
        st.header("Column Mapping")

        # --- CASE 1: Yield Stress vs. Temperature ---
        if analysis_type == "Yield Stress vs. Temperature":
            x_col = st.selectbox("Select the Temperature column (X-axis)", options, index=options.index('Temperature') if 'Temperature' in options else 0)
            y_col = st.selectbox("Select the Yield Stress column (Y-axis)", options, index=options.index('Mpa') if 'Mpa' in options else 1)
            
            run_button = st.toggle("Run Analysis")

        # --- CASE 2: Fatigue Life Analysis ---
        elif analysis_type == "Fatigue Life Analysis":
            temp_col = st.selectbox("Select the Temperature column", options, index=options.index('Temperature') if 'Temperature' in options else 0)
            strain_col = st.selectbox("Select the Strain Amplitude column", options, index=options.index('Strain amplitude') if 'Strain amplitude' in options else 1)
            cycles_col = st.selectbox("Select the Failure Cycles column", options, index=options.index('Failure cycle') if 'Failure cycle' in options else 2)

            run_button = st.toggle("Run Fatigue Analysis")

        elif analysis_type == "Fatigue Crack growth Analysis":
            temp_col = st.selectbox("Select the Temperature column", options, index=options.index('Temperature, C') if 'Temperature, C' in options else 0)
            c_col = st.selectbox("Select c column", options, index=options.index('c ') if 'c ' in options else 0)
            m_col = st.selectbox("Select m column", options, index=options.index('m') if 'm' in options else 0)
            krange_col = st.selectbox("Select ▲ K Range column", options, index=options.index('▲ K Range') if '▲ K Range' in options else 0)
            r_ratio_col = st.selectbox("Select R- Ratio column", options, index=options.index('R- Ratio') if 'R- Ratio' in options else 0)

            isLinear = st.selectbox("Regression Equation",["Linear","Arrhenius"],index=0)

            run_button = st.toggle("Run Fatigue Crack growth Analysis")
            

# --- Run Analysis and Display Results ---
    if run_button:
        st.header("📈 Model Results")
        
        # --- EXECUTE CASE 1 ---
        if analysis_type == "Yield Stress vs. Temperature":
            # Process data using selected columns
            df, df_dict = df_processor(data, temp_col=x_col, stress_col=y_col)
            X_values = df['Inverse_Temp'].values
            Y_values = df['Mpa'].values

            # Fit various models (your predefined logic)
            weibull = WeibullModel(X_values, Y_values)
            lognormal = LognormalModel(X_values, Y_values)
            weibull_p = WeibullModel(np.log(df[x_col].values), Y_values, power_law=True)
            lognormal_p = LognormalModel(np.log(df[x_col].values), Y_values, power_law=True)
            normal = NormalModel(X_values, Y_values)
            weibull3 = WeibullModel3(X_values, Y_values)
            lognormal3 = LognormalModel3(X_values, Y_values)
            gumbell = Gumbell(X_values, Y_values)
            expo = Exponential(X_values, Y_values)
            gamma = Gamma(X_values, Y_values)
            models = [weibull, lognormal, weibull_p, lognormal_p, normal, weibull3, lognormal3, gumbell, expo, gamma]
            
            # Display results in tabs (your predefined logic)
            tab_models = st.tabs([m.tab_name for m in models])
            for i, tab in enumerate(tab_models):
                with tab:
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

                    row_space1 = st.columns(
                                (0.1, 0.7, 0.1)
                            )
                    with row_space1[1]:
                        with st.container(border=True):
                            st.markdown(f"#### Model Details")
                            models[i].st_description

        # --- EXECUTE CASE 2 ---
        elif analysis_type == "Fatigue Life Analysis":
            data_lcf = data.copy()
            data_lcf['Inverse_Temp'] = 11604.53 / (data_lcf[temp_col] + 273.16)
            data_lcf['Ln_Strain'] = np.log(data_lcf[strain_col])
            diff_temp = data_lcf[temp_col].unique()

            # Fit various models (your predefined logic)
            weibull = WeibullModel(data_lcf['Inverse_Temp'], data_lcf[cycles_col], data_lcf['Ln_Strain'])
            lognormal = LognormalModel(data_lcf['Inverse_Temp'], data_lcf[cycles_col], data_lcf['Ln_Strain'])
            normal = NormalModel(data_lcf['Inverse_Temp'], data_lcf[cycles_col], data_lcf[strain_col])
            expo = Exponential(data_lcf['Inverse_Temp'], data_lcf[cycles_col], data_lcf['Ln_Strain'])
            gumbell = Gumbell(data_lcf['Inverse_Temp'], data_lcf[cycles_col], data_lcf['Ln_Strain'])
            gamma = Gamma(data_lcf['Inverse_Temp'], data_lcf[cycles_col], data_lcf['Ln_Strain'])
            models = [weibull, lognormal, normal, expo, gumbell, gamma]
            
            # Display results in tabs (your predefined logic)
            tab_models = st.tabs([m.tab_name for m in models])
            for i, tab in enumerate(tab_models):
                with tab:
                    st.subheader(models[i].name)
                    plots = st.tabs([f"Plots for Temp: {temp}°" for temp in diff_temp])
                    for j, plot_tab in enumerate(plots):
                        with plot_tab:
                            row_space1 = st.columns(
                                (0.1, 0.7, 0.1)
                                )
                            with row_space1[1]:
                                temp_data = data_lcf[data_lcf[temp_col] == diff_temp[j]]
                                params = models[i].minimize(models[i].bounds, (temp_data['Inverse_Temp'], temp_data[cycles_col], temp_data['Ln_Strain']))
                                plot_different_cdf_two_var(models[i].two_var_predict, params, diff_temp[j], data_lcf)
                    
                    with st.container(border=True):
                        st.markdown(f"#### Model Details")
                        models[i].st_description

        elif analysis_type == "Fatigue Crack growth Analysis":
            df = fatigue_crack_preprocess_df(data,temp_col, c_col, m_col, krange_col, r_ratio_col )
            walker = WalkerEq(df, isLinear=(isLinear=="Linear"))
            # walker = WalkerEq(df, isLinear=False)

            tab_models = st.tabs(["Regression", "da_dN vs R", "da_dN vs ΔK", "da_dN vs Temperature"])

            with tab_models[0]:
                row_space1 = st.columns(
                                (0.1, 0.7, 0.1)
                                )
                with row_space1[1]:
                    op = ["Regression Plot","da/dN regression Plot","da/dN regression error Plot"]
                    walker_plot = st.selectbox("Select Plot",op)
                    
                    if walker_plot == op[0]:
                        walker.regression_plot(walker.slope_, walker.intercept_)
                    if walker_plot == op[1]:
                        walker.regression_dAdN_plot()
                    if walker_plot == op[2]:
                        walker.regression_dAdN_error_plot()

            with tab_models[1]:
                row_space1 = st.columns(
                                (0.1, 0.7, 0.1)
                                )
                with row_space1[1]:
                    walker.plot_da_dN_vs_r_ratio_equation()

            with tab_models[2]:
                row_space1 = st.columns(
                                (0.1, 0.7, 0.1)
                                )
                with row_space1[1]:
                    walker.plot_da_dN_vs_deltaK_equation()

            with tab_models[3]:
                row_space1 = st.columns(
                                (0.1, 0.7, 0.1)
                                )
                with row_space1[1]:
                    walker.plot_da_dN_vs_temperature_equation()
            
            row_space1 = st.columns(
                                (0.1, 0.7, 0.1)
                                )
            with row_space1[1]:
                walker.st_description()


except Exception as e:
    st.error(f"An error occurred during processing: {e}")
    st.exception(e) # This will show a full traceback for easier debugging