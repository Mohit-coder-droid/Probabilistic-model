import numpy as np
import scipy.stats as stats
from scipy.special import erfinv
from abc import ABC, abstractmethod
from scipy import optimize
from utils import median_rank
import streamlit as st

class ProbModel(ABC):
    @abstractmethod
    def log_likelihood(self):
        pass

    @abstractmethod
    def minimize(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def st_description(self, cdf, pdf, re_cdf, ar_cdf, fatigue_cdf, variable_values):
        st.header("📋 Mathematical Formulation")
    
        # CDF Section
        st.subheader("1. Cumulative Distribution Function (CDF)")
        st.markdown("""
        The CDF of this model is given by:
        """)
        st.latex(cdf)

        # PDF Section
        st.subheader("2. Probability Density Function (PDF)")
        st.markdown("""
        The PDF of this model can be expressed as:
        """)
        st.latex(pdf)

        # Rearranged equation
        st.subheader("3. Rearranged CDF")
        st.markdown("""
        Equation (1) can be rearranged as:
        """)
        st.latex(re_cdf)

        # Arrhenius equation
        st.subheader("4. Arrhenius Relationship")
        st.markdown("""
        Using Arrhenius equation in Equation (3):
        """)
        st.latex(ar_cdf)

        # Final fatigue life prediction model
        st.subheader("5. Fatigue Life Prediction Model")
        st.markdown("""
        The complete fatigue life prediction model can be presented as:
        """)
        st.latex(fatigue_cdf)

        st.subheader("Variable Values")
        st.markdown("In the case of data given, values of the variables are")
        st.markdown(variable_values)
        

class WeibullModel(ProbModel):
    def __init__(self,X_values, Y_values, power_law=False):
        self.X_values = X_values
        self.Y_values = Y_values
        self.name = "Weibull Model"
        self.tab_name = "Weibull"
        if power_law:
            self.name = "Weibull Model With Power Law"
            self.tab_name = "Weibull (Power)"
        
        self.power_law = power_law
        self.minimize()

    def log_likelihood(self,params, temp, sigma_values):
        """Weibull regression model"""
        shape = params[0]
        u = params[1]
        w = params[2]

        if shape <= 0:
            return np.inf
        scale = np.exp(u + w * temp)

        return -np.sum(stats.weibull_min.logpdf(sigma_values, c=shape, scale=scale))
    
    def minimize(self):
        init_params = [2.0, np.log(np.mean(self.Y_values)), 0.0]
        bounds = [(1e-6, None), (None, None), (None, None)]

        result_regression = optimize.minimize(
            self.log_likelihood,
            init_params,
            args=(self.X_values, self.Y_values),
            bounds=bounds,
            method='L-BFGS-B'
        )

        self.shape, self.intercept, self.slope = result_regression.x

    def predict(self,cdf, temperature_values):
        if self.power_law:
            return np.exp(
                (self.intercept + (self.slope * np.log(temperature_values))) +
                ((1 / self.shape) * np.log(np.log(1 / (1 - cdf))))
            )
        return np.exp(
            (self.intercept + (self.slope * 11604.53 / (temperature_values + 273.16))) +
            ((1 / self.shape) * np.log(np.log(1 / (1 - cdf))))
        )
    
    @staticmethod
    def estimate_params(data):
        shape, loc, scale = stats.weibull_min.fit(data, floc=0)  # Lock location to 0 for typical Weibull fitting
        return shape, scale
    
    def transform(self, data):
        n = len(data)
        cdf_values = np.array([median_rank(n, i + 1) for i in range(n)])

        # Apply Weibull probability plot transformation
        sigma_values = np.log(data)  # X-axis: ln(data)
        wb_sigma_values = np.log(-np.log(1 - cdf_values))  # Y-axis: ln(-ln(1 - p))

        shape, scale = self.estimate_params(data)

        # Generate fitted Weibull line
        sigma_line = np.linspace(min(data), max(data), 100)
        pred_sigma_line = np.log(-np.log(1 - stats.weibull_min.cdf(sigma_line, shape, scale=scale)))
        
        self.transform_y_label = "ln(-ln(1 - p))"  # name to be displayed in the y-axis of the graph

        return sigma_values, wb_sigma_values, np.log(sigma_line), pred_sigma_line
    
    @property
    def st_description(self):
        cdf = r"""
        f_w = 1 - exp\left(-\left(\frac{\sigma_f}{\sigma_m}\right)^m\right) \quad \text{...(1)}
        """

        pdf = r"""
        F_w = \left(\frac{m}{\sigma_m}\right)\left(\frac{\sigma_f}{\sigma_m}\right)^{m-1}exp\left(-\left(\frac{\sigma_f}{\sigma_m}\right)^m\right) \quad \text{...(2)}
        """
        
        re_cdf = r"""
        \sigma_f = exp\left(ln(\sigma_m)+\frac{1}{m}ln\left(ln\left(\frac{1}{1-f_w}\right)\right)\right) \quad \text{...(3)}
        """
        
        ar_cdf = r"""
        \sigma_f = exp\left(\biggl\{U_t+\frac{W_t}{T}\biggl\}+\frac{1}{m}ln\left(ln\left(\frac{1}{1-f_w}\right)\right)\right) \quad \text{...(4)}
        """
        
        fatigue_cdf = r"""
        \sigma_f = exp\left(\biggl\{U_t+\frac{W_t}{T}+Vln(\epsilon)\biggl\}+\frac{1}{m}ln\left(ln\left(\frac{1}{1-f_w}\right)\right)\right) \quad \text{...(5)}
        """

        if self.power_law:
            ar_cdf = r"""
            \sigma_f = exp\left(\biggl\{U_t+W_t ln(T)\biggl\}+\frac{1}{m}ln\left(ln\left(\frac{1}{1-f_w}\right)\right)\right) \quad \text{...(4)}
            """
            
            fatigue_cdf = r"""
            \sigma_f = exp\left(\biggl\{U_t++W_t ln(T)+Vln(\epsilon)\biggl\}+\frac{1}{m}ln\left(ln\left(\frac{1}{1-f_w}\right)\right)\right) \quad \text{...(5)}
            """

        variable_values = f"""
        | Variable | Values |
        |----------|-------------|
        | $U_t$ | {self.intercept:.6f} |
        | $W_t$ | {self.slope:.6f} |
        | $m$ | {self.shape:.6f} |
        """

        super().st_description(cdf, pdf, re_cdf, ar_cdf, fatigue_cdf, variable_values)

        return ''

class NormalModel(ProbModel):
    def __init__(self, X_values, Y_values):
        self.X_values = X_values
        self.Y_values = Y_values
        self.name = "Normal Model"
        self.tab_name = "Normal"
        self.minimize()
    
    def log_likelihood(self, params,temp, sigma_values):
        p,r, sigma = params
        if sigma <= 0:
            return np.inf  # Avoid invalid sigma
        
        mu = p + r * temp
        log_likelihood = np.sum(stats.norm.logpdf(sigma_values, loc=mu, scale=sigma))
        return -log_likelihood

    def minimize(self):
        init_params = [10, 1,1]
        bounds = [(None, None),(None, None), (1e-10, None)]  # mu unbounded, sigma > 0
        result_lognormal = optimize.minimize(
            self.log_likelihood,
            init_params,
            args=(self.X_values, self.Y_values,),
            method='L-BFGS-B',
            bounds=bounds
        )

        self.intercept, self.slope, self.sigma = result_lognormal.x

    def predict(self,cdf, temperature_values):
        z = np.sqrt(2) * self.sigma * erfinv(2 * cdf - 1)
        return self.intercept + (self.slope * 11604.53) / (temperature_values + 273.16) + z
    
    @staticmethod
    def estimate_params(data):
        mu, sigma = np.mean(data), np.std(data, ddof=0)
        return mu, sigma
    
    def transform(self,data):
        n = len(data)
        cdf_values = np.array([median_rank(n, i + 1) for i in range(n)])

        # X-axis: ln(data), Y-axis: inverse CDF of normal
        sigma_values = data
        pred_sigma_values = stats.norm.ppf(cdf_values)  # Normal Quantile

        mu, sigma = self.estimate_params(data)

        # Generate fitted line
        sigma_line = np.linspace(min(data), max(data), 100)

        pred_sigma_line = (sigma_line - mu) / sigma  # Standardized normal score
        # Above line is just a simplification of the below line
        # pred_sigma_line = stats.norm.ppf(stats.norm.cdf(sigma_line, loc=mu,scale=sigma))

        self.transform_y_label = "Standard Normal Quantile"

        return sigma_values, pred_sigma_values, sigma_line, pred_sigma_line
    
    @property
    def st_description(self):
        cdf = r"""
        f_n = \frac{1}{2} + \frac{1}{2} \, \text{erf} \left( \frac{\sigma_f - \sigma_m}{\sqrt{2} \sigma_{sl}} \right) \quad \text{...(1)}
        """

        pdf = r"""
        F_n = \frac{1}{\sigma_{sl} \sqrt{2\pi}} \exp\left( -\left(\frac{\sigma_f - \sigma_m}{\sqrt{2}\sigma_{sl}}\right)^2 \right) \quad \text{...(2)}
        """
        
        re_cdf = r"""
        \sigma_f = \sigma_m + \sigma_{sl} \sqrt{2} \, \text{erf}^{-1}(2f_n - 1) \quad \text{...(3)}
        """
        
        ar_cdf = r"""
        \sigma_f = \left( P_t + \frac{R_t}{T} \right) + \left( \sigma_{sl} \sqrt{2} \, \text{erf}^{-1}(2f_n - 1) \right) \quad \text{...(4)}
        """
        
        fatigue_cdf = r"""
        \sigma_f = \left( P_t + Q_t \ln(\epsilon) + \frac{R_t}{T} \right) + \left( \sigma_{sl} \sqrt{2} \, \text{erf}^{-1}(2f_n - 1) \right) \quad \text{...(5)}
        """

        variable_values = f"""
        | Variable | Values |
        |----------|-------------|
        | $P_t$ | {self.intercept} |
        | $R_t$ | {self.slope} |
        | $\sigma_{{sl}}$ | {self.slope} |
        """

        super().st_description(cdf, pdf, re_cdf, ar_cdf, fatigue_cdf, variable_values)

        return ''

class LognormalModel(ProbModel):
    def __init__(self, X_values, Y_values, power_law=False):
        self.X_values = X_values
        self.Y_values = Y_values
        self.name = "LogNormal Model"
        self.tab_name = "LogNormal"
        if power_law:
            self.name = "LogNormal Model With Power Law"
            self.tab_name = "LogNormal (Power)"
        self.power_law = power_law

        self.minimize()

    def log_likelihood(self, params,temp, sigma_values):
        k,m, sigma = params
        if sigma <= 0:
            return np.inf  # Avoid invalid sigma
        
        mu = k + m * temp
        log_likelihood = np.sum(stats.norm.logpdf(np.log(sigma_values), loc=mu, scale=sigma) - np.log(sigma_values))
        return -log_likelihood  
    
    def minimize(self):
        init_params = [10, 1,1]
        bounds = [(None, None),(None, None), (1e-10, None)]  # mu unbounded, sigma > 0
        result_lognormal = optimize.minimize(
            self.log_likelihood,
            init_params,
            args=(self.X_values, self.Y_values,),
            method='L-BFGS-B',
            bounds=bounds
        )

        self.k, self.m, self.sigma = result_lognormal.x

    def predict(self,cdf, temperature_values):
        z = np.sqrt(2) * self.sigma * erfinv(2 * cdf - 1)
        if self.power_law:
            return np.exp(self.k + self.m * np.log(temperature_values) + z)
        return np.exp(self.k + (self.m * 11604.53) / (temperature_values + 273.16) + z)
    
    @staticmethod
    def estimate_params(data):
        log_data = np.log(data)
        mu, sigma = np.mean(log_data), np.std(log_data, ddof=0)
        return mu, sigma
    
    def transform(self,data):
        n = len(data)
        cdf_values = np.array([median_rank(n, i + 1) for i in range(n)])

        # X-axis: ln(data), Y-axis: inverse CDF of normal
        sigma_values = np.log(data)
        pred_sigma_values = stats.norm.ppf(cdf_values)  # log(Normal quantile)

        mu, sigma = self.estimate_params(data)

        # Generate fitted line
        sigma_line = np.log(np.linspace(min(data), max(data), 100))

        pred_sigma_line = (sigma_line - mu) / sigma  # Standardized normal score
        # Above line is just a simplification of the below line
        # pred_sigma_line = stats.norm.ppf(stats.norm.cdf(sigma_fit_log, loc=mu,scale=sigma))

        self.transform_y_label = "Standard Normal Quantile"

        return sigma_values, pred_sigma_values, sigma_line, pred_sigma_line
    
    @property
    def st_description(self):
        cdf = r"""
        f_{ln} = \frac{1}{2} + \frac{1}{2} \, \text{erf} \left[ \frac{\ln(\sigma_f) - \ln(\sigma_m)}{\sqrt{2} \sigma_{sl}} \right] \quad \text{...(1)}
        """

        pdf = r"""
        F_{ln} = \frac{1}{(\sigma_f) \sigma_{sl} \sqrt{2\pi}} \exp\left( -\left( \frac{\ln(\sigma_f) - \ln(\sigma_m)}{\sqrt{2} \sigma_{sl}} \right)^2 \right) \quad \text{...(2)}
        """
        
        re_cdf = r"""
        \sigma_f = \exp \left( \ln(\sigma_m) + \sqrt{2} \, \sigma_{sl} \, \text{erf}^{-1}(2f_{ln} - 1) \right) \quad \text{...(3)}
        """
        
        ar_cdf = r"""
        \sigma_f = \exp \left( \left\{ K_t + \frac{M_t}{T} \right\} + \left\{ \sqrt{2} \, \sigma_{sl} \, \text{erf}^{-1}(2f_{ln} - 1) \right\} \right) \quad \text{...(4)}
        """
        
        fatigue_cdf = r"""
        \sigma_f = \exp \left( \left\{ K_t + L \ln(\epsilon) + \frac{M_t}{T} \right\} + \left\{ \sqrt{2} \, \sigma_{sl} \, \text{erf}^{-1}(2f_{ln} - 1) \right\} \right) \quad \text{...(5)}
        """

        if self.power_law:
            ar_cdf = r"""
            \sigma_f = \exp \left( \left\{ K_t + M_t ln(T) \right\} + \left\{ \sqrt{2} \, \sigma_{sl} \, \text{erf}^{-1}(2f_{ln} - 1) \right\} \right) \quad \text{...(4)}
            """
            
            fatigue_cdf = r"""
            \sigma_f = \exp \left( \left\{ K_t + L \ln(\epsilon) + M_t ln(T) \right\} + \left\{ \sqrt{2} \, \sigma_{sl} \, \text{erf}^{-1}(2f_{ln} - 1) \right\} \right) \quad \text{...(5)}
        """

        variable_values = f"""
        | Variable | Values |
        |----------|-------------|
        | $K_t$ | {self.k:.6f} |
        | $M_t$ | {self.m:.6f} |
        | $\sigma{{sl}}$ | {self.sigma:.6f} |
        """

        super().st_description(cdf, pdf, re_cdf, ar_cdf, fatigue_cdf, variable_values)

        return ''
    
class WeibullModel3(ProbModel):
    def __init__(self,X_values, Y_values): 
        self.X_values = X_values
        self.Y_values = Y_values
        self.name = "3-Parameter Weibull Model"
        self.tab_name = "3-Parameter Weibull"
        self.minimize()

    def log_likelihood(self,params, temp, sigma_values):
        """Weibull regression model"""
        shape = params[0]
        u = params[1]
        w = params[2]
        loc = params[3]

        if shape <= 0 or np.any(self.Y_values - loc <= 0):
            return np.inf
        scale = np.exp(u + w * temp)

        return -np.sum(stats.weibull_min.logpdf(sigma_values-loc, c=shape, scale=scale))
    
    def minimize(self):
        init_params = [2.0, np.log(np.mean(self.Y_values)), 0.0, np.min(self.Y_values) * 0.9]
        bounds = [(1e-6, None), (None, None), (None, None), (None, np.min(self.Y_values) - 1e-6)]

        rng   = np.random.default_rng(seed=212) 
        perm  = rng.permutation(len(self.X_values))

        result_regression = optimize.minimize(
            self.log_likelihood,
            init_params,
            args=(self.X_values[perm], self.Y_values[perm]),
            bounds=bounds,
            method='L-BFGS-B'
        )

        self.shape, self.intercept, self.slope,self.delta = result_regression.x

    def predict(self,cdf, temperature_values):
        return np.exp(
            (self.intercept + (self.slope * 11604.53 / (temperature_values + 273.16))) +
            ((1 / self.shape) * np.log(np.log(1 / (1 - cdf))))
        ) + self.delta
    
    @staticmethod
    def estimate_params(data, **kwargs):
        shape, loc, scale = stats.weibull_min.fit(data, **kwargs)  # Lock location to 0 for typical Weibull fitting
        return shape, scale, loc
    
    def transform(self, data):
        n = len(data)
        cdf_values = np.array([median_rank(n, i + 1) for i in range(n)])

        shape, scale,loc = self.estimate_params(data, f0=self.shape)

        # Apply Weibull probability plot transformation
        sigma_values = np.log(data-loc)  # X-axis: ln(data)
        wb_sigma_values = np.log(-np.log(1 - cdf_values))  # Y-axis: ln(-ln(1 - p))

        # Generate fitted Weibull line
        sigma_line = np.linspace(min(data), max(data), 100)
        pred_sigma_line = np.log(-np.log(1 - stats.weibull_min.cdf(sigma_line, shape, scale=scale,loc=loc)))

        self.transform_y_label = "ln(-ln(1 - p))"  # name to be displayed in the y-axis of the graph

        return sigma_values, wb_sigma_values, np.log(sigma_line - loc), pred_sigma_line
    
    @property
    def st_description(self):
        cdf = r"""
        f_w = 1 - exp\left(-\left(\frac{\sigma_f - \delta}{\sigma_m}\right)^m\right) \quad \text{...(1)}
        """

        pdf = r"""
        F_w = \left(\frac{m}{\sigma_m}\right)\left(\frac{\sigma_f - \delta}{\sigma_m}\right)^{m-1}exp\left(-\left(\frac{\sigma_f - \delta}{\sigma_m}\right)^m\right) \quad \text{...(2)}
        """
        
        re_cdf = r"""
        \sigma_f = exp\left(ln(\sigma_m)+\frac{1}{m}ln\left(ln\left(\frac{1}{1-f_w}\right)\right)\right)+\delta \quad \text{...(3)}
        """
        
        ar_cdf = r"""
        \sigma_f = exp\left(\biggl\{U_t+\frac{W_t}{T}\biggl\}+\frac{1}{m}ln\left(ln\left(\frac{1}{1-f_w}\right)\right)\right)+\delta \quad \text{...(4)}
        """
        
        fatigue_cdf = r"""
        \sigma_f = exp\left(\biggl\{U_t+\frac{W_t}{T}+Vln(\epsilon)\biggl\}+\frac{1}{m}ln\left(ln\left(\frac{1}{1-f_w}\right)\right)\right)+\delta \quad \text{...(5)}
        """

        variable_values = f"""
        | Variable | Values |
        |----------|-------------|
        | $U_t$ | {self.intercept:.6f} |
        | $W_t$ | {self.slope:.6f} |
        | $m$ | {self.shape:.6f} |
        | $\delta$ | {self.delta:.6f} |
        """

        super().st_description(cdf, pdf, re_cdf, ar_cdf, fatigue_cdf, variable_values)

        return ''

class LognormalModel3(ProbModel):
    def __init__(self, X_values, Y_values):
        self.X_values = X_values
        self.Y_values = Y_values
        self.name = "3-Parameter LogNormal Model"
        self.tab_name = "3-Parameter LogNormal"

        self.minimize()

    def log_likelihood(self, params,temp, sigma_values):
        k,m, sigma,gamma = params
        if sigma <= 0:
            return np.inf  # Avoid invalid sigma
        
        mu = k + m * temp
        log_likelihood = np.sum(stats.norm.logpdf(np.log(sigma_values-gamma), loc=mu, scale=sigma) - np.log(sigma_values))
        return -log_likelihood  
    
    def minimize(self):
        init_params = [10, 1,1,1]
        bounds = [(None, None),(None, None), (1e-10, None),(None, None)]  # mu unbounded, sigma > 0
        result_lognormal = optimize.minimize(
            self.log_likelihood,
            init_params,
            args=(self.X_values, self.Y_values,),
            method='L-BFGS-B',
            bounds=bounds
        )

        self.k, self.m, self.sigma,self.gamma = result_lognormal.x

    def predict(self,cdf, temperature_values):
        z = np.sqrt(2) * self.sigma * erfinv(2 * cdf - 1)
        return np.exp(self.k + (self.m * 11604.53) / (temperature_values + 273.16) + z) + self.gamma
    
    @staticmethod
    def estimate_params(data):
        log_data = np.log(data)
        mu, sigma = np.mean(log_data), np.std(log_data, ddof=0)
        return mu, sigma
    
    def transform(self,data):
        n = len(data)
        cdf_values = np.array([median_rank(n, i + 1) for i in range(n)])

        # X-axis: ln(data), Y-axis: inverse CDF of normal
        sigma_values = np.log(data)
        pred_sigma_values = stats.norm.ppf(cdf_values)  # log(Normal quantile)

        mu, sigma = self.estimate_params(data)

        # Generate fitted line
        sigma_line = np.log(np.linspace(min(data), max(data), 100))

        pred_sigma_line = (sigma_line - mu) / sigma  # Standardized normal score
        # Above line is just a simplification of the below line
        # pred_sigma_line = stats.norm.ppf(stats.norm.cdf(sigma_fit_log, loc=mu,scale=sigma))

        self.transform_y_label = "Standard Normal Quantile"

        return sigma_values, pred_sigma_values, sigma_line, pred_sigma_line
    
    @property
    def st_description(self):
        cdf = r"""
        f_{ln} = \frac{1}{2} + \frac{1}{2} \, \text{erf} \left[ \frac{\ln(\sigma_f-\gamma) - \ln(\sigma_m)}{\sqrt{2} \sigma_{sl}} \right] \quad \text{...(1)}
        """

        pdf = r"""
         F_{ln} = \frac{1}{(\sigma_f-\gamma) \sigma_{sl} \sqrt{2\pi}} \exp\left( -\left( \frac{\ln(\sigma_f-\gamma) - \ln(\sigma_m)}{\sqrt{2} \sigma_{sl}} \right)^2 \right) \quad \text{...(2)}
        """
        
        re_cdf = r"""
        \sigma_f = \exp \left( \ln(\sigma_m) + \sqrt{2} \, \sigma_{sl} \, \text{erf}^{-1}(2f_{ln} - 1) \right)+\gamma \quad \text{...(3)}
        """
        
        ar_cdf = r"""
        \sigma_f = \exp \left( \left\{ K_t + \frac{M_t}{T} \right\} + \left\{ \sqrt{2} \, \sigma_{sl} \, \text{erf}^{-1}(2f_{ln} - 1) \right\} \right)+\gamma \quad \text{...(4)}
        """
        
        fatigue_cdf = r"""
        \sigma_f = \exp \left( \left\{ K_t + L \ln(\epsilon) + \frac{M_t}{T} \right\} + \left\{ \sqrt{2} \, \sigma_{sl} \, \text{erf}^{-1}(2f_{ln} - 1) \right\} \right)+\gamma \quad \text{...(5)}
        """

        variable_values = f"""
        | Variable | Values |
        |----------|-------------|
        | $K_t$ | {self.k:.6f} |
        | $M_t$ | {self.m:.6f} |
        | $\sigma_{{sl}}$ | {self.sigma:.6f} |
        | $\gamma$ | {self.gamma:.6f} |
        """

        super().st_description(cdf, pdf, re_cdf, ar_cdf, fatigue_cdf, variable_values)

        return ''
    
class Gumbell(ProbModel):   
    def __init__(self,X_values, Y_values):
        self.X_values = X_values
        self.Y_values = Y_values
        self.name = "Gumbell Model"
        self.tab_name = "Gumbell"
        self.minimize()

    def log_likelihood(self,params, temp, sigma_values):
        """Gumbell Log likelihood"""
        u = params[0]           # Intercept
        w = params[1]        # Slope
        scale = params[2]        # Scale

        if scale <= 0:
            return np.inf
        
        loc = u + w * temp
        z = (sigma_values - loc) / scale
        z = np.clip(z, -700, 700)  # -exp(-z) overflows around -745
        logpdf = -z - np.exp(-z) - np.log(scale)
        return -np.sum(logpdf)
    
    def minimize(self):
        init_params = [np.mean(self.Y_values), 0.0, np.std(self.Y_values)]
        bounds = [(None, None), (None, None), (1e-6, None)]

        result_regression = optimize.minimize(
            self.log_likelihood,
            init_params,
            args=(self.X_values, self.Y_values),
            bounds=bounds,
            method="L-BFGS-B"
        )

        self.intercept, self.slope,self.scale = result_regression.x

    def predict(self,cdf, temperature_values):
        inv_temp = 11604.53 / (temperature_values + 273.16)
        return self.intercept + self.slope * inv_temp - self.scale * np.log(-np.log(cdf))
    
    @staticmethod
    def estimate_params(data, **kwargs):
        pass
    
    def transform(self, data, temp):
        n = len(data)
        cdf_values = np.array([median_rank(n, i + 1) for i in range(n)])

        # Apply Weibull probability plot transformation
        sigma_values = np.log(data)  # X-axis: data
        wb_sigma_values = -np.log(-np.log(cdf_values))  # Reduced variate Y

        # Predicted location (mean) , 
        # Why are we using temperature here, in this transformation
        inv_temp = 11604.53 / (temp + 273.16)
        mu = self.intercept + self.slope * inv_temp

        # Generate prediction line
        sigma_line = np.linspace(min(data), max(data), 100)
        cdf_fit = stats.gumbel_r.cdf(sigma_line, loc=mu, scale=self.scale)
        pred_sigma_line = -np.log(-np.log(cdf_fit))

        self.transform_y_label = "ln(-ln(1 - p))"  # name to be displayed in the y-axis of the graph

        return sigma_values, wb_sigma_values, np.log(sigma_line), pred_sigma_line
    
    @property
    def st_description(self):
        cdf = r"""
        f_{gb} = exp\left(-exp\left(-\frac{\sigma_f - \mu}{\sigma_m}\right)\right) \quad \text{...(1)}
        """

        pdf = r"""
        F_{gb} = \frac{1}{\sigma_m}\left(-exp\left(-\frac{\sigma_f - \mu}{\sigma_m} + e^{-z}\right)\right) \quad \text{...(2)}
        """
        
        re_cdf = r"""
        \sigma_f = -\sigma_mln(-ln(f_{gb})) + \mu \quad \text{...(3)}
        """
        
        ar_cdf = r"""
         \sigma_f = -\sigma_mln(-ln(f_{gb})) + \left( P_t + \frac{R_t}{T} \right) \quad \text{...(4)}
        """
        
        fatigue_cdf = r"""
        \sigma_f = -\sigma_mln(-ln(f_{gb})) + \left( P_t + Q_tln(\epsilon) +\frac{R_t}{T} \right) \quad \text{...(5)}
        """

        variable_values = f"""
        | Variable | Values |
        |----------|-------------|
        | $U_t$ | {self.intercept:.6f} |
        | $W_t$ | {self.slope:.6f} |
        | $\sigma_m$ | {self.scale:.6f} |
        """

        super().st_description(cdf, pdf, re_cdf, ar_cdf, fatigue_cdf, variable_values)

        return ''

class Exponential(ProbModel):
    def __init__(self,X_values, Y_values):
        self.X_values = X_values
        self.Y_values = Y_values
        self.name = "Exponential Model"
        self.tab_name = "Exponential"
        self.minimize()

    def log_likelihood(self,params, temp, sigma_values):
        """Exponential Log likelihood"""
        u = params[0]
        w = params[1]
        scale = np.exp(u + w * temp)
        
        if np.any(scale <= 0):
            return np.inf
        
        return -np.sum(stats.expon.logpdf(sigma_values, scale=scale))
    
    def minimize(self):
        init_params = [np.log(np.mean(self.Y_values)), 0.0]
        bounds = [(None, None), (None, None)]

        result_regression = optimize.minimize(
            self.log_likelihood,
            init_params,
            args=(self.X_values, self.Y_values),
            bounds=bounds,
            method='L-BFGS-B'
        )

        self.intercept, self.slope = result_regression.x
        self.scale = np.exp(self.intercept + self.slope * self.X_values)

    def predict(self,cdf, temperature_values):
        inv_temp_range = 11604.53 / (temperature_values + 273.16)
        lambda_vals = np.exp(self.intercept + self.slope * inv_temp_range)
        return -lambda_vals * np.log(1 - cdf)
    
    @staticmethod
    def estimate_params(data, **kwargs):
        pass
    
    def transform(self, data, temp):
        n = len(data)
        cdf_values = np.array([median_rank(n, i + 1) for i in range(n)])

        # Exponential probability transformation
        sigma_values = np.log(data)
        exp_sigma_values = np.log(-np.log(1 - cdf_values))  # Reduced variate for exponential

        # Predicted location (mean) , 
        # Why are we using temperature here, in this transformation
        inv_temp = 11604.53 / (temp + 273.16)
        lambda_val = np.exp(self.intercept + self.slope * inv_temp)

        # Generate prediction line
        sigma_line = np.linspace(min(data), max(data), 100)
        cdf_fit = stats.expon.cdf(sigma_line, scale=lambda_val)
        pred_sigma_line = np.log(-np.log(1-cdf_fit))

        self.transform_y_label = "ln(-ln(1 - p))"  # name to be displayed in the y-axis of the graph

        return sigma_values, exp_sigma_values, np.log(sigma_line), pred_sigma_line
    
    @property
    def st_description(self):
        cdf = r"""
        f_{exp} = 1 - \exp\left(- \frac{\sigma_f}{\sigma_m}\right) \quad \text{...(1)}
        """

        pdf = r"""
        F_{exp} = \frac{1}{\sigma_m}\exp\left(- \frac{\sigma_f}{\sigma_m}\right) \quad \text{...(2)}
        """
        
        re_cdf = r"""
        \sigma_f = -\sigma_m \ln(1-f_{exp}) \quad \text{...(3)}
        """
        
        ar_cdf = r"""
        \sigma_f = -\left( U_t +\frac{W_t}{T} \right) \ln(1-f_{exp}) \quad \text{...(4)}
        """
        
        fatigue_cdf = r"""
        \sigma_f = -\left( U_t + V_t\ln(\epsilon) +\frac{W_t}{T} \right) \ln(1-f_{exp}) \quad \text{...(5)}
        """

        variable_values = f"""
        | Variable | Values |
        |----------|-------------|
        | $U_t$ | {self.intercept:.6f} |
        | $W_t$ | {self.slope:.6f} |
        """

        super().st_description(cdf, pdf, re_cdf, ar_cdf, fatigue_cdf, variable_values)

        return ''

class Gamma(ProbModel):
    def __init__(self,X_values, Y_values):
        self.X_values = X_values
        self.Y_values = Y_values
        self.name = "Gamma Model"
        self.tab_name = "Gamma"
        self.minimize()

    def log_likelihood(self,params, temp, sigma_values):
        """Exponential Log likelihood"""
        shape = params[0]
        u = params[1]
        w = params[2]
        scale = np.exp(u + w * temp)
        
        if shape <= 0:
            return np.inf
        
        return -np.sum(stats.gamma.logpdf(sigma_values, a=shape, scale=scale))
    
    def minimize(self):
        init_params = [2.0, np.log(np.mean(self.Y_values)), 0.0]
        bounds = [(1e-6, None), (None, None), (None, None)]

        result_regression = optimize.minimize(
            self.log_likelihood,
            init_params,
            args=(self.X_values,self.Y_values),
            bounds=bounds,
            method='L-BFGS-B'
        )

        self.shape,self.intercept, self.slope = result_regression.x
        self.scale = np.exp(self.intercept + self.slope * self.X_values)

    def predict(self,cdf, temperature_values):
        inv_temp_range = 11604.53 / (temperature_values + 273.16)
        scale_range = np.exp(self.intercept + self.slope * inv_temp_range)
        return stats.gamma.ppf(cdf, a=self.shape, scale=scale_range)
    
    @staticmethod
    def estimate_params(data, **kwargs):
        return stats.gamma.fit(data, **kwargs)
    
    def transform(self, data, temp):
        n = len(data)
        cdf_values = np.array([median_rank(n, i + 1) for i in range(n)])

        # Estimate scale using fixed shape
        _, loc, scale = stats.gamma.fit(data, floc=0, f0=self.shape)

        # Exponential probability transformation
        sigma_values = np.log(data)
        gm_sigma_values = stats.gamma.ppf(cdf_values, a=self.shape)

        # Predicted location (mean)
        inv_temp = 11604.53 / (temp + 273.16)
        lambda_val = np.exp(self.intercept + self.slope * inv_temp)

        # Generate prediction line
        sigma_line = np.linspace(min(data), max(data), 100)
        pred_sigma_line = stats.gamma.ppf(stats.gamma.cdf(sigma_line, a=self.shape, scale=scale), a=self.shape)

        self.transform_y_label = "ln(-ln(1 - p))"  # name to be displayed in the y-axis of the graph

        return sigma_values, gm_sigma_values, np.log(sigma_line), pred_sigma_line
    
    @property
    def st_description(self):
        cdf = r"""
        f_{gm}(x) = \frac{1}{\Gamma(m)} \, \gamma\left(m, \frac{\sigma_f}{\sigma_m}\right) \quad \text{...(1)}
        """

        pdf = r"""
        F_{gm}(x) = \frac{1}{\sigma_m^{\alpha}\Gamma(m)} \,\sigma_f^{\alpha-1}\,exp\left(-\frac{\sigma_f}{\sigma_m}\right) \quad \text{...(2)}
        """
        
        st.header("📋 Mathematical Formulation")
    
        # CDF Section
        st.subheader("1. Cumulative Distribution Function (CDF)")
        st.markdown("""
        The CDF of this model is given by:
        """)
        st.latex(cdf)

        # PDF Section
        st.subheader("2. Probability Density Function (PDF)")
        st.markdown("""
        The PDF of this model can be expressed as:
        """)
        st.latex(pdf)

        st.markdown("Equation (1) can't be written in terms of $\sigma_f$, so we have to do iteration to get $\sigma_f$")

        st.subheader("Variable Values")
        st.markdown("In the case of data given, values of the variables are")
        variable_values = f"""
        | Variable | Values |
        |----------|-------------|
        | $U_t$ | {self.intercept:.6f} |
        | $W_t$ | {self.slope:.6f} |
        | $\sigma_m$ | {self.shape:.6f} |
        """
        st.markdown(variable_values)

        return ''
    