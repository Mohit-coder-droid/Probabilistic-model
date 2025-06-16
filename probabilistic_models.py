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

class WeibullModel(ProbModel):
    def __init__(self,X_values, Y_values):
        self.X_values = X_values
        self.Y_values = Y_values
        self.name = "Weibull Model"
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
        st.write("Equation: ")
        st.latex(r"\sigma_f = exp\left(\biggl\{U_t+\frac{W_t}{T}\biggl\}+\frac{1}{m}ln\left(ln\left(\frac{1}{1-f_w}\right)\right)\right)")
        st.write("Here: ")
        st.write(f"m (shape parameter) :  {self.shape:.6f}")
        st.write(f"$W_t$ (slope) :  {self.slope:.6f}")
        st.write(f"$U_t$ (intercept) :  {self.intercept:.6f}")

        return ''

class NormalModel(ProbModel):
    def __init__(self, X_values, Y_values):
        self.X_values = X_values
        self.Y_values = Y_values
        self.name = "Normal Model"
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
        st.write("Equation: ")
        st.latex(r"x = \left( P_t + \frac{R_t}{T} \right) + \left( \sigma_{sl} \sqrt{2} \, \text{erf}^{-1}(2f_n - 1) \right)")
        st.write("Here: ")
        st.write(f"$\sigma_{{sl}}$ (shape parameter) :  {self.sigma:.6f}")
        st.write(f"$R_t$ (slope) :  {self.slope:.6f}")
        st.write(f"$P_t$ (intercept) :  {self.intercept:.6f}")

        return ''
    def __init__(self, X_values, Y_values):
        self.X_values = X_values
        self.Y_values = Y_values
        self.name = "Normal Model"
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
        st.write("Equation: ")
        st.latex(r"f_n = \frac{1}{2} + \frac{1}{2} \, \text{erf} \left( \frac{x - \sigma_m}{\sqrt{2} \sigma_{sl}} \right)")
        st.write("Here: ")
        st.write(f"$\sigma_{{sl}}$ (shape parameter) :  {self.sigma:.6f}")
        st.write(f"$M_t$ (slope) :  {self.slope:.6f}")
        st.write(f"$K_t$ (intercept) :  {self.intercept:.6f}")

        return ''

class LognormalModel(ProbModel):
    def __init__(self, X_values, Y_values):
        self.X_values = X_values
        self.Y_values = Y_values
        self.name = "LogNormal Model"

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
        st.write("Equation: ")
        st.latex(r"f_{ln} = \frac{1}{2} + \frac{1}{2} \, \text{erf} \left[ \frac{\ln(\sigma_f) - \ln(\sigma_m)}{\sqrt{2} \sigma_{sl}} \right]")
        st.write("Here: ")
        st.write(f"$\sigma_{{sl}}$ (shape parameter) :  {self.sigma:.6f}")
        st.write(f"$M_t$ (slope) :  {self.m:.6f}")
        st.write(f"$K_t$ (intercept) :  {self.k:.6f}")

        return ''
    
class WeibullModel3(ProbModel):
    def __init__(self,X_values, Y_values):
        self.X_values = X_values
        self.Y_values = Y_values
        self.name = "3-Parameter Weibull Model"
        self.minimize()

    def log_likelihood(self,params, temp, sigma_values):
        """Weibull regression model"""
        shape = params[0]
        u = params[1]
        w = params[2]
        loc = params[3]

        if shape <= 0:
            return np.inf
        scale = np.exp(u + w * temp)

        return -np.sum(stats.weibull_min.logpdf(sigma_values, c=shape, scale=scale, loc=loc))
    
    def minimize(self):
        init_params = [2.0, np.log(np.mean(self.Y_values)), 0.0,0.0]
        bounds = [(1e-6, None), (None, None), (None, None),(None,None)]

        result_regression = optimize.minimize(
            self.log_likelihood,
            init_params,
            args=(self.X_values, self.Y_values),
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
    def estimate_params(data):
        shape, loc, scale = stats.weibull_min.fit(data)  # Lock location to 0 for typical Weibull fitting
        return shape, scale, loc
    
    def transform(self, data):
        n = len(data)
        cdf_values = np.array([median_rank(n, i + 1) for i in range(n)])

        # Apply Weibull probability plot transformation
        sigma_values = np.log(data)  # X-axis: ln(data)
        wb_sigma_values = np.log(-np.log(1 - cdf_values))  # Y-axis: ln(-ln(1 - p))

        shape, scale,loc = self.estimate_params(data)

        # Generate fitted Weibull line
        sigma_line = np.linspace(min(data), max(data), 100)
        pred_sigma_line = np.log(-np.log(1 - stats.weibull_min.cdf(sigma_line, shape, scale=scale,loc=loc)))

        self.transform_y_label = "ln(-ln(1 - p))"  # name to be displayed in the y-axis of the graph

        return sigma_values, wb_sigma_values, np.log(sigma_line), pred_sigma_line
    
    @property
    def st_description(self):
        st.write("Equation: ")
        st.latex(r"\sigma_f = exp\left(\biggl\{U_t+\frac{W_t}{T}\biggl\}+\frac{1}{m}ln\left(ln\left(\frac{1}{1-f_w}\right)\right)\right)+\delta")
        st.write("Here: ")
        st.write(f"m (shape parameter) :  {self.shape:.6f}")
        st.write(f"$W_t$ (slope) :  {self.slope:.6f}")
        st.write(f"$U_t$ (intercept) :  {self.intercept:.6f}")

        return ''
    

class LognormalModel3(ProbModel):
    def __init__(self, X_values, Y_values):
        self.X_values = X_values
        self.Y_values = Y_values
        self.name = "3-Parameter LogNormal Model"

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
        st.write("Equation: ")
        st.latex(r"\sigma_f = \exp \left( \ln(\sigma_m) + \sqrt{2} \, \sigma_{sl} \, \text{erf}^{-1}(2f_{ln} - 1) \right)+\gamma")
        st.write("Here: ")
        st.write(f"$\sigma_{{sl}}$ (shape parameter) :  {self.sigma:.6f}")
        st.write(f"$M_t$ (slope) :  {self.m:.6f}")
        st.write(f"$K_t$ (intercept) :  {self.k:.6f}")

        return ''
    