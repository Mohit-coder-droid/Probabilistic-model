import numpy as np
import scipy.stats as stats
from scipy.special import erfinv
from abc import ABC, abstractmethod
from scipy import optimize

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

        self.minimize()

    def log_likelihood(self,params, x, y, three_params:bool=False):
        """Weibull regression model"""
        shape = params[0]
        u = params[1]
        w = params[2]

        if three_params:
            loc = params[3]
        
        if shape <= 0:
            return np.inf
        scale = np.exp(u + w * x)

        if three_params:
            return -np.sum(stats.weibull_min.logpdf(y, c=shape, scale=scale, loc=loc))

        return -np.sum(stats.weibull_min.logpdf(y, c=shape, scale=scale))
    
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

class NormalModel(ProbModel):
    def __init__(self, X_values, Y_values):
        self.X_values = X_values
        self.Y_values = Y_values
        self.minimize()
    
    def log_likelihood(self,params, y):
        return super().log_likelihood()

class LognormalModel(ProbModel):
    def __init__(self, X_values, Y_values):
        self.X_values = X_values
        self.Y_values = Y_values

        self.minimize()

    def log_likelihood(self, params,x, y):
        k,m, sigma = params
        if sigma <= 0:
            return np.inf  # Avoid invalid sigma
        
        mu = k + m * x
        log_likelihood = np.sum(stats.norm.logpdf(np.log(y), loc=mu, scale=sigma) - np.log(y))
        return -log_likelihood  # Minimize negative log-likelihood
    
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
