import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.utils import resample
import logging

logger = logging.getLogger(__name__)

class Calibration:
    """
    Manages the calibration of diffusion coefficient (D) to apparent molecular weight (MW).
    """
    DEFAULT_STANDARDS = [
        {'name': 'GFP monomer', 'mw_kda': 27, 'd_um2_s': 25.0},
        {'name': 'GFP dimer', 'mw_kda': 54, 'd_um2_s': 17.7},
        {'name': 'GFP trimer', 'mw_kda': 81, 'd_um2_s': 14.4},
    ]

    def __init__(self, standards=None, n_bootstrap=1000):
        """
        Initializes the Calibration object with a list of standards.

        Args:
            standards (list of dicts, optional): A list of dictionaries, where each dict
                represents a standard with keys 'name', 'mw_kda', and 'd_um2_s'.
                Defaults to None, which uses DEFAULT_STANDARDS.
            n_bootstrap (int): Number of bootstrap samples for confidence intervals.
        """
        if standards is None:
            standards = self.DEFAULT_STANDARDS

        self.standards = pd.DataFrame(standards)
        self.n_bootstrap = n_bootstrap
        self.model = None
        self.results = None
        self.alpha = None
        self.bootstrap_params = None
        self.residuals = None
        self.warning = None

        self.fit_model()

    def fit_model(self):
        """
        Fits a log-log linear model to the standards data.
        The model is log10(D) = a + b * log10(MW).
        Also performs bootstrapping to estimate parameter uncertainty.
        """
        df = self.standards.copy()
        df = df[pd.to_numeric(df['mw_kda'], errors='coerce').notnull()]
        df = df[pd.to_numeric(df['d_um2_s'], errors='coerce').notnull()]

        if len(df) < 2:
            self.warning = "Only one standard provided. Using Stokes-Einstein assumption with alpha=0.4."
            logger.warning(self.warning)
            self.alpha = 0.4
            return

        df['log_mw'] = np.log10(df['mw_kda'])
        df['log_d'] = np.log10(df['d_um2_s'])

        X = sm.add_constant(df['log_mw'])
        y = df['log_d']

        self.model = sm.OLS(y, X)
        self.results = self.model.fit()
        self.alpha = -self.results.params['log_mw']
        self.residuals = self.results.resid

        # Bootstrap for prediction intervals
        self.bootstrap_params = []
        if len(df) >= 2:
            for _ in range(self.n_bootstrap):
                df_sample = resample(df)
                X_sample = sm.add_constant(df_sample['log_mw'])
                y_sample = df_sample['log_d']
                model_sample = sm.OLS(y_sample, X_sample).fit()
                self.bootstrap_params.append(model_sample.params)
            self.bootstrap_params = pd.DataFrame(self.bootstrap_params)


    def estimate_apparent_mw(self, D):
        """
        Estimates the apparent molecular weight (MW) for a given diffusion coefficient (D).

        Args:
            D (float): The diffusion coefficient in μm²/s.

        Returns:
            tuple: A tuple containing (mw_kda, ci_low, ci_high).
                   Returns (np.nan, np.nan, np.nan) if the model is not fitted.
        """
        if D <= 0:
            return np.nan, np.nan, np.nan

        log_d = np.log10(D)

        if self.results and self.bootstrap_params is not None and self.residuals is not None:
            # Main estimate
            intercept, slope = self.results.params
            log_mw = (log_d - intercept) / slope
            mw_kda = 10**log_mw

            # Bootstrap for prediction interval
            mw_dist = []
            for i in range(len(self.bootstrap_params)):
                boot_intercept, boot_slope = self.bootstrap_params.iloc[i]
                # Add random residual to the log_d value
                noisy_log_d = log_d + np.random.choice(self.residuals)
                boot_log_mw = (noisy_log_d - boot_intercept) / boot_slope
                mw_dist.append(10**boot_log_mw)

            # Calculate confidence interval from bootstrap distribution
            ci_low = np.percentile(mw_dist, 2.5)
            ci_high = np.percentile(mw_dist, 97.5)

            return mw_kda, ci_low, ci_high

        elif self.alpha is not None: # Single point case
            # Use the single standard as a reference point
            ref_mw = self.standards['mw_kda'].iloc[0]
            ref_d = self.standards['d_um2_s'].iloc[0]

            # D/D_ref = (MW/MW_ref)^(-alpha)
            # MW = MW_ref * (D/D_ref)^(-1/alpha)
            mw_kda = ref_mw * (D / ref_d)**(-1 / self.alpha)

            # No confidence interval for the single-point estimate
            return mw_kda, np.nan, np.nan

        else:
            return np.nan, np.nan, np.nan
