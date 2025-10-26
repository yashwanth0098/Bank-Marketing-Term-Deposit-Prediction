## Note this for the understanding purpose
## This file contains the common data validation step we perform

# =============================================================================
# DATA VALIDATION — CODING CHECKLIST
# =============================================================================

# 1. BASIC SETUP
# -----------------------------------------------------------------------------
# ☐ Import all required libraries
# ☐ Set up logging for validation steps  
# ☐ Read input datasets — base_data (train/reference) & current_data (new)
# ☐ Load schema/config file (YAML or JSON) for rules
# ☐ Initialize validation report dictionary

#  2. SCHEMA VALIDATION
# -----------------------------------------------------------------------------
# ☐ Check all expected columns are present
# ☐ Check no extra/unexpected columns are added
# ☐ Check data types of each column (match schema)
# ☐ Check column order (optional but recommended)
# ☐ Record missing/extra/type mismatch columns in report

#  3. MISSING VALUE CHECKS
# -----------------------------------------------------------------------------
# ☐ Calculate missing percentage per column
# ☐ Compare with allowed threshold (e.g., 30%)
# ☐ Flag columns exceeding threshold
# ☐ Optionally impute or drop those columns (if allowed)

# 4. DUPLICATE CHECKS
# -----------------------------------------------------------------------------
# ☐ Count duplicate rows
# ☐ Identify duplicate primary keys (if key column exists)
# ☐ Log number of duplicates detected

# 5. STATISTICAL DRIFT CHECKS
# -----------------------------------------------------------------------------
# ☐ Select numeric and categorical columns separately
# ☐ For numeric: use KS-Test or PSI to compare train vs current distribution
# ☐ For categorical: use Chi-Square or PSI
# ☐ Record p-values or drift score
# ☐ Mark drift_detected = True if threshold crossed (e.g., p < 0.05)

# 6. RANGE / DOMAIN CHECKS
# -----------------------------------------------------------------------------
# ☐ Validate numeric columns are within valid min–max range
# ☐ Validate categorical values are within allowed domain (from schema)
# ☐ Record invalid/out-of-bound values

# 7. OUTLIER DETECTION (Optional but Good Practice)
# -----------------------------------------------------------------------------
# ☐ Detect outliers using z-score or IQR method
# ☐ Record number and percentage of outliers per column

# 8. DATA TYPE VALIDATION
# -----------------------------------------------------------------------------
# ☐ Ensure integer/float/string types are correct
# ☐ Attempt conversions if safe, else log as error
# ☐ Use libraries like pandera or pydantic (optional for automation)

# 9. INTEGRITY / RELATIONSHIP CHECKS (Optional)
# -----------------------------------------------------------------------------
# ☐ Validate logical relationships (e.g., start_date <= end_date)
# ☐ Check ID consistency across files (foreign key checks if multi-table)

# 10. GENERATE VALIDATION REPORT
# -----------------------------------------------------------------------------
# ☐ Combine results from all checks into one dictionary/report
# ☐ Assign overall status = "Passed" or "Failed"
# ☐ Save report as .json or .yaml file under artifacts folder

# 11. LOGGING AND EXCEPTION HANDLING
# -----------------------------------------------------------------------------
# ☐ Add logging messages for each step (info, warning, error)
# ☐ Handle exceptions gracefully and log errors
# ☐ Raise validation error if critical checks fail

# 12. RETURN VALIDATION STATUS
# -----------------------------------------------------------------------------
# ☐ Return True/False or a validation summary
# ☐ If validation fails → stop next pipeline step (data transformation)
# =============================================================================



## How to Decide Which Drift Method to Use

| Type                                      | Use When                      | Works Best For                    | Description                                                                                        |
| ----------------------------------------- | ----------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------- |
| **1. KS Test (Kolmogorov–Smirnov)**       | Continuous (numeric) features | Medium → large datasets           | Non-parametric test comparing distributions (baseline vs new). Sensitive to distributional shifts. |
| **2. PSI (Population Stability Index)**   | Continuous or categorical     | Large-scale production monitoring | Measures how much distribution has shifted (binned version of drift). Easy to interpret.           |
| **3. K2 SAM (Chi-square test)**           | Categorical features          | Moderate data                     | Tests difference between categorical distributions. Often used for drift in discrete variables.    |
| **4. JS Divergence / KL Divergence**      | Continuous + Categorical      | Theoretical or research use       | Measures information difference between two probability distributions. More mathematical.          |
| **5. Wasserstein / Earth Mover Distance** | Numerical features            | High-resolution continuous data   | Measures distance between cumulative distributions. Very effective for continuous data drift.      |


## Drift Across Model Types

| Model Type                   | Drift Concept                                       | What to Track                                            | Typical Methods                  |
| ---------------------------- | --------------------------------------------------- | -------------------------------------------------------- | -------------------------------- |
| **Classification**           | Same concept                                        | Feature drift + label drift (class ratio changes)        | KS, PSI, K2 SAM                  |
| **Regression**               | Same concept                                        | Feature drift + target drift (mean/std shift)            | KS, PSI, Wasserstein             |
| **Anomaly Detection**        | Critical (since anomalies depend on data pattern)   | Input feature drift                                      | KS, PSI, JS divergence           |
| **Ranking / Recommendation** | Drift affects user-item features and embeddings     | Track drift in embedding norms or category distributions | PSI, cosine similarity drift     |
| **LLMs / NLP Models**        | Concept drift in text embeddings or prompt patterns | Monitor embedding similarity drift or vocabulary change  | Cosine similarity, JS divergence |



## References of other codes as discussed above :

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance, entropy
from sklearn.metrics.pairwise import cosine_similarity
import os, sys, json

class DriftDetection:
    def __init__(self, data_validation_config):
        self.data_validation_config = data_validation_config

    # =========================
    # 1️⃣ Population Stability Index (PSI)
    # =========================
    def calculate_psi(self, expected, actual, buckets=10):
        """Calculate Population Stability Index (PSI) for numeric features."""
        try:
            expected = expected.dropna()
            actual = actual.dropna()

            quantiles = np.linspace(0, 1, buckets + 1)
            breakpoints = np.quantile(expected, quantiles)

            expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
            actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

            expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
            actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

            psi_value = np.sum((expected_percents - actual_percents) *
                               np.log(expected_percents / actual_percents))
            return psi_value
        except Exception as e:
            raise Exception(f"PSI calculation failed: {e}")

    # =========================
    # 2️⃣ KS Statistic
    # =========================
    def calculate_ks(self, expected, actual):
        """Kolmogorov–Smirnov test for feature distribution drift."""
        try:
            expected, actual = expected.dropna(), actual.dropna()
            ks_stat, p_value = ks_2samp(expected, actual)
            return {"ks_statistic": ks_stat, "p_value": p_value}
        except Exception as e:
            raise Exception(f"KS test failed: {e}")

    # =========================
    # 3️⃣ Wasserstein Distance
    # =========================
    def calculate_wasserstein(self, expected, actual):
        """Compute Wasserstein distance (earth mover’s distance)."""
        try:
            expected, actual = expected.dropna(), actual.dropna()
            return wasserstein_distance(expected, actual)
        except Exception as e:
            raise Exception(f"Wasserstein calculation failed: {e}")

    # =========================
    # 4️⃣ Jensen–Shannon Divergence
    # =========================
    def calculate_js_divergence(self, expected, actual, bins=30):
        """Jensen–Shannon Divergence for probability distribution drift."""
        try:
            expected, actual = expected.dropna(), actual.dropna()
            e_hist, _ = np.histogram(expected, bins=bins, density=True)
            a_hist, _ = np.histogram(actual, bins=bins, density=True)

            e_hist = np.where(e_hist == 0, 1e-6, e_hist)
            a_hist = np.where(a_hist == 0, 1e-6, a_hist)

            m = 0.5 * (e_hist + a_hist)
            js = 0.5 * (entropy(e_hist, m) + entropy(a_hist, m))
            return js
        except Exception as e:
            raise Exception(f"JS Divergence failed: {e}")

    # =========================
    # 5️⃣ Cosine Similarity Drift (for embeddings)
    # =========================
    def calculate_cosine_drift(self, expected_embeddings, actual_embeddings):
        """Cosine similarity drift for vector embeddings."""
        try:
            expected_mean = np.mean(expected_embeddings, axis=0).reshape(1, -1)
            actual_mean = np.mean(actual_embeddings, axis=0).reshape(1, -1)
            similarity = cosine_similarity(expected_mean, actual_mean)[0][0]
            drift = 1 - similarity  # higher = more drift
            return drift
        except Exception as e:
            raise Exception(f"Cosine similarity drift failed: {e}")

    # =========================
    # 🧾 Combined Drift Report
    # =========================
    def detect_dataset_drift(self, base_df, current_df, psi_threshold=0.25):
        """
        Detect dataset drift using multiple statistical methods.
        """
        try:
            report = {}

            for col in base_df.columns:
                if pd.api.types.is_numeric_dtype(base_df[col]):
                    psi = self.calculate_psi(base_df[col], current_df[col])
                    ks = self.calculate_ks(base_df[col], current_df[col])
                    wasserstein = self.calculate_wasserstein(base_df[col], current_df[col])
                    js = self.calculate_js_divergence(base_df[col], current_df[col])

                    drift_detected = psi >= psi_threshold or ks["p_value"] < 0.05

                    report[col] = {
                        "PSI": round(psi, 4),
                        "KS_Statistic": round(ks["ks_statistic"], 4),
                        "KS_p_value": round(ks["p_value"], 4),
                        "Wasserstein": round(wasserstein, 4),
                        "JS_Divergence": round(js, 4),
                        "Drift_Detected": bool(drift_detected)
                    }
                else:
                    report[col] = {"Note": "Non-numeric column – skipped."}

            # Save report as JSON/YAML
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            with open(drift_report_file_path, "w") as f:
                json.dump(report, f, indent=4)

            # Return True if no drift, False otherwise
            return not any(v.get("Drift_Detected", False) for v in report.values())

        except Exception as e:
            raise Exception(f"Drift detection failed: {e}")
