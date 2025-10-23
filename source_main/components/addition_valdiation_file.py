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