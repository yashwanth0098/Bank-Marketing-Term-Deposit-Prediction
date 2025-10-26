## Simple Thumb rule before performing the data validation in mlops


| Transformation Type       | Where to Perform         | Reason                                     |
| ------------------------- | ------------------------ | ------------------------------------------ |
| Missing value handling    | Data transformation step | Ensure consistent logic & reproducibility  |
| Encoding (Label, One-Hot) | Data transformation step | Avoid unseen category issues               |
| Scaling / Normalization   | Data transformation step | Maintain same scale in inference           |
| Feature selection / PCA   | Data transformation step | Keep same projection for inference         |
| Data balancing            | Training step only       | Affects only model training, not inference |



# ============================================
#  COMMON DATA TRANSFORMATION STEPS IN MLOps
# ============================================

# 1️.DATA CLEANING
# ----------------------------
# - Handle missing values (imputation or removal)
# - Remove duplicates
# - Fix inconsistent data formats (e.g., dates, category labels)
# - Handle outliers
# - Remove noisy or incorrect entries

# 2️.DATA TYPE CONVERSION
# ----------------------------
# - Convert categorical, numerical, and datetime columns to proper data types
# - Ensure consistent encoding of categorical features

# 3️.FEATURE ENCODING
# ----------------------------
# - Label Encoding
# - One-Hot Encoding
# - Ordinal Encoding
# - Frequency or Target Encoding (for categorical variables)

# 4️.FEATURE SCALING & NORMALIZATION
# ----------------------------
# - Standardization (Z-score)
# - Min-Max normalization
# - Robust scaling (outlier resistance)
# - Log or power transformations (for skewed data)

# 5️. FEATURE ENGINEERING
# ----------------------------
# - Create new derived features or ratios
# - Aggregate group-based statistics
# - Extract text-based features (TF-IDF, embeddings)
# - Extract temporal features (day, month, week, season, etc.)

# 6️.DIMENSIONALITY REDUCTION
# ----------------------------
# - PCA (Principal Component Analysis)
# - Feature selection (filter, wrapper, embedded methods)
# - Remove multicollinear or redundant features

# 7️.DATA INTEGRATION / MERGING
# ----------------------------
# - Combine multiple data sources (joins, merges)
# - Handle mismatched keys or duplicates
# - Align schema across datasets

# 8️.DATA SPLITTING
# ----------------------------
# - Train–validation–test split
# - Stratified sampling (for balanced classes)
# - Time-based splitting (for time-series data)

# 9️.DATA BALANCING (for Classification)
# ----------------------------
# - Oversampling (SMOTE, ADASYN)
# - Undersampling
# - Class weighting

# 10.DATA FORMATTING & SERIALIZATION
# ----------------------------
# - Save in standard formats (Parquet, CSV, TFRecord, etc.)
# - Dataset versioning (DVC, Delta Lake)

# 11.PIPELINE CONSISTENCY
# ----------------------------
# - Ensure reproducible transformations (fit-transform)
# - Use scikit-learn Pipelines or Feature Stores
# - Save transformation logic for inference phase

# 12. DATA VALIDATION & DRIFT CHECKS
# ----------------------------
# - Schema validation (Great Expectations, TFDV)
# - Monitor data drift & distribution shifts
# - Ensure training and production data consistency
