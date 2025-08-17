# [Default prediction] - Problem Framing  
**Stage:** 01 (Problem Framing & Scoping)  

## Problem Statement  
Bank needs to predict small business loan defaults because business impact.

## Stakeholder & User  
- **Decision Owner:** Chief Risk Officer
- **Tool User:** Underwriting Team"
- **Decision Window:** Real-time during loan applications and monthly model retraining

## Useful Answer & Decision 
- **Predictive:** Output: Default probability score (0-100%)"

## Assumptions & Constraints
- Data: 3 years of historical loan data available
- Constraints: Model latency <500ms for real-time use

## Known Unknowns / Risks 
- Risk: Regulatory changes impacting model features

## Lifecycle Plan  
- build predictive model → modeling (Stage 01) → python module in /src/ with report

## Repo plan
/data/, /src/, /notebooks/, /docs/ ; cadence for updates

## Data Storage

This project uses the following structure:
- `data/raw/`: For original, unprocessed data in CSV format
- `data/processed/`: For cleaned data in Parquet format

Format choices:
- CSV: Widely compatible, human-readable
- Parquet: Efficient for large datasets, preserves dtypes

Environment variables:
- `DATA_DIR_RAW`: Path for raw data (default: data/raw)
- `DATA_DIR_PROCESSED`: Path for processed data (default: data/processed)

Validation checks:
- Shape consistency
- Datetime and numeric dtypes preserved