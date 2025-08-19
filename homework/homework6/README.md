## Data Cleaning Strategy

### Preprocessing Steps:
1. **Missing Value Handling**:
   - Dropped columns with >50% missing data
   - Filled remaining numeric missing values with median

2. **Normalization**:
   - Applied StandardScaler to all numeric columns
   - Ensures features have mean=0 and variance=1

3. **Assumptions**:
   - Median imputation preserves data distribution for numeric columns
   - High missingness columns provide little predictive value
   - Standardization improves model performance for distance-based algorithms

### Trade-offs Considered:
- **Dropping vs Imputation**: Chose to drop high-missing columns to avoid introducing bias
- **Median vs Mean**: Median more robust to outliers in the data
- **Manual vs Automated**: Manual column selection ensures domain relevance