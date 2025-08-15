
def get_summary_stats(df, group_col='category', value_col='value'):   
    return {
        'mean': df.groupby(group_col)[value_col].mean(),
        'median': df.groupby(group_col)[value_col].median(),
        'std': df.groupby(group_col)[value_col].std()
    }
