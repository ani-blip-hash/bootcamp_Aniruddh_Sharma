import pandas as pd

def get_summary_stats(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return basic summary statistics for numeric columns."""
    return dataframe.describe()