# %%
# runs code as interactive cell 
# https://code.visualstudio.com/docs/python/jupyter-support-py

from pathlib import PurePath, Path
import numpy as np
import pandas as pd

path_files_directory: Path = Path.home()/'exiobase' 
path_files_directory.mkdir(parents=True, exist_ok=True)

# DATA IMPORT ###################################
# download from https://doi.org/10.5281/zenodo.14169675

df_hybrid_A_sparse = pd.read_pickle(path_files_directory/'df_exiobase_A_hybrid_sparse.pkl')
df_hybrid_intermed_sparse = pd.read_pickle(path_files_directory/'df_results_hybrid_spain_sparse.pkl')

df_monetary_A_sparse = pd.read_pickle(path_files_directory/'df_exiobase_A_monetary_sparse.pkl')
df_monetary_intermed_sparse = pd.read_pickle(path_files_directory/'df_results_monetary_spain_sparse.pkl')

# DATA MANIPULATION ##############################


def prepare_sparse_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna(0)
    df = df.sparse.to_dense()
    return df


def identify_inactive_sectors(df: pd.DataFrame) -> pd.MultiIndex:
    zero_rows: pd.MultiIndex = df.index[df.apply(lambda row: np.all(row == 0), axis=1)]
    zero_columns: pd.MultiIndex = df.columns[df.apply(lambda col: np.all(col == 0), axis=0)]
    both_zero: pd.MultiIndex = zero_rows.intersection(zero_columns)
    return both_zero


def identify_household_sectors(df: pd.DataFrame) -> pd.MultiIndex:
    return df.index[df.index.get_level_values(1).str.contains('Private households')]


def remove_inactive_and_household_sectors(
    df_intermediate: pd.DataFrame,
    index_inactive_sectors: pd.MultiIndex,
    index_household_sectors: pd.MultiIndex
) -> pd.DataFrame:
    df_filtered = df_intermediate.drop(
        labels=index_inactive_sectors,
        axis=0,
        errors='ignore'
    )
    df_filtered = df_filtered.drop(
        labels=index_inactive_sectors,
        axis=1,
        errors='ignore'
    )
    df_filtered = df_filtered.drop(
        labels=index_household_sectors,
        axis=0,
        errors='ignore'
    )
    df_filtered = df_filtered.drop(
        labels=index_household_sectors,
        axis=1,
        errors='ignore'
    )
    return df_filtered


def list_zero_columns(df: pd.DataFrame) -> pd.MultiIndex:
    return list(df.columns[df.apply(lambda col: np.all(col == 0), axis=0)])

# %%

df_hybrid_A = prepare_sparse_dataframe(df_hybrid_A_sparse)
df_hybrid_intermed = prepare_sparse_dataframe(df_hybrid_intermed_sparse)

index_inactive_sectors_hybrid = identify_inactive_sectors(df_hybrid_A)
index_household_sectors_hybrid = identify_household_sectors(df_hybrid_A)

df_hybrid_intermed_filtered = remove_inactive_and_household_sectors(
    df_intermediate=df_hybrid_intermed,
    index_inactive_sectors=index_inactive_sectors_hybrid,
    index_household_sectors=index_household_sectors_hybrid
)

list_zero_columns(df_hybrid_intermed_filtered.transpose())

df_hybrid_intermed_filtered.to_pickle(path_files_directory/'df_results_hybrid_v3_3_18_spain_filtered_sparse.pkl')

# %%

df_monetary_A = prepare_sparse_dataframe(df_monetary_A_sparse)
df_monetary_intermed = prepare_sparse_dataframe(df_monetary_intermed_sparse)

index_inactive_sectors_monetary = identify_inactive_sectors(df_monetary_A)
index_household_sectors_monetary = identify_household_sectors(df_monetary_A)

df_monetary_intermed_filtered = remove_inactive_and_household_sectors(
    df_intermediate=df_monetary_intermed,
    index_inactive_sectors=index_inactive_sectors_monetary,
    index_household_sectors=index_household_sectors_monetary
)

list_zero_columns(df_monetary_intermed_filtered.transpose())

df_monetary_intermed_filtered.to_pickle(path_files_directory/'df_results_monetary_v3_8_spain_filtered.pkl')
# %%
