#%%
# runs code as interactive cell 
# https://code.visualstudio.com/docs/python/jupyter-support-py

import numpy as np
import pandas as pd
from pathlib import Path

path_files_directory: Path = Path.home()/'exiobase' 
path_files_directory.mkdir(parents=True, exist_ok=True)

def compute_intermediate_demand(
    A: np.ndarray,
    final_demand_sectors: list,
) -> np.ndarray:
    """
    Given a technical coefficient matrix A
    and a list of final demand vectors f and intermediate demand sectors to consider,
    compute the intermediate demand vectors x for each final demand vector f.

    Solves the governing equation of input-output analysis for x:

    $$
    (\mathbf{I} - \mathbf{A}) \vec{x} = \vec{f}
    $$

    Notes
    -----
    Ensure that you have installed NumPy in a way that links it to the highest performing
    BLAS/LAPACK libraries available on your platform.
    You can use `numpy.show_config()` to check which libraries are being used.

    For instance, on macOS (Apple Silicon), you must install NumPy through `pip`.
    As of November 2024, installations through `conda` will not be linked to the Apple Accelerate framework
    and will not use more than one CPU core.

    On a MacBook Pro (M1 Pro CPU), it takes ~4 seconds to compute the array of intermediate demand vectors
    for a single final demand vector. The computation for all sectors of the Spanish economy (~200 sectors)
    therefore takes ~13 minutes.

    See Also
    --------
    - [Miller & Blair (2022, 3rd Edition), Eq.(2.10)](https://doi.org/10.1017/9781108676212)

    Parameters
    ----------
    A : np.ndarray
        Technical coefficient matrix. If taken from Exiobase, is `exio3.A.to_numpy()`.

    intermediate_demand_sectors : list
        List of sector indices (of the matrix A) to consider as intermediate demand sectors. 

    final_demand_sectors : list
        List of sector indices (of the matrix A) to consider as final demand sectors. A demand of 1 unit is placed on each sector.
    """

    I: np.ndarray = np.eye(A.shape[0])

    list_of_intermediate_demand_vectors: list = []

    for i in final_demand_sectors:
        f = np.zeros((A.shape[0], 1))
        f[i] = 1
        x = np.linalg.solve(I-A, f)
        list_of_intermediate_demand_vectors.append(x)

    return np.hstack(list_of_intermediate_demand_vectors)


def compute_intermediate_demand_for_specific_region(
    region: str,
    df: pd.DataFrame,
    filename: str,
) -> None:
    index_positions_region = (df.index.get_level_values(0) == region).nonzero()[0].tolist()

    results_array: np.ndarray = compute_intermediate_demand(
        A=df,
        final_demand_sectors=index_positions_region
    )

    df_results = pd.DataFrame(
        results_array,
        index=df.index,
        columns=df.index[index_positions_region]
    ).replace(0, np.nan)

    df_results_sparse = df_results.astype(pd.SparseDtype("float", np.nan))
    df_results_sparse.to_pickle(filename)


df_exio_hybrid_sparse = pd.read_pickle(path_files_directory/'df_exiobase_A_hybrid_sparse.pkl')
df_exio_monetary_sparse = pd.read_pickle(path_files_directory/'df_exiobase_A_monetary_sparse.pkl')

df_exio_hybrid = df_exio_hybrid_sparse.fillna(0)
df_exio_monetary = df_exio_monetary_sparse.fillna(0)

# %%

compute_intermediate_demand_for_specific_region(
    region="ES",
    df=df_exio_hybrid,
    filename=path_files_directory/'df_results_hybrid_spain_sparse.pkl'
)

compute_intermediate_demand_for_specific_region(
    region="ES",
    df=df_exio_monetary,
    filename=path_files_directory/'df_results_monetary_spain_sparse.pkl'
)