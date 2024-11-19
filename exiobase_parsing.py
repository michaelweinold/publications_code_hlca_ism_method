#%%
# runs code as interactive cell 
# https://code.visualstudio.com/docs/python/jupyter-support-py

import pymrio
import numpy as np
import pandas as pd
from pathlib import Path

path_files_directory: Path = Path.home()/'exiobase' 
path_files_directory.mkdir(parents=True, exist_ok=True)

def create_exiobase_hybrid_dataframe(path_to_csv: Path) -> pd.DataFrame:
    """
    """
    df = pd.read_csv(
        path_to_csv, 
        header=[0,1],
        skiprows=[2,3],
        index_col=[0,1],
        low_memory=False
    )
    df.index.names = ['sector', 'region']
    df.columns.names = ['sector', 'region']

    df = df.drop(df.columns[[0,1,2]], axis=1)
    df.columns = df.index
    df = df.replace(0, np.nan)
    df_sparse = df.astype(pd.SparseDtype("float", np.nan))

    return df_sparse


def create_exiobase_monetary_dataframe():
    """
    Check if Exiobase 3 is already downloaded. If not, download it to `/tmp/`.
    
    Notes
    -----
    - https://pymrio.readthedocs.io/en/latest/notebooks/autodownload.html#EXIOBASE-3-download
    - https://pymrio.readthedocs.io/en/latest/api_doc/pymrio.parse_exiobase3.html#pymrio.parse_exiobase3
    """

    exio3_zip_name: str = "IOT_2012_pxp.zip"
    exio3_folder: str = "/tmp/pymrio/autodownload/EXIO3"
    exio3_path: str = exio3_folder + "/" + exio3_zip_name

    if os.path.exists(exio3_folder + "/" + 'exio3_zip_name'):
        print("Exiobase 3 already downloaded.")
    else:
        print("Downloading Exiobase 3.")
        pymrio.download_exiobase3(
            doi="10.5281/zenodo.4277368",
            storage_folder=exio3_folder,
            system="pxp",
            years=[2012]
        )
    print("Parsing Exiobase 3.")

    exio3 = pymrio.parse_exiobase3(path=exio3_path)

    df = exio3.A
    df = df.replace(0, np.nan)
    df_sparse = df.astype(pd.SparseDtype("float", np.nan))
    
    return df_sparse

# %%

path_exiobase_hybrid = Path(path_files_directory/'MR_HIOT_2011_v3_3_18_by_product_technology.csv')

df_exio_hybrid_sparse = create_exiobase_hybrid_dataframe(path_exiobase_hybrid)
df_exio_hybrid_sparse.to_pickle(path_files_directory/'df_exiobase_A_hybrid_sparse.pkl')

df_exio_monetary_sparse = create_exiobase_monetary_dataframe()
df_exio_monetary_sparse.to_pickle(path_files_directory/'df_exiobase_A_monetary_sparse.pkl')