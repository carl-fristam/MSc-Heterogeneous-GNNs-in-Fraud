import os
os.environ['KAGGLE_CACHE_FOLDER'] = r'/Users/cvdf/Developer/AML-GNNs-and-HGMAE/data'

import kagglehub
path = kagglehub.dataset_download("berkanoztas/synthetic-transaction-monitoring-dataset-aml")