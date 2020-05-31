import numpy as np
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("data_RNA_Seq_v2_expression_median.txt",sep="\t",index_col=0)
meta = pd.read_csv("data_bcr_clinical_data_patient.txt",sep="\t",header=5)

data.describe()
meta.describe()

data_scaled = preprocessing.scale(data)

