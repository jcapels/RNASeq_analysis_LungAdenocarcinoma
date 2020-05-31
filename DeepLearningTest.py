import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn import preprocessing

data = pd.read_csv("data_RNA_Seq_v2_expression_median.txt",sep="\t",index_col=1)
meta_patients = pd.read_csv("luad_tcga_clinical_data.tsv",sep="\t",header=0,index_col=2)

# verifying the number of columns and lines of both datasets
meta_patients=meta_patients.drop("Study ID",axis=1) # removing the column "Study ID" of the metadata
print(data.shape)
print(meta_patients.shape)

data_scaled = preprocessing.scale(data.iloc[:,1:])

data_scaled = pd.DataFrame(data_scaled, index =data.index , columns = data.columns[1:])
print(data_scaled.shape)

data_scaled = data_scaled.transpose()

sex = meta_patients.loc[meta_patients["Patient Smoking History Category"].dropna().index, "Sex"]
sex = sex.replace("Male", 1)
sex = sex.replace("MALE", 1)
sex = sex.replace("Female", 0)
data_scaled_ref_mv = data_scaled.loc[meta_patients["Patient Smoking History Category"].dropna().index, :]
data_scaled_ref_mv['Sex'] = sex


def var_converter(var):
    dic = {}
    c = 0
    for i in var.index:
        if var[i] not in dic.keys():
            dic[var[i]] = dic.get(var[i], c + 1)
            c += 1
            var[i] = c
        else:
            var[i] = dic[var[i]]


labels_mv = ["Diagnosis Age", 'American Joint Committee on Cancer Metastasis Stage Code',
             'American Joint Committee on Cancer Metastasis Stage Code',
             'Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code',
             'American Joint Committee on Cancer Tumor Stage Code',
             'Prior Cancer Diagnosis Occurence', 'ICD-10 Classification', "Overall Survival (Months)",
             'Patient Primary Tumor Site',
             'Tissue Source Site', 'Neoplasm Disease Stage American Joint Committee on Cancer Code.1',
             'Fraction Genome Altered']

for label in labels_mv:
    data_scaled_ref_mv = data_scaled_ref_mv.loc[-meta_patients[label].isna(), :]
    label_var = meta_patients.loc[data_scaled_ref_mv.index, label]
    if type(meta_patients[label].tolist()[0]) == str:
        var_converter(label_var)
    data_scaled_ref_mv[label] = label_var

indices_mv = np.random.permutation(data_scaled_ref_mv.index)

input_data_mv=data_scaled_ref_mv
output_data_mv=meta_patients.loc[data_scaled_ref_mv.index,"Patient Smoking History Category"].values

train_in_mv = data_scaled_ref_mv.loc[indices_mv[:-100]]
train_out_mv = meta_patients.loc[indices_mv[:-100],"Patient Smoking History Category"].values

test_in_mv  = data_scaled_ref_mv.loc[indices_mv[-100:]]
test_out_mv = meta_patients.loc[indices_mv[-100:],"Patient Smoking History Category"].values

from keras import models
from keras import layers

print(list(set(train_out_mv)))
print(list(set(test_out_mv)))
train_in_mv = train_in_mv.astype("float32")
# train_out_mv = to_categorical(train_out_mv)
test_in_mv = test_in_mv.astype("float32")
# test_out_mv = to_categorical(test_out_mv)
print(test_out_mv)

def build_model():
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(train_in_mv.shape[1],)))
    network.add(layers.Dense(400, activation='relu'))
    network.add(layers.Dense(200, activation='relu'))
    network.add(layers.Dense(6, activation="softmax"))
    network.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
    return network
# network.fit(train_in_mv.values, train_out_mv, epochs=20, batch_size=50)
# test_loss, test_acc = network.evaluate(test_in_mv, test_out_mv)
# print(test_loss, test_acc)

import numpy as np
k = 5
num_val_samples = len(train_in_mv) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_in_mv[i*num_val_samples: (i+1)*num_val_samples]
    val_targets = train_out_mv[i*num_val_samples: (i+1)*num_val_samples]
    partial_train_data = np.concatenate( [train_in_mv[:i*num_val_samples],
    train_in_mv[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate( [train_out_mv[:i*num_val_samples],
    train_out_mv[(i+1)*num_val_samples:]], axis=0)
    model = build_model()
    print("fjisekandjak")
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1)

val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
all_scores.append(val_mae)
print(all_scores)
print(np.mean(all_scores))
