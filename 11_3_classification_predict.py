"""
Use the banks to calsscify the images.
"""

import os, pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

group = "internal"
# group = "external"

top_k = 10
# ------------------------------------------------------------------------------
path_data_train = os.path.join("data", "dataset_classfication_train.npy")

if group == "internal":
    path_data_test = os.path.join("data", "dataset_classfication_test_in.npy")
if group == "external":
    path_data_test = os.path.join("data", "dataset_classfication_test_ex.npy")

# ------------------------------------------------------------------------------
path_analysis = os.path.join("results", "figures", "analysis", "classification", group)
os.makedirs(path_analysis, exist_ok=True)

data_train = np.load(path_data_train, allow_pickle=True).item()
data_test = np.load(path_data_test, allow_pickle=True).item()

images_embedding_train = data_train["images_embedding"]
labels_train = data_train["labels"]
structure_types_train = data_train["structure_types"]

# ------------------------------------------------------------------------------
images_embedding_test_raw = data_test["images_embedding"]
labels_test_raw = data_test["labels"]

# only use the sample with label exist in the training set
images_embedding_test = []
labels_test = []
for i in range(len(labels_test_raw)):
    if labels_test_raw[i] in structure_types_train:
        images_embedding_test.append(images_embedding_test_raw[i])
        labels_test.append(labels_test_raw[i])
images_embedding_test = np.array(images_embedding_test)

structure_types_test = []
for label in labels_test:
    if label not in structure_types_test:
        structure_types_test.append(label)
print("[INFO] smaples in test (orginal):", len(labels_test_raw))
print("[INFO] smaples in test (after filter):", len(labels_test))


# ------------------------------------------------------------------------------
# convert the labels to indices, use the types in training set as the labels
labels_train = [structure_types_train.index(label) for label in labels_train]
labels_test = [structure_types_train.index(label) for label in labels_test]

num_structure_types_train = len(structure_types_train)
num_structure_types_test = len(structure_types_test)

print("-" * 80)
print("[INFO] Training set:")
print("[INFO] Embedding shape: {}".format(images_embedding_train.shape))
print("[INFO] Number of Labels: {}".format(len(labels_train)))
print("[INFO] Structure types: {}".format(structure_types_train))
print("-" * 80)
print("[INFO] Testing set:")
print("[INFO] Embedding shape: {}".format(images_embedding_test.shape))
print("[INFO] Number of Labels: {}".format(len(labels_test)))
print("[INFO] Structure types: {}".format(structure_types_test))
print("-" * 80)

# ------------------------------------------------------------------------------
# image retrieval
# ------------------------------------------------------------------------------
print("[INFO] Calculating cosine similarity matrix...")
dot_ptoduct = np.matmul(images_embedding_test, images_embedding_train.T)
norm_test = np.linalg.norm(images_embedding_test, axis=1).reshape(-1, 1)
norm_train = np.linalg.norm(images_embedding_train, axis=1).reshape(1, -1)
corr_matrix = dot_ptoduct / (norm_test * norm_train)

# ------------------------------------------------------------------------------
# get the top k indices from big to small
top_k_indices = np.argsort(corr_matrix, axis=1)[:, -top_k:]
top_k_indices = top_k_indices[:, ::-1]

# get the top k labels
labels_top_k = []
for i in range(len(labels_test)):
    labels_top_k.append([labels_train[j] for j in top_k_indices[i]])

# the estimated label is the label with the most frequent appearance
labels_est = []
labels_top_k_count = []
minlength = num_structure_types_train
for labels in labels_top_k:
    counts = np.bincount(labels, minlength=minlength)
    labels_top_k_count.append(counts)
    labels_est.append(np.argmax(counts))
    # labels_est.append(labels[0])

# calculate confusion matrix
confusion_matrix = np.zeros((num_structure_types_train, num_structure_types_train))
for i in range(len(labels_test)):
    confusion_matrix[labels_test[i], labels_est[i]] += 1
confusion_matrix = confusion_matrix.astype(int)

print("[INFO] Confusion matrix:")
print(confusion_matrix)
# compute the accuracy
accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
print("[INFO] Accuracy: {:.2f}%".format(accuracy * 100))

# ------------------------------------------------------------------------------
# save the results into a excel file
# convert the label id back to labels
labels_test_str = [structure_types_train[label] for label in labels_test]
labels_est_str = [structure_types_train[label] for label in labels_est]

path_save_excel = os.path.join(path_analysis, "image_retrieval.xlsx")
columes = ["label_gt", "label_est"]
for i in range(minlength):
    columes.append(structure_types_train[i])

data_frame = pandas.DataFrame(columns=columes)
for i in range(len(labels_test)):
    data_frame.loc[i] = [
        labels_test_str[i],
        labels_est_str[i],
    ] + labels_top_k_count[i].tolist()

data_frame.to_excel(path_save_excel, index=False)

# save the confusion matrix
path_save_excel = os.path.join(path_analysis, "confusion_matrix.xlsx")
data_frame = pandas.DataFrame(columns=["label (true)"] + structure_types_train)
for i in range(num_structure_types_train):
    data_frame.loc[i] = [structure_types_train[i]] + confusion_matrix[i].tolist()
data_frame.to_excel(path_save_excel, index=False)
