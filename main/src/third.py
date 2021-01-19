import os
import config as cn
from tensorflow.keras.preprocessing import image_dataset_from_directory
from pathlib import Path

# from office31 import office31
import tensorflow as tf
import numpy as np

# from tensorflow.keras.preprocessing.image import image_dataset_from_directory

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# amazon_ds = image_dataset_from_directory(
#     directory=cn.OFFICE_DS_PATH / Path("amazon"),
#     labels="inferred",
#     label_mode="int",
#     batch_size=1,
#     image_size=(227, 227),
# )
# webcam_ds = image_dataset_from_directory(
#     directory=cn.OFFICE_DS_PATH / Path("webcam"),
#     labels="inferred",
#     label_mode="int",
#     batch_size=1,
#     image_size=(227, 227),
# )
# dslr_ds = image_dataset_from_directory(
#     directory=cn.OFFICE_DS_PATH / Path("dslr"),
#     labels="inferred",
#     label_mode="int",
#     batch_size=16,
#     image_size=(227, 227),
# )
print("Hello")
ds1 = tf.data.Dataset.zip((amazon_ds, webcam_ds.repeat(8)))
source = []
target = []
source_labels = []
target_labels = []
# for x, y in tf.data.Dataset.zip((amazon_ds, webcam_ds.repeat(126))):
#     source.append(x[0])
#     target.append(y[0])
#     source_labels.append(x[1])
#     target_labels.append(y[1])
# print("Finished")
# target = [x for x, y in webcam_ds.repeat(10)]
# ds2 = tf.data.Dataset.zip((amazon_ds, dslr_ds.repeat(6)))
# ds3 = tf.data.Dataset.zip((webcam_ds.repeat(4), amazon_ds))
# ds4 = tf.data.Dataset.zip((webcam_ds, dslr_ds.repeat(2)))
# ds5 = tf.data.Dataset.zip((dslr_ds.repeat(6), amazon_ds))
# ds6 = tf.data.Dataset.zip((dslr_ds.repeat(2), webcam_ds))
for x, y in tf.data.Dataset.zip((amazon_ds, webcam_ds.repeat(126))):
    source.append(tf.squeeze(x[0]))
    target.append(y[0])
    source_labels.append(x[1])
    target_labels.append(y[1])

ds_train = tf.data.Dataset.from_tensor_slices(((source, target), source_labels))
ds_test = tf.data.Dataset.from_tensor_slices(((target, target), target_labels))

# train, val, test = office31(~
#     source_name="amazon",
#     target_name="webcam",
#     seed=1000,
#     same_to_diff_class_ratio=2,
#     image_resize=(227, 227),
#     group_in_out=True,  # groups data: ((img_s, img_t), (lbl_s, _lbl_t))
#     framework_conversion="tensorflow",
#     office_path=None,  # automatically downloads to "~/data"
# )

# for id, _ in enumerate(ds):
#     print(id)
# print(len(list(train.as_numpy_iterator())))
# print(len(list(val.as_numpy_iterator())))
# # for id, _ in enumerate(val):
# #     print(id)
# # for id, _ in enumerate(test):
# #     print(id)
# print(len(list(test.as_numpy_iterator())))

# print(list(train.as_numpy_iterator())[0])


# for id, data_imgs in enumerate(train):
#     source.append(data_imgs[0][0].numpy())
#     target.append(data_imgs[0][1].numpy())
#     source_labels.append(data_imgs[1][0].numpy().decode("utf-8"))
#     target_labels.append(data_imgs[1][1].numpy().decode("utf-8"))

# for id, data_imgs in enumerate(val):
#     source.append(data_imgs[0][0].numpy())
#     target.append(data_imgs[0][1].numpy())
#     source_labels.append(data_imgs[1][0].numpy().decode("utf-8"))
#     target_labels.append(data_imgs[1][1].numpy().decode("utf-8"))

# for id, data_imgs in enumerate(test):
#     source.append(data_imgs[0][0].numpy())
#     target.append(data_imgs[0][1].numpy())
#     source_labels.append(data_imgs[1][0].numpy().decode("utf-8"))
#     target_labels.append(data_imgs[1][1].numpy().decode("utf-8"))

np.save("source.npy", np.stack(source))
np.save("target.npy", np.stack(target))
np.save("source_lbl.npy", np.stack(source_labels))
np.save("target_lbl.npy", np.stack(target_labels))

print("Hello1")
