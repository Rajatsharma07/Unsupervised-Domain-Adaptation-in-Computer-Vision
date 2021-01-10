# import os
# import matplotlib.pyplot

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import tensorflow_datasets as tfds

# physical_devices = tf.config.list_physical_devices("GPU")
# # tf.config.experimental.set_memory_growth(physical_devices[0], True)

# (ds_train, ds_val, ds_test), ds_info = tfds.load(
#     "visual_domain_decathlon/gtsrb",
#     split=["train", "validation", "test"],
#     shuffle_files=True,
#     as_supervised=True,  # will return tuple (img, label) otherwise dict
#     with_info=True,  # able to get info about dataset
# )

# print("In")
import os
import config as cn
from tensorflow.keras.preprocessing import image_dataset_from_directory
from pathlib import Path
from office31 import office31
import tensorflow as tf
import numpy as np

# from tensorflow.keras.preprocessing.image import image_dataset_from_directory

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


amazon_ds = image_dataset_from_directory(
    directory=cn.OFFICE_DS_PATH / Path("amazon"),
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(227, 227),
)
webcam_ds = image_dataset_from_directory(
    directory=cn.OFFICE_DS_PATH / Path("webcam"),
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(227, 227),
)
dslr_ds = image_dataset_from_directory(
    directory=cn.OFFICE_DS_PATH / Path("dslr"),
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(227, 227),
)
print("Hello")
ds1 = tf.data.Dataset.zip((amazon_ds, webcam_ds.repeat(4)))
ds2 = tf.data.Dataset.zip((amazon_ds, dslr_ds.repeat(6)))
ds3 = tf.data.Dataset.zip((webcam_ds.repeat(4), amazon_ds))
ds4 = tf.data.Dataset.zip((webcam_ds, dslr_ds.repeat(2)))
ds5 = tf.data.Dataset.zip((dslr_ds.repeat(6), amazon_ds))
ds6 = tf.data.Dataset.zip((dslr_ds.repeat(2), webcam_ds))


train, val, test = office31(
    source_name="amazon",
    target_name="webcam",
    seed=1000,
    same_to_diff_class_ratio=2,
    image_resize=(227, 227),
    group_in_out=True,  # groups data: ((img_s, img_t), (lbl_s, _lbl_t))
    framework_conversion="tensorflow",
    office_path=None,  # automatically downloads to "~/data"
)

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
source = []
target = []
source_labels = []
target_labels = []
i = 0
for id, data_imgs in enumerate(train):
    source.append(data_imgs[0][0].numpy())
    target.append(data_imgs[0][1].numpy())
    source_labels.append(data_imgs[1][0].numpy().decode("utf-8"))
    target_labels.append(data_imgs[1][1].numpy().decode("utf-8"))

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
