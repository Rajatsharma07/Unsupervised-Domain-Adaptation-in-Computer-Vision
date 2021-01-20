import os
from pathlib import Path
import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
BASE_DIR = Path("/root/Master-Thesis")  # Base path
# import config as cn
# from tensorflow.keras.preprocessing import image_dataset_from_directory


OFFICE_DS_PATH = BASE_DIR / Path("data/office31/")
# from office31 import office31


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


print("Hello")
# source = []
# target = []
# source_labels = []
# target_labels = []
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
# for x, y in tf.data.Dataset.zip((amazon_ds, webcam_ds.repeat(126))):
#     source.append(tf.squeeze(x[0]))
#     target.append(y[0])
#     source_labels.append(x[1])
#     target_labels.append(y[1])

# ds_train = tf.data.Dataset.from_tensor_slices(((source, target), source_labels))
# ds_test = tf.data.Dataset.from_tensor_slices(((target, target), target_labels))

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

# np.save("source.npy", np.stack(source))
# np.save("target.npy", np.stack(target))
# np.save("source_lbl.npy", np.stack(source_labels))
# np.save("target_lbl.npy", np.stack(target_labels))
source_directory = OFFICE_DS_PATH / "amazon"
target_directory = OFFICE_DS_PATH / "webcam"
# ds_train = tf.data.Dataset.list_files(str(Path(directory + "*.jpg")))


def create_paths(path):
    all_image_paths = [str(path) for path in list(path.glob("*/*"))]
    label_names = sorted(item.name for item in path.glob("*/") if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))

    all_image_labels = [
        label_to_index[Path(path).parent.name] for path in all_image_paths
    ]

    return all_image_paths, all_image_labels


source_images, source_labels = create_paths(source_directory)
target_images, target_labels = create_paths(target_directory)

# path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# label_ds = tf.data.Dataset.from_tensor_slices(all_image_labels)


def process_image(file, label):
    image = tf.io.read_file(file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(
        image,
        [227, 227],
        antialias=True,
        method="nearest",
    )
    image = image / 255.0

    return image, label


# train_ds = tf.data.Dataset.from_tensor_slices(
#     ((source_images, target_images), (source_labels, target_labels))
# )

source_ds = tf.data.Dataset.from_tensor_slices((source_images, source_labels))
source_ds = source_ds.map(process_image)
target_ds = tf.data.Dataset.from_tensor_slices((target_images, target_labels))
target_ds = target_ds.map(process_image)
target_ds = target_ds.repeat(4)

source = []
target = []
source_labels = []
target_labels = []

for x, y in tf.data.Dataset.zip((source_ds, target_ds)):
    source.append(x[0])
    target.append(y[0])
    source_labels.append(x[1])
    target_labels.append(y[1])

# train_ds = train_ds.map(
#     lambda x, y: (process_image(x[0], y[0]), process_image(x[1], y[1])),
#     num_parallel_calls=AUTOTUNE,
# )
ds_train = tf.data.Dataset.from_tensor_slices(((source, target), source_labels))

print("Hello1")
