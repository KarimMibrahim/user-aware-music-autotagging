# General Imports
import os
import numpy as np
import pandas as pd
from time import strftime, localtime
import matplotlib.pyplot as plt
import seaborn as sn

# Deep Learning
import tensorflow as tf

# importing the utility functions defined in utilities.py
from utilities import *

plt.rcParams.update({'font.size': 22})
os.environ["CUDA_VISIBLE_DEVICES"]="2"

SOURCE_PATH = "/src_code/repo/"
SPECTROGRAMS_PATH = "/src_code/repo/spectrograms"
OUTPUT_PATH = "/src_code/repo/experiments_results/"
# [TODO] PATH to the pretrained model
extra_exp_dir =   "/src_code/repo/extra_experiment_results/audio_system_multilabel/2020-10-01_10-12-32"


EXPERIMENTNAME = "audio_system_single_label"
INPUT_SHAPE = (646, 96, 1)
LABELS_LIST = ['car', 'gym', 'happy', 'night', 'relax',
       'running', 'sad', 'summer', 'work', 'workout']

global_labels = pd.read_csv("/src_code/repo/GroundTruth/single_label_all.csv")


BATCH_SIZE = 32

# Dataset pipelines
def get_labels_py(song_id,user_id):
    labels = global_labels[global_labels.song_id == song_id][global_labels.user_id == user_id]
    labels = labels.iloc[:, 2:].values.flatten() # TODO: fix this shift in dataframe columns when read
    labels = labels.astype(np.float32)
    return labels


def tf_get_labels_py(sample, device="/cpu:0"):
    with tf.device(device):
        input_args = [sample["song_id"],sample["user_id"]]
        labels = tf.py_func(get_labels_py,
                            input_args,
                            [tf.float32],
                            stateful=False)
        res = dict(list(sample.items()) + [("binary_label", labels)])
        return res


def get_dataset(input_csv, input_shape=INPUT_SHAPE, batch_size=32, shuffle=True,
                infinite_generator=True, random_crop=False, cache_dir=os.path.join(OUTPUT_PATH, "tmp/tf_cache/"),
                num_parallel_calls=32):
    # build dataset from csv file
    dataset = dataset_from_csv(input_csv)
    # Shuffle data
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100, seed=1, reshuffle_each_iteration=True)

    # compute mel spectrogram
    dataset = dataset.map(lambda sample: load_spectrogram_tf(sample), num_parallel_calls=1)

    # filter out errors
    dataset = dataset.filter(lambda sample: tf.logical_not(sample["error"]))

    # map dynamic compression
    C = 100
    dataset = dataset.map(lambda sample: dict(sample, features=tf.log(1 + C * sample["features"])),
                          num_parallel_calls=num_parallel_calls)

    # Apply permute dimensions
    dataset = dataset.map(lambda sample: dict(sample, features=tf.transpose(sample["features"], perm=[1, 2, 0])),
                          num_parallel_calls=num_parallel_calls)

    # Filter by shape (remove badly shaped tensors)
    dataset = dataset.filter(lambda sample: check_tensor_shape(sample["features"], input_shape))

    # set features shape
    dataset = dataset.map(lambda sample: dict(sample,
                                              features=set_tensor_shape(sample["features"], input_shape)))

    # if cache_dir:
    #    os.makedirs(cache_dir, exist_ok=True)
    #    dataset = dataset.cache(cache_dir)

    dataset = dataset.map(lambda sample: tf_get_labels_py(sample), num_parallel_calls=1)


    # set output shape
    dataset = dataset.map(lambda sample: dict(sample, binary_label=set_tensor_shape(
        sample["binary_label"], (len(LABELS_LIST)))))

    if infinite_generator:
        # Repeat indefinitly
        dataset = dataset.repeat(count=-1)

    # Make batch
    dataset = dataset.batch(batch_size)

    # Select only features and annotation
    dataset = dataset.map(lambda sample: (
    sample["features"], sample["binary_label"],sample["song_id"],sample["user_id"]))

    return dataset

def get_model(x_input, current_keep_prob, train_phase):
    # Define model architecture
    # C4_model
    x_norm = tf.layers.batch_normalization(x_input, training=train_phase)

    with tf.name_scope('CNN_1'):
        conv1 = conv_layer_with_relu(x_norm, [3, 3, 1, 32], name="conv_1")
        max1 = max_pooling(conv1, shape=[1, 2, 2, 1], name="max_pool_1")

    with tf.name_scope('CNN_2'):
        conv2 = conv_layer_with_relu(max1, [3, 3, 32, 64], name="conv_2")
        max2 = max_pooling(conv2, shape=[1, 2, 2, 1], name="max_pool_2")

    with tf.name_scope('CNN_3'):
        conv3 = conv_layer_with_relu(max2, [3, 3, 64, 128], name="conv_3")
        max3 = max_pooling(conv3, shape=[1, 2, 2, 1], name="max_pool_3")

    with tf.name_scope('CNN_4'):
        conv4 = conv_layer_with_relu(max3, [3, 3, 128, 256], name="conv_4")
        max4 = max_pooling(conv4, shape=[1, 2, 2, 1], name="max_pool_4")

    with tf.name_scope('Fully_connected_1'):
        flattened = tf.reshape(max4, [-1, 41 * 6 * 256])
        fully1 = tf.nn.sigmoid(full_layer(flattened, 256))


    with tf.name_scope('Fully_connected_2'):
        dropped = tf.nn.dropout(fully1, keep_prob=current_keep_prob)
        logits = full_layer(dropped, len(LABELS_LIST))

    output = tf.nn.sigmoid(logits)
    tf.summary.histogram('outputs', output)
    return logits, output


def main():
    """
    Run the main loop
    """
    print("Current Experiment: " + EXPERIMENTNAME + "\n\n\n")
    # Loading datasets
    # TODO: fix directories

    # Setting up model
    y = tf.placeholder(tf.float32, [None, len(LABELS_LIST)], name="true_labels")
    x_input = tf.placeholder(tf.float32, [None, 646, 96, 1], name="input")
    current_keep_prob = tf.placeholder(tf.float32, name="dropout_rate")
    train_phase = tf.placeholder(tf.bool, name="is_training")
    logits, model_output = get_model(x_input,current_keep_prob, train_phase)
    one_hot = tf.one_hot(tf.argmax(model_output, dimension = 1), depth = len(LABELS_LIST))

    # Defining loss and metrics
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
    '''
    These following lines are needed for batch normalization to work properly
    check https://timodenk.com/blog/tensorflow-batch-normalization/
    '''
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):
    #    train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Setting up saving directory
    experiment_name = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    exp_dir = os.path.join(OUTPUT_PATH, EXPERIMENTNAME, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Loading model with best validation
        saver.restore(sess, os.path.join(extra_exp_dir, "best_validation.ckpt"))
        print("Model with best validation restored before testing.")

        test_labels = pd.read_csv(os.path.join(SOURCE_PATH, "GroundTruth/test_set.csv"))
        test_dataset = get_dataset(os.path.join(SOURCE_PATH, "GroundTruth/test_set.csv"),shuffle = False)
        test_classes = np.zeros_like(test_labels.iloc[:, 2:].values, dtype=float)
        # test_images, test_classes = load_test_set_raw(test_split)

        TEST_NUM_STEPS = int(np.floor((len(test_classes) / 32)))
        # split_size = int(len(test_classes) / TEST_NUM_STEPS)
        test_pred_prob = np.zeros_like(test_classes, dtype=float)
        test_one_hot = np.zeros_like(test_classes, dtype=float)
        test_iterator = test_dataset.make_one_shot_iterator()
        test_next_element = test_iterator.get_next()
        test_song_ids = np.zeros([test_classes.shape[0],1])
        test_user_ids = np.zeros([test_classes.shape[0],1])

        for test_batch_counter in range(TEST_NUM_STEPS):
            start_idx = (test_batch_counter * BATCH_SIZE)
            end_idx = (test_batch_counter * BATCH_SIZE) + BATCH_SIZE
            test_batch = sess.run(test_next_element)
            test_batch_images = test_batch[0]
            test_batch_labels = np.squeeze(test_batch[1])
            test_song_ids[start_idx:end_idx] = test_batch[2].reshape([-1, 1])
            test_user_ids[start_idx:end_idx] = test_batch[3].reshape([-1, 1])
            test_classes[start_idx:end_idx, :] = test_batch_labels
            test_pred_prob[start_idx:end_idx, :],test_one_hot[start_idx:end_idx, :] = sess.run([model_output, one_hot],
                                                            feed_dict={x_input: test_batch_images,
                                                                       current_keep_prob: 1.0,
                                                                       train_phase: False})
        
        print("saving prediction to disk")
        np.savetxt(os.path.join(exp_dir, 'tracks_ids.txt'), test_song_ids, delimiter=',')
        np.savetxt(os.path.join(exp_dir, 'user_ids.txt'), test_user_ids, delimiter=',')
        np.savetxt(os.path.join(exp_dir, 'predictions.out'), test_pred_prob, delimiter=',')
        np.savetxt(os.path.join(exp_dir, 'test_output_one_hot.out'), test_one_hot, delimiter=',')
        np.savetxt(os.path.join(exp_dir, 'test_ground_truth_classes.txt'), test_classes, delimiter=',')
    

if __name__ == "__main__":
    main()