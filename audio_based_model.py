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

from sklearn.metrics import cohen_kappa_score,f1_score,accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, \
    hamming_loss

plt.rcParams.update({'font.size': 22})
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# [TODO] Edit directories to match machine
SOURCE_PATH = "/src_code/repo/"
OUTPUT_PATH = "/src_code/repo/experiments_results/"
EXTRA_OUTPUTS = "/src_code/repo/extra_experiment_results"

EXPERIMENTNAME = "audio_system_multilabel"
INPUT_SHAPE = (646, 96, 1)
LABELS_LIST = ['car', 'gym', 'happy', 'night', 'relax',
       'running', 'sad', 'summer', 'work', 'workout']

global_labels = pd.read_csv("/src_code/repo/GroundTruth/multilabel_all.csv")
train_partial = pd.read_csv("/src_code/repo/GroundTruth/train_multilabel.csv")
POS_WEIGHTS = len(train_partial)/train_partial.sum()[1:]
POS_WEIGHTS = [np.float32(x) for x in POS_WEIGHTS]

BATCH_SIZE = 32

limit_memory_usage(0.3)

# Dataset pipelines
def get_labels_py(song_id):
    labels = global_labels[global_labels.song_id == song_id]
    labels = labels.iloc[:, 1:].values.flatten() # TODO: fix this shift in dataframe columns when read
    labels = labels.astype(np.float32)
    return labels


def tf_get_labels_py(sample, device="/cpu:0"):
    with tf.device(device):
        input_args = [sample["song_id"]]
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
    sample["features"], sample["binary_label"]))

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

def evaluate_model(test_pred_prob, test_classes, saving_path, evaluation_file_path):
    """
    Evaluates a given model using accuracy, area under curve and hamming loss
    :param model: model to be evaluated
    :param spectrograms: the test set spectrograms as an np.array
    :param test_classes: the ground truth labels
    :return: accuracy, auc_roc, hamming_error
    """
    test_pred = np.round(test_pred_prob)
    # Accuracy
    accuracy = 100 * accuracy_score(test_classes, test_pred)
    print("Exact match accuracy is: " + str(accuracy) + "%")
    # Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    auc_roc = roc_auc_score(test_classes, test_pred_prob)
    print("Macro Area Under the Curve (AUC) is: " + str(auc_roc))
    auc_roc_micro = roc_auc_score(test_classes, test_pred_prob, average="micro")
    print("Micro Area Under the Curve (AUC) is: " + str(auc_roc_micro))
    auc_roc_weighted = roc_auc_score(test_classes, test_pred_prob, average="weighted")
    print("Weighted Area Under the Curve (AUC) is: " + str(auc_roc_weighted))
    # Hamming loss is the fraction of labels that are incorrectly predicted.
    hamming_error = hamming_loss(test_classes, test_pred)
    print("Hamming Loss (ratio of incorrect tags) is: " + str(hamming_error))
    with open(evaluation_file_path, "w") as f:
        f.write("Exact match accuracy is: " + str(accuracy) + "%\n" + "Area Under the Curve (AUC) is: " + str(auc_roc)
                + "\nMicro AUC is:" + str(auc_roc_micro) + "\nWeighted AUC is:" + str(auc_roc_weighted)
                + "\nHamming Loss (ratio of incorrect tags) is: " + str(hamming_error))
    print("saving prediction to disk")
    np.savetxt(os.path.join(saving_path, 'predictions.out'), test_pred_prob, delimiter=',')
    np.savetxt(os.path.join(saving_path, 'test_ground_truth_classes.txt'), test_classes, delimiter=',')
    return accuracy, auc_roc, hamming_error

def main():
    print("Current Experiment: " + EXPERIMENTNAME + "\n\n\n")
    # Loading datasets
    # TODO: fix directories
    training_dataset = get_dataset(os.path.join(SOURCE_PATH, "GroundTruth/train_multilabel.csv"))
    val_dataset = get_dataset(os.path.join(SOURCE_PATH, "GroundTruth/validation_multilabel.csv"))

    # Setting up model
    y = tf.placeholder(tf.float32, [None, len(LABELS_LIST)], name="true_labels")
    x_input = tf.placeholder(tf.float32, [None, 646, 96, 1], name="input")
    current_keep_prob = tf.placeholder(tf.float32, name="dropout_rate")
    weights = tf.constant(POS_WEIGHTS)
    train_phase = tf.placeholder(tf.bool, name="is_training")
    logits, model_output = get_model(x_input,current_keep_prob, train_phase)

    # Defining loss and metrics
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y,logits,pos_weight=weights))

    # Learning rate decay
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=1000,
                                               decay_rate=0.95, staircase=True)
    '''
    These following lines are needed for batch normalization to work properly
    check https://timodenk.com/blog/tensorflow-batch-normalization/
    '''
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=global_step)
    correct_prediction = tf.equal(tf.round(model_output), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Adding tensorboard summaries
    tf.summary.scalar('Original cross_entropy', loss)
    tf.summary.scalar('Accuracy', accuracy)
    # Merge all the summaries
    merged = tf.summary.merge_all()

    # Setting up dataset iterator
    training_iterator = training_dataset.make_one_shot_iterator()
    training_next_element = training_iterator.get_next()
    validation_iterator = val_dataset.make_one_shot_iterator()
    validation_next_element = validation_iterator.get_next()

    ## Setting up early stopping parameters
    # Best validation accuracy seen so far.
    best_validation_loss = 10e6  # Just some large number before storing the first validation loss
    # Iteration-number for last improvement to validation accuracy.
    last_improvement = 0
    # Stop optimization if no improvement found in this many iterations.
    min_epochs_for_early_stop = 10

    # Training paramaeters
    TRAINING_STEPS = 3125
    VALIDATION_STEPS = 927
    NUM_EPOCHS = 60

    # Setting up saving directory
    experiment_name = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    exp_dir = os.path.join(OUTPUT_PATH, EXPERIMENTNAME, experiment_name)
    extra_exp_dir =  os.path.join(EXTRA_OUTPUTS, EXPERIMENTNAME, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    epoch_losses_history, epoch_accurcies_history, val_losses_history, val_accuracies_history = [], [], [], []
    my_loss_history, my_loss_val_history = [], []
    with tf.Session() as sess:
        # Write summaries to LOG_DIR -- used by TensorBoard
        train_writer = tf.summary.FileWriter(extra_exp_dir + '/tensorboard/train', graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(extra_exp_dir + '/tensorboard/test', graph=tf.get_default_graph())
        print("Execute the following in a terminal:\n" + "tensorboard --logdir=" + extra_exp_dir)
        sess.run(tf.global_variables_initializer())
        for epoch in range(NUM_EPOCHS):
            batch_loss, batch_accuracy = np.zeros([TRAINING_STEPS, 1]), np.zeros([TRAINING_STEPS, 1])
            batch_my_loss, val_my_loss = np.zeros([TRAINING_STEPS, 1]), np.zeros([VALIDATION_STEPS, 1])
            val_accuracies, val_losses = np.zeros([VALIDATION_STEPS, 1]), np.zeros([VALIDATION_STEPS, 1])
            for batch_counter in range(TRAINING_STEPS):
                batch = sess.run(training_next_element)
                batch_labels = np.squeeze(batch[1])
                summary, batch_loss[batch_counter], batch_accuracy[batch_counter], _ = sess.run(
                    [merged, loss, accuracy, train_step],
                    feed_dict={current_keep_prob: 0.3, x_input: batch[0], y: batch_labels,
                               train_phase: True})
            print("Epoch #{}".format(epoch + 1), "Loss: {:.4f}".format(np.mean(batch_loss)),
                  "My_loss: {:.4f}".format(np.mean(batch_my_loss)),
                  "accuracy: {:.4f}".format(np.mean(batch_accuracy)))
            epoch_losses_history.append(np.mean(batch_loss));
            epoch_accurcies_history.append(np.mean(batch_accuracy))
            my_loss_history.append(np.mean(batch_my_loss))
            # Add to summaries
            train_writer.add_summary(summary, epoch)

            for validation_batch in range(VALIDATION_STEPS):
                val_batch = sess.run(validation_next_element)
                summary, val_losses[validation_batch], val_accuracies[validation_batch], = sess.run(
                    [merged, loss, accuracy],
                    feed_dict={
                        x_input: val_batch[0],
                        y: np.squeeze(val_batch[1]),
                        current_keep_prob: 1.0,
                        train_phase: False})
            print("validation Loss : {:.4f}".format(np.mean(val_losses)),
                  "validation accuracy: {:.4f}".format(np.mean(val_accuracies)))
            val_losses_history.append(np.mean(val_losses));
            val_accuracies_history.append(np.mean(val_accuracies))
            my_loss_val_history.append(np.mean(val_my_loss))
            test_writer.add_summary(summary, epoch)

            # If validation loss is an improvement over best-known.
            if np.mean(val_losses) < best_validation_loss:
                # Update the best-known validation accuracy.
                best_validation_loss = np.mean(val_losses)

                # Set the iteration for the last improvement to current.
                last_improvement = epoch

                # Save all variables of the TensorFlow graph to file.
                save_path = saver.save(sess, os.path.join(extra_exp_dir, "best_validation.ckpt"))
                # print("Model with best validation saved in path: %s" % save_path)

            # If no improvement found in the required number of iterations.
            if epoch - last_improvement > min_epochs_for_early_stop:
                print("No improvement found in a last 10 epochs, stopping optimization.")
                # Break out from the for-loop.
                break

        save_path = saver.save(sess, os.path.join(extra_exp_dir, "last_epoch.ckpt"))
        print("Last iteration model saved in path: %s" % save_path)

        # Loading model with best validation
        saver.restore(sess, os.path.join(extra_exp_dir, "best_validation.ckpt"))
        print("Model with best validation restored before testing.")

        test_labels = pd.read_csv(os.path.join(SOURCE_PATH, "GroundTruth/test_multilabel.csv"))
        test_dataset = get_dataset(os.path.join(SOURCE_PATH, "GroundTruth/test_multilabel.csv"), shuffle = True)
        test_classes = np.zeros_like(test_labels.iloc[:, 1:].values, dtype=float)
        # test_images, test_classes = load_test_set_raw(test_split)

        TEST_NUM_STEPS = int(np.floor((len(test_classes) / 32)))
        # split_size = int(len(test_classes) / TEST_NUM_STEPS)
        test_pred_prob = np.zeros_like(test_classes, dtype=float)
        test_iterator = test_dataset.make_one_shot_iterator()
        test_next_element = test_iterator.get_next()

        for test_batch_counter in range(TEST_NUM_STEPS):
            start_idx = (test_batch_counter * BATCH_SIZE)
            end_idx = (test_batch_counter * BATCH_SIZE) + BATCH_SIZE
            test_batch = sess.run(test_next_element)
            test_batch_images = test_batch[0]
            test_batch_labels = np.squeeze(test_batch[1])
            test_classes[start_idx:end_idx, :] = test_batch_labels
            test_pred_prob[start_idx:end_idx, :] = sess.run(model_output,
                                                            feed_dict={x_input: test_batch_images,
                                                                       current_keep_prob: 1.0,
                                                                       train_phase: False})
        
        accuracy_out, auc_roc, hamming_error = evaluate_model(test_pred_prob, test_classes,
                                                              saving_path=exp_dir,
                                                              evaluation_file_path= \
                                                                  os.path.join(exp_dir, "evaluation_results.txt"))
        model_output_rounded = np.round(test_pred_prob)
        results = create_analysis_report(test_pred_prob,model_output_rounded, test_classes, exp_dir, LABELS_LIST)

    # Plot and save losses
    plot_loss_acuracy(epoch_losses_history, epoch_accurcies_history, val_losses_history, val_accuracies_history,
                      exp_dir)


if __name__ == "__main__":
    main()
