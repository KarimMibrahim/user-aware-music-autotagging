# General Imports
import os
import numpy as np
import pandas as pd
from time import strftime, localtime
import matplotlib.pyplot as plt
import seaborn as sn 

# Deep Learning
import tensorflow as tf


from sklearn.metrics import cohen_kappa_score,f1_score,accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, \
    hamming_loss

plt.rcParams.update({'font.size': 22})
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#TODO: fix directories
SOURCE_PATH = "/srv/workspace/research/user_based_contexts_tagging/"
SPECTROGRAMS_PATH = "/srv/workspace/research/user_based_contexts_tagging/dataset/"
OUTPUT_PATH = "/srv/workspace/research/user_based_contexts_tagging/experiments_results/"
EXTRA_OUTPUTS = "/srv/workspace/research/extra_experiment_results"

EXPERIMENTNAME = "classic_updated_dataset_long"
INPUT_SHAPE = (646, 96, 1)
LABELS_LIST = ['car', 'gym', 'happy', 'night', 'relax',
       'running', 'sad', 'summer', 'work', 'workout']

global_labels = pd.read_csv("/srv/workspace/research/user_based_contexts_tagging/GroundTruth/all_multipllized.csv")
train_partial = pd.read_csv("/srv/workspace/research/user_based_contexts_tagging/GroundTruth/train_single_multipllized.csv")
POS_WEIGHTS = len(train_partial)/train_partial.sum()[1:]
POS_WEIGHTS = [np.float32(x) for x in POS_WEIGHTS]

BATCH_SIZE = 32

from tensorflow.keras.backend import set_session


def limit_memory_usage(gpu_memory_fraction=0.1):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    set_session(tf.Session(config=config))


limit_memory_usage(0.3)

def dataset_from_csv(csv_path, **kwargs):
    """
        Load dataset from a csv file.
        kwargs are forwarded to the pandas.read_csv function.
    """
    df = pd.read_csv(csv_path, **kwargs)

    dataset = (
        tf.data.Dataset.from_tensor_slices(
            {
                key:df[key].values
                for key in df
            }
        )
    )
    return dataset

def set_tensor_shape(tensor, tensor_shape):
        """
            set shape for a tensor (not in place, as opposed to tf.set_shape)
        """
        tensor.set_shape(tensor_shape)
        return tensor

def check_tensor_shape(tensor_tf, target_shape):
    """
        Return a Tensorflow boolean graph that indicates whether sample[features_key] has the specified target shape
        Only check not None entries of target_shape.
    """
    res = tf.constant(True)
    for idx,target_length in enumerate(target_shape):
        if target_length:
            res = tf.logical_and(res, tf.equal(tf.constant(target_length), tf.shape(tensor_tf)[idx]))

    return res

def load_spectrogram(*args):
    """
        loads spectrogram with error tracking.
        args : song ID, path to dataset
        return:
            Features: numpy ndarray, computed features (if no error occured, otherwise: 0)
            Error: boolean, False if no error, True if an error was raised during features computation.
    """
    path = SPECTROGRAMS_PATH
    song_id, dummy_path = args
    try:
        spect = np.load(os.path.join(path, "mels" + str(song_id) + '.npz'))['arr_0']
        if (spect.shape != (1, 646, 96)):
            print("\n Error while computing features for" +  str(song_id) + '\n')
            return np.float32(0.0), True
        return spect, False
    except Exception as err:
        print("\n Error while computing features for " + str(song_id) + '\n')
        return np.float32(0.0), True

def load_spectrogram_tf(sample, identifier_key="song_id",
                        path="/my_data/MelSpectograms_top20/", device="/cpu:0",
                        features_key="features"):
    """
        wrap load_spectrogram into a tensorflow function.
    """
    with tf.device(device):
        input_args = [sample[identifier_key], tf.constant(path)]
        res = tf.py_func(load_spectrogram,
                         input_args,
                         (tf.float32, tf.bool),
                         stateful=False),
        spectrogram, error = res[0]

        res = dict(list(sample.items()) + [(features_key, spectrogram), ("error", error)])
        return res


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


def get_weights(shape):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return w


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial)
    return b

def conv_2d(x, W, name=""):
    return tf.nn.conv2d(x, W, [1, 1, 1, 1], padding="SAME", name=name)


def max_pooling(x, shape, name=""):
    return tf.nn.max_pool(x, shape, strides=[1, 2, 2, 1], padding="SAME", name=name)

def conv_layer_with_relu(input, shape, name=""):
    W = get_weights(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv_2d(input, W, name) + b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = get_weights([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b


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


def plot_loss_acuracy(epoch_losses_history, epoch_accurcies_history, val_losses_history, val_accuracies_history, path):
    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 10))
    plt.plot(epoch_accurcies_history)
    plt.plot(val_accuracies_history)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(path, "model_accuracy.png"))
    plt.savefig(os.path.join(path, "model_accuracy.pdf"), format='pdf')
    # Plot training & validation loss values
    plt.figure(figsize=(10, 10))
    plt.plot(epoch_losses_history)
    plt.plot(val_losses_history)
    plt.title('Model loss (Cross Entropy without weighting)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(path, "model_loss.png"))
    plt.savefig(os.path.join(path, "model_loss.pdf"), format='pdf')


def plot_output_coocurances(model_output_rounded, output_path, LABELS_LIST):
    # Getting coocuarances
    test_pred_df = pd.DataFrame(model_output_rounded, columns=LABELS_LIST)
    coocurrances = pd.DataFrame(columns=test_pred_df.columns)
    for column in test_pred_df.columns:
        coocurrances[column] = test_pred_df[test_pred_df[column] == 1].sum()
    coocurrances = coocurrances.T
    # Plotting coocurances
    plt.figure(figsize=(30, 30));
    sn.set(font_scale=2)  # for label size
    cmap = 'PuRd'
    plt.axes([.1, .1, .8, .7])
    plt.figtext(.5, .83, 'Number of track coocurances in model output', fontsize=34, ha='center')
    sn.heatmap(coocurrances, annot=True, annot_kws={"size": 24}, fmt='.0f', cmap=cmap);
    plt.savefig(output_path + ".pdf", format="pdf")
    plt.savefig(output_path + ".png")


def plot_false_netgatives_confusion_matrix(model_output_rounded, groundtruth, output_path, LABELS_LIST):
    # Getting false negatives coocuarances
    test_pred_df = pd.DataFrame(model_output_rounded, columns=LABELS_LIST)
    test_classes_df = pd.DataFrame(groundtruth, columns=LABELS_LIST)
    FN_coocurrances = pd.DataFrame(columns=test_pred_df.columns)
    for column in test_pred_df.columns:
        FN_coocurrances[column] = test_pred_df[[negative_prediction and positive_sample
                                                for negative_prediction, positive_sample in
                                                zip(test_pred_df[column] == 0, test_classes_df[column] == 1)]].sum()
    FN_coocurrances = FN_coocurrances.T
    # Plotting coocurances
    plt.figure(figsize=(30, 30));
    sn.set(font_scale=2)  # for label size
    cmap = 'PuRd'
    plt.axes([.1, .1, .8, .7])
    plt.figtext(.5, .83, 'False negatives confusion matrix', fontsize=34, ha='center')
    sn.heatmap(FN_coocurrances, annot=True, annot_kws={"size": 24}, fmt='.0f', cmap=cmap);
    plt.savefig(output_path + ".pdf", format="pdf")
    plt.savefig(output_path + ".png")


def plot_true_poisitve_vs_all_positives(model_output_rounded, groundtruth, output_path, LABELS_LIST):
    # Creating a plot of true positives vs all positives
    true_positives_perclass = sum((model_output_rounded == groundtruth) * (groundtruth == 1))
    true_positives_df = pd.DataFrame(columns=LABELS_LIST)
    true_positives_df.index.astype(str, copy=False)
    true_positives_df.loc[0] = true_positives_perclass
    percentage_of_positives_perclass = sum(groundtruth)
    true_positives_df.loc[1] = percentage_of_positives_perclass
    true_positives_df.index = ['True Positives', 'Positive Samples']
    true_positives_ratio_perclass = sum((model_output_rounded == groundtruth) * (groundtruth == 1)) / sum(groundtruth)
    # Plot the figure
    labels = [label + " (" + "{:.1f}".format(true_positives_ratio_perclass[idx] * 100) + "%) " for idx, label in
              enumerate(LABELS_LIST)]
    true_positives_df.columns = labels
    true_positives_df.T.plot.bar(figsize=(32, 22), fontsize=28)
    plt.xticks(rotation=45)
    plt.title(
        "Number of true positive per class compared to the total number of positive samples \n Average true positive rate: " + "{:.2f}".format(
            true_positives_ratio_perclass.mean()))
    plt.savefig(output_path + ".pdf", format="pdf")
    plt.savefig(output_path + ".png")


def create_analysis_report(model_output, groundtruth, output_path, LABELS_LIST, validation_output=None,
                           validation_groundtruth=None):
    """
    Create a report of all the different evaluation metrics, including optimizing the threshold with the validation set
    if it is passed in the parameters
    """
    # Round the probabilities at 0.5
    model_output_rounded = np.round(model_output)
    model_output_rounded = np.clip(model_output_rounded, 0, 1)
    # Create a dataframe where we keep all the evaluations, starting by prediction accuracy
    accuracies_perclass = sum(model_output_rounded == groundtruth) / len(groundtruth)
    results_df = pd.DataFrame(columns=LABELS_LIST)
    results_df.index.astype(str, copy=False)
    percentage_of_positives_perclass = sum(groundtruth) / len(groundtruth)
    results_df.loc[0] = percentage_of_positives_perclass
    results_df.loc[1] = accuracies_perclass
    results_df.index = ['Ratio of positive samples', 'Model accuracy']

    # plot the accuracies per class
    results_df.T.plot.bar(figsize=(22, 12), fontsize=18)
    plt.title('Model accuracy vs the ratio of positive samples per class')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_path, "accuracies_vs_positiveRate.pdf"), format="pdf")
    plt.savefig(os.path.join(output_path, "accuracies_vs_positiveRate.png"))

    # Getting the true positive rate perclass
    true_positives_ratio_perclass = sum((model_output_rounded == groundtruth) * (groundtruth == 1)) / sum(groundtruth)
    results_df.loc[2] = true_positives_ratio_perclass
    # Get true negative ratio
    true_negative_ratio_perclass = sum((model_output_rounded == groundtruth)
                                       * (groundtruth == 0)) / (len(groundtruth) - sum(groundtruth))
    results_df.loc[3] = true_negative_ratio_perclass
    # compute additional metrics (AUC,f1,recall,precision)
    auc_roc_per_label = roc_auc_score(groundtruth, model_output, average=None)
    precision_perlabel = precision_score(groundtruth, model_output_rounded, average=None)
    recall_perlabel = recall_score(groundtruth, model_output_rounded, average=None)
    f1_perlabel = f1_score(groundtruth, model_output_rounded, average=None)
    kappa_perlabel = [cohen_kappa_score(groundtruth[:, x], model_output_rounded[:, x]) for x in range(len(LABELS_LIST))]
    results_df = results_df.append(
        pd.DataFrame([auc_roc_per_label,recall_perlabel, precision_perlabel, f1_perlabel, kappa_perlabel], columns=LABELS_LIST))
    results_df.index = ['Ratio of positive samples', 'Model accuracy', 'True positives ratio',
                        'True negatives ratio', "AUC", "Recall", "Precision", "f1-score", "Kappa score"]

    # Creating evaluation plots
    plot_true_poisitve_vs_all_positives(model_output_rounded, groundtruth,
                                        os.path.join(output_path, 'TruePositive_vs_allPositives'), LABELS_LIST)
    plot_output_coocurances(model_output_rounded, os.path.join(output_path, 'output_coocurances'), LABELS_LIST)
    plot_false_netgatives_confusion_matrix(model_output_rounded, groundtruth,
                                           os.path.join(output_path, 'false_negative_coocurances'), LABELS_LIST)

    # Adjusting threshold based on validation set
    if (validation_groundtruth is not None and validation_output is not None):
        np.savetxt(os.path.join(output_path, 'validation_predictions.out'), validation_output, delimiter=',')
        np.savetxt(os.path.join(output_path, 'valid_ground_truth_classes.txt'), validation_groundtruth, delimiter=',')
        thresholds = np.arange(0, 1, 0.01)
        f1_array = np.zeros((len(LABELS_LIST), len(thresholds)))
        for idx, label in enumerate(LABELS_LIST):
            f1_array[idx, :] = [
                f1_score(validation_groundtruth[:, idx], np.clip(np.round(validation_output[:, idx] - threshold + 0.5), 0, 1))
                for threshold in thresholds]
        threshold_arg = np.argmax(f1_array, axis=1)
        threshold_per_class = thresholds[threshold_arg]

        # plot the f1 score across thresholds
        plt.figure(figsize=(20, 20))
        for idx, x in enumerate(LABELS_LIST):
            plt.plot(thresholds, f1_array[idx, :], linewidth=5)
        plt.legend(LABELS_LIST, loc='best')
        plt.title("F1 Score vs different prediction threshold values for each class")
        plt.savefig(os.path.join(output_path, "f1_score_vs_thresholds.pdf"), format="pdf")
        plt.savefig(os.path.join(output_path, "f1_score_vs_thresholds.png"))

        # Applying thresholds optimized per class
        model_output_rounded = np.zeros_like(model_output)
        for idx, label in enumerate(LABELS_LIST):
            model_output_rounded[:, idx] = np.clip(np.round(model_output[:, idx] - threshold_per_class[idx] + 0.5), 0, 1)

        accuracies_perclass = sum(model_output_rounded == groundtruth) / len(groundtruth)
        # Getting the true positive rate perclass
        true_positives_ratio_perclass = sum((model_output_rounded == groundtruth) * (groundtruth == 1)) / sum(
            groundtruth)
        # Get true negative ratio
        true_negative_ratio_perclass = sum((model_output_rounded == groundtruth)
                                           * (groundtruth == 0)) / (len(groundtruth) - sum(groundtruth))
        results_df = results_df.append(
            pd.DataFrame([accuracies_perclass, true_positives_ratio_perclass,
                          true_negative_ratio_perclass], columns=LABELS_LIST))
        # compute additional metrics (AUC,f1,recall,precision)
        auc_roc_per_label = roc_auc_score(groundtruth, model_output, average=None)
        precision_perlabel = precision_score(groundtruth, model_output_rounded, average=None)
        recall_perlabel = recall_score(groundtruth, model_output_rounded, average=None)
        f1_perlabel = f1_score(groundtruth, model_output_rounded, average=None)
        kappa_perlabel = [cohen_kappa_score(groundtruth[:, x], model_output_rounded[:, x]) for x in
                          range(len(LABELS_LIST))]
        results_df = results_df.append(
            pd.DataFrame([auc_roc_per_label, precision_perlabel, recall_perlabel, f1_perlabel,kappa_perlabel],
                         columns=LABELS_LIST))
        results_df.index = ['Ratio of positive samples', 'Model accuracy', 'True positives ratio',
                            'True negatives ratio', "AUC", "Precision", "Recall", "f1-score",  "Kappa score",
                            'Optimized model accuracy', 'Optimized true positives ratio',
                            'Optimized true negatives ratio', "Optimized AUC",
                            "Optimized precision", "Optimized recall", "Optimized f1-score",  "Optimized Kappa score"]

        # Creating evaluation plots
        plot_true_poisitve_vs_all_positives(model_output_rounded, groundtruth,
                                            os.path.join(output_path, 'TruePositive_vs_allPositives[optimized]'),
                                            LABELS_LIST)
        plot_output_coocurances(model_output_rounded, os.path.join(output_path, 'output_coocurances[optimized]'),
                                LABELS_LIST)
        plot_false_netgatives_confusion_matrix(model_output_rounded, groundtruth,
                                               os.path.join(output_path, 'false_negative_coocurances[optimized]'),
                                               LABELS_LIST)
    results_df['average'] = results_df.mean(numeric_only=True, axis=1)
    results_df.T.to_csv(os.path.join(output_path, "results_report.csv"), float_format="%.2f")
    return results_df



def main():
    print("Current Experiment: " + EXPERIMENTNAME + "\n\n\n")
    # Loading datasets
    # TODO: fix directories
    training_dataset = get_training_dataset(os.path.join(SOURCE_PATH, "GroundTruth/train_single_multipllized.csv"))
    val_dataset = get_validation_dataset(os.path.join(SOURCE_PATH, "GroundTruth/validation_single_multipllized.csv"))

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
                if(batch_counter % 100 ==0):
                    print(batch_counter)
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

        test_labels = pd.read_csv(os.path.join(SOURCE_PATH, "GroundTruth/test_single_multipllized.csv"))
        test_dataset = get_dataset(os.path.join(SOURCE_PATH, "GroundTruth/test_single_multipllized.csv"))
        test_classes = np.zeros_like(test_labels.iloc[:, 1:].values, dtype=float)
        # test_images, test_classes = load_test_set_raw(test_split)

        TEST_NUM_STEPS = int(np.floor((len(test_classes) / 32)))
        # split_size = int(len(test_classes) / TEST_NUM_STEPS)
        test_pred_prob = np.zeros_like(test_classes, dtype=float)
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
            test_pred_prob[start_idx:end_idx, :] = sess.run(model_output,
                                                            feed_dict={x_input: test_batch_images,
                                                                       current_keep_prob: 1.0,
                                                                       train_phase: False})

        np.savetxt(os.path.join(exp_dir, 'tracks_ids.txt'), test_song_ids, delimiter=',')
        np.savetxt(os.path.join(exp_dir, 'user_ids.txt'), test_user_ids, delimiter=',')

        accuracy_out, auc_roc, hamming_error = evaluate_model(test_pred_prob, test_classes,
                                                              saving_path=exp_dir,
                                                              evaluation_file_path= \
                                                                  os.path.join(exp_dir, "evaluation_results.txt"))
        results = create_analysis_report(test_pred_prob, test_classes, exp_dir, LABELS_LIST)

    # Plot and save losses
    plot_loss_acuracy(epoch_losses_history, epoch_accurcies_history, val_losses_history, val_accuracies_history,
                      exp_dir)


if __name__ == "__main__":
    main()
