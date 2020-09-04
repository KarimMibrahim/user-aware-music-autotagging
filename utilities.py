# General imports
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

pd.option_context('display.float_format', '{:0.2f}'.format)
sn.set(font_scale=2)  # for label size

SOURCE_PATH = "/home/karim/Documents/research/sourceCode/context_classification_cnn/"
SPECTROGRAMS_PATH = "/home/karim/Documents/BalancedDatasetDeezer/mel_specs/mel_specs/"
OUTPUT_PATH = "/home/karim/Documents/research/experiments_results"

SOURCE_PATH = "/srv/workspace/research/context_classification_cnn/"
SPECTROGRAMS_PATH = "/srv/workspace/research/balanceddata/mel_specs/"
OUTPUT_PATH = "/srv/workspace/research/balanceddata/experiments_results/"

LABELS_LIST = ['car', 'chill', 'club', 'dance', 'gym', 'happy', 'night', 'party', 'relax', 'running',
               'sad', 'sleep', 'summer', 'work', 'workout']

def tf_idf(track_count,hot_encoded, number_of_classes = 15):
    #hot_encoded = pd.read_csv("/home/karim/Documents/BalancedDatasetDeezer/GroundTruth/balanced_ground_truth_hot_vector.csv")
    #track_count = pd.read_csv("/home/karim/Documents/BalancedDatasetDeezer/GroundTruth/balanced_ground_truth_track_count.csv")
    track_count = track_count.set_index("song_id")
    hot_encoded = hot_encoded.set_index("song_id")
    class_total_tracks = track_count.sum()
    class_count_per_sample = hot_encoded.sum(axis=1)
    track_tf = track_count.copy()
    # compute tf (number of track occurances in a context / total number of occurances in this context)
    track_tf = track_count.div(class_total_tracks, axis=1)
    # compute idf (number of contexts / number of positive context in this class)
    track_idf = np.log(number_of_classes / class_count_per_sample)
    track_tf_idf = track_tf.copy()
    track_tf_idf = track_tf.mul(track_idf, axis=0)
    # plotting
    # track_tf_idf[track_tf_idf>0].boxplot(figsize=(20,20),rot=60, fontsize=30)
    # Normalize the values
    track_tf_idf = (track_tf_idf[track_tf_idf>0] - track_tf_idf[track_tf_idf>0].mean(axis=0)) / track_tf_idf[track_tf_idf>0].std(axis=0) # zero mean unit variance computed only on positive values (ignore zeros) [[Note must replce NaNs with zero later]]
    track_tf_idf = track_tf_idf + 1  # adding one to move mean to one
    track_tf_idf = track_tf_idf.clip(lower = 0 , upper = 2) # clip to a max of 2 and min of 0
    track_tf_idf = track_tf_idf.fillna(0) # Replace NaN with zero for a stable computing of loss
    # track_tf_idf[track_tf_idf>0].boxplot(figsize=(20,20),rot=60, fontsize=30) # plotting
    # Choose between combinations of different normalizations below instead of the above one
    # track_tf_idf. = (track_tf_idf - track_tf_idf.min(axis=0)) / (track_tf_idf.max(axis=0) - track_tf_idf.min(axis=0)) # between zero and 1
    # track_tf_idf = (track_tf_idf - track_tf_idf.mean(axis=0)) / track_tf_idf.std(axis=0) # zero mean unit variance
    #track_tf_idf = track_tf_idf / track_tf_idf.max(axis=0)
    #track_tf_idf.to_csv("/home/karim/Documents/BalancedDatasetDeezer/GroundTruth/positive_weights.csv")
    return track_tf_idf

def negative_labeles_probabilities(hot_encoded):
    # count the number of times a combination has appeared with the negative label as 1 / the total number of
    # occurances of that combination without the negative label
    negative_weights = np.zeros([len(hot_encoded), len(LABELS_LIST)])
    for sample_idx in range(len(hot_encoded)):
        for label_idx in range(len(LABELS_LIST)):
            if hot_encoded.iloc[sample_idx, label_idx+1] == 1:
                negative_weights[sample_idx, label_idx] = 0
            else:
                temp_combination = hot_encoded.iloc[sample_idx,1:].copy()
                temp_combination[label_idx] = 1
                positive_samples = len(hot_encoded[(hot_encoded.iloc[:, 1:].values == temp_combination.values).all(axis = 1)])
                negative_samples = len(hot_encoded[(hot_encoded.iloc[:, 1:].values == hot_encoded.iloc[sample_idx, 1:].values).all(axis=1)])
                negative_weights[sample_idx, label_idx] = negative_samples / (positive_samples + negative_samples)
    negative_weights_df = pd.DataFrame(negative_weights, columns=LABELS_LIST)
    negative_weights_df["song_id"] = hot_encoded.song_id
    negative_weights_df = negative_weights_df[["song_id"] + LABELS_LIST]
    #negative_weights_df.to_csv("/home/karim/Documents/BalancedDatasetDeezer/GroundTruth/negative_weights.csv",index=False)
    return negative_weights_df

def negative_labeles_probabilities_ignoring_zeros(hot_encoded):
    # count the number of times a combination has appeared with the negative label as 1 / the total number of
    # occurances of that combination without the negative label
    negative_weights = np.zeros([len(hot_encoded), len(LABELS_LIST)])
    for sample_idx in range(len(hot_encoded)):
        for label_idx in range(len(LABELS_LIST)):
            if hot_encoded.iloc[sample_idx, label_idx+1] == 1:
                negative_weights[sample_idx, label_idx] = 0
            else:
                temp_combination = hot_encoded.iloc[sample_idx,1:].copy()
                temp_combination[label_idx] = 1
                # Compare only columns that are equal to 1, and count number of matches
                # adding one to skip the song_id column, which exists in the hot_encoded dataframe
                positive_columns = np.where(temp_combination.values == 1)[0] + 1
                positive_samples = len(hot_encoded[(hot_encoded.iloc[:, positive_columns].values == 1).all(axis = 1)])
                # Count occurances with the negative sample
                temp_combination[label_idx] = 0
                positive_columns = np.where(temp_combination.values == 1)[0] + 1
                total_occurances_of_pattern = len(hot_encoded[(hot_encoded.iloc[:, positive_columns].values == 1).all(axis = 1)])
                negative_weights[sample_idx, label_idx] = (total_occurances_of_pattern - positive_samples) / total_occurances_of_pattern
    negative_weights_df = pd.DataFrame(negative_weights, columns=LABELS_LIST)
    negative_weights_df["song_id"] = hot_encoded.song_id
    negative_weights_df = negative_weights_df[["song_id"] + LABELS_LIST]
    #negative_weights_df.to_csv("/home/karim/Documents/BalancedDatasetDeezer/GroundTruth/negative_weights.csv",index=False)
    return negative_weights_df

def negative_labels_probabilities_pairwise(hot_encoded):
    # count the number of times a combination has appeared with the negative label as 1 / the total number of
    # occurances of that combination without the negative label
    negative_weights = np.ones([len(hot_encoded), len(LABELS_LIST)])
    for sample_idx in range(len(hot_encoded)):
        for label_idx in range(len(LABELS_LIST)):
            if hot_encoded.iloc[sample_idx, label_idx+1] == 1:
                negative_weights[sample_idx, label_idx] = 0
            else:
                for other_labels_index in range(len(LABELS_LIST)):
                    # Iterate through all other labels for each label_index
                    # check patterns if the other label == 1 (weight is #times occured as 0 / number of times occured as 0 or 1 for the target label)
                    if (other_labels_index != label_idx):
                        if (hot_encoded.iloc[sample_idx, other_labels_index+1] == 1):
                            positive_occurances = len(hot_encoded[(hot_encoded.iloc[:, [label_idx+1,other_labels_index+1]].values == [1,1]).all(axis = 1)])
                            negative_occurances = len(hot_encoded[(hot_encoded.iloc[:, [label_idx+1,other_labels_index+1]].values == [0,1]).all(axis = 1)])
                            weight = negative_occurances / (negative_occurances + positive_occurances)
                    # Check the patterns if the other labels == 0
                        if (hot_encoded.iloc[sample_idx, other_labels_index+1] == 0):
                            positive_occurances = len(hot_encoded[(hot_encoded.iloc[:, [label_idx+1,other_labels_index+1]].values == [1,0]).all(axis = 1)])
                            negative_occurances = len(hot_encoded[(hot_encoded.iloc[:, [label_idx+1,other_labels_index+1]].values == [0,0]).all(axis = 1)])
                            weight = negative_occurances / (negative_occurances + positive_occurances)
                        negative_weights[sample_idx, label_idx] += weight
                negative_weights[sample_idx, label_idx] /= (len(LABELS_LIST) - 1)
    negative_weights_df = pd.DataFrame(negative_weights, columns=LABELS_LIST)
    negative_weights_df["song_id"] = hot_encoded.song_id
    negative_weights_df = negative_weights_df[["song_id"] + LABELS_LIST]
    #negative_weights_df.to_csv("/home/karim/Documents/BalancedDatasetDeezer/GroundTruth/negative_weights_paris.csv",index=False)
    return negative_weights_df

def load_old_test_set_raw(LOADING_PATH=os.path.join(SOURCE_PATH, "GroundTruth/"),
                          SPECTROGRAM_PATH="/home/karim/Documents/MelSpectograms_top20/"):
    # Loading testset groundtruth
    test_ground_truth = pd.read_csv(os.path.join(LOADING_PATH, "old_test_ground_truth[unbalanced].csv"))
    all_ground_truth = pd.read_pickle(os.path.join(LOADING_PATH, "old_ground_truth_hot_vector[unblanced].pkl"))
    all_ground_truth.drop(['playlists_count', 'train', 'shower', 'park', 'morning', 'training'], axis=1, inplace=True);
    all_ground_truth = all_ground_truth[all_ground_truth.song_id.isin(test_ground_truth.song_id)]
    test_ground_truth = test_ground_truth[test_ground_truth.song_id.isin(all_ground_truth.song_id)]
    all_ground_truth = all_ground_truth.set_index('song_id')
    all_ground_truth = all_ground_truth.loc[test_ground_truth.song_id]
    test_classes = all_ground_truth.values
    test_classes = test_classes.astype(int)

    spectrograms = np.zeros([len(test_ground_truth), 646, 96])
    songs_ID = np.zeros([len(test_ground_truth), 1])
    for idx, filename in enumerate(list(test_ground_truth.song_id)):
        try:
            spect = np.load(os.path.join(SPECTROGRAM_PATH, str(filename) + '.npz'))['feat']
        except:
            continue
        if (spect.shape == (1, 1292, 96)):
            spect = spect[:, 323: 323 + 646, :]
            spectrograms[idx] = spect
            songs_ID[idx] = filename
    spectrograms = np.expand_dims(spectrograms, axis=3)
    return spectrograms, test_classes


def load_validation_set_raw(LOADING_PATH=os.path.join(SOURCE_PATH, "GroundTruth/"),
                            SPECTROGRAM_PATH=SPECTROGRAMS_PATH):
    # Loading testset groundtruth
    test_ground_truth = pd.read_csv(os.path.join(LOADING_PATH, "validation_ground_truth.csv"))
    all_ground_truth = pd.read_csv(os.path.join(LOADING_PATH, "balanced_ground_truth_hot_vector.csv"))
    # all_ground_truth.drop("playlists_count", axis=1, inplace=True);
    all_ground_truth = all_ground_truth[all_ground_truth.song_id.isin(test_ground_truth.song_id)]
    all_ground_truth = all_ground_truth.set_index('song_id')
    all_ground_truth = all_ground_truth.loc[test_ground_truth.song_id]
    test_classes = all_ground_truth.values
    test_classes = test_classes.astype(int)

    spectrograms = np.zeros([len(test_ground_truth), 646, 96])
    songs_ID = np.zeros([len(test_ground_truth), 1])
    for idx, filename in enumerate(list(test_ground_truth.song_id)):
        try:
            spect = np.load(os.path.join(SPECTROGRAM_PATH, str(filename) + '.npz'))['arr_0']
        except:
            continue
        if (spect.shape == (1, 646, 96)):
            spectrograms[idx] = spect
            songs_ID[idx] = filename

    # Apply same transformation as trianing [ALWAYS DOUBLE CHECK TRAINING PARAMETERS]
    C = 100
    spectrograms = np.log(1 + C * spectrograms)

    spectrograms = np.expand_dims(spectrograms, axis=3)
    return spectrograms, test_classes

def mark_groups_for_samples(df, n_samples, extra_criterion):
    """
        Return groups, an array of size n_samples, marking the group to which each sample belongs
        The default group is -1 if extra_criterion is None
        If a criterion is given (artist or album), then this information is taken into account
    """
    groups = np.array([-1 for _ in range(n_samples)])
    if extra_criterion is None:
        return groups

    if extra_criterion == "artist":
        crit_col = "artist_id"
    elif extra_criterion == "album":
        crit_col = "releasegroupmbid"
    else:
        return groups

    gp = df.groupby(crit_col)
    i_key = 0
    for g_key in gp.groups:
        samples_idx_per_group = gp.groups[g_key].tolist()
        groups[samples_idx_per_group] = i_key
        i_key += 1
    return groups


def select_fold(index_label, desired_samples_per_label_per_fold, desired_samples_per_fold, random_state):
    """
        For a label, finds the fold where the next sample should be distributed
    """
    # Find the folds with the largest number of desired samples for this label
    largest_desired_label_samples = max(desired_samples_per_label_per_fold[:, index_label])
    folds_targeted = np.where(desired_samples_per_label_per_fold[:, index_label] == largest_desired_label_samples)[0]

    if len(folds_targeted) == 1:
        selected_fold = folds_targeted[0]
    else:
        # Break ties by considering the largest number of desired samples
        largest_desired_samples = max(desired_samples_per_fold[folds_targeted])
        folds_re_targeted = np.intersect1d(np.where(
            desired_samples_per_fold == largest_desired_samples)[0], folds_targeted)

        # If there is still a tie break it picking a random index
        if len(folds_re_targeted) == 1:
            selected_fold = folds_re_targeted[0]
        else:
            selected_fold = random_state.choice(folds_re_targeted)
    return selected_fold


def iterative_split(df, out_file, target, n_splits, extra_criterion=None, seed=None):
    """
        Implement iterative split algorithm
        df is the input data
        out_file is the output file containing the same data as the input plus a column about the fold
        n_splits the number of folds
        target is the target source for which the files are generated
        extra_criterion, an extra condition to be taken into account in the split such as the artist
    """
    print("Starting the iterative split")
    random_state = check_random_state(seed)

    mlb_target = MultiLabelBinarizer()
    M = mlb_target.fit_transform(df[target].str.split('\t'))

    n_samples = len(df)
    n_labels = len(mlb_target.classes_)

    # If the extra criterion is given create "groups", which shows to which group each sample belongs
    groups = mark_groups_for_samples(df, n_samples, extra_criterion)

    ratios = np.ones((1, n_splits)) / n_splits
    # Calculate the desired number of samples for each fold
    desired_samples_per_fold = ratios.T * n_samples

    # Calculate the desired number of samples of each label for each fold
    number_samples_per_label = np.asarray(M.sum(axis=0)).reshape((n_labels, 1))
    desired_samples_per_label_per_fold = np.dot(ratios.T, number_samples_per_label.T)  # shape: n_splits, n_samples

    seen = set()
    out_folds = np.array([-1 for _ in range(n_samples)])

    count_seen = 0
    print("Going through the samples")
    while n_samples > 0:
        # Find the index of the label with the fewest remaining examples
        valid_idx = np.where(number_samples_per_label > 0)[0]
        index_label = valid_idx[number_samples_per_label[valid_idx].argmin()]
        label = mlb_target.classes_[index_label]

        # Find the samples belonging to the label with the fewest remaining examples
        # second select all samples belonging to the selected label and remove the indices
        # of the samples which have been already seen
        all_label_indices = set(M[:, index_label].nonzero()[0])
        indices = all_label_indices - seen
        assert (len(indices) > 0)

        print(label, index_label, number_samples_per_label[index_label], len(indices))

        for i in indices:
            if i in seen:
                continue

            # Find the folds with the largest number of desired samples for this label
            selected_fold = select_fold(index_label, desired_samples_per_label_per_fold,
                                        desired_samples_per_fold, random_state)

            # put in this fold all the samples which belong to the same group
            idx_same_group = np.array([i])
            if groups[i] != -1:
                idx_same_group = np.where(groups == groups[i])[0]

            # Update the folds, the seen, the number of samples and desired_samples_per_fold
            out_folds[idx_same_group] = selected_fold
            seen.update(idx_same_group)
            count_seen += idx_same_group.size
            n_samples -= idx_same_group.size
            desired_samples_per_fold[selected_fold] -= idx_same_group.size

            # The sample may have multiple labels so update for all
            for idx in idx_same_group:
                all_labels = M[idx].nonzero()
                desired_samples_per_label_per_fold[selected_fold, all_labels] -= 1
                number_samples_per_label[all_labels] -= 1

    df['fold'] = out_folds
    df.drop("index", axis=1, inplace=True)
    print(count_seen, len(df))
    df.to_csv(out_file, sep=',', index=False)
    return df


def split_dataset(csv_path=os.path.join(SOURCE_PATH, "GroundTruth/ground_truth_single_label.csv"),
                  artists_csv_path=os.path.join(SOURCE_PATH, "GroundTruth/songs_artists.tsv"),
                  test_size=0.25, seed=0, save_csv=True, n_splits=4,
                  train_save_path=os.path.join(SOURCE_PATH, "GroundTruth/"),
                  test_save_path=os.path.join(SOURCE_PATH, "GroundTruth/"),
                  validation_save_path=os.path.join(SOURCE_PATH, "GroundTruth/"),
                  folds_save_path=os.path.join(SOURCE_PATH, "GroundTruth/ground_truth_folds.csv")):
    song_artist = pd.read_csv(artists_csv_path, delimiter='\t')
    groundtruth = pd.read_csv(csv_path)
    ground_truth_artist = groundtruth.merge(song_artist, on='song_id')
    ground_truth_artist = ground_truth_artist.drop_duplicates("song_id")
    ground_truth_artist = ground_truth_artist.reset_index()

    groundtruth_folds = iterative_split(df=ground_truth_artist, out_file=folds_save_path, target='label',
                                        n_splits=n_splits, extra_criterion='artist', seed=seed)
    test = groundtruth_folds[groundtruth_folds.fold == 0]
    train_validation_combined = groundtruth_folds[groundtruth_folds.fold.isin(np.arange(1, n_splits))]
    train, validation = train_test_split(train_validation_combined, test_size=0.1, random_state=seed)
    train.drop(["artist_id", "fold"], axis=1, inplace=True)
    test.drop(["artist_id", "fold"], axis=1, inplace=True)
    validation.drop(["artist_id", "fold"], axis=1, inplace=True)
    # train, test = train_test_split(train, test_size=test_size, random_state=seed)
    if save_csv:
        pd.DataFrame.to_csv(train, os.path.join(train_save_path, "train_ground_truth.csv"), index=False)
        pd.DataFrame.to_csv(validation, os.path.join(validation_save_path, "validation_ground_truth.csv"), index=False)
        pd.DataFrame.to_csv(test, os.path.join(test_save_path, "test_ground_truth.csv"), index=False)
    # Save data in binarized format as well
    mlb_target = MultiLabelBinarizer()
    M = mlb_target.fit_transform(test.label.str.split('\t'))
    Mdf = pd.DataFrame(M, columns=LABELS_LIST)
    test.reset_index(inplace=True, drop=True)
    test_binarized = pd.concat([test, Mdf], axis=1)
    test_binarized.drop(['label'], inplace=True, axis=1)
    # For validation
    mlb_target = MultiLabelBinarizer()
    M = mlb_target.fit_transform(validation.label.str.split('\t'))
    Mdf = pd.DataFrame(M, columns=LABELS_LIST)
    validation.reset_index(inplace=True, drop=True)
    validation_binarized = pd.concat([validation, Mdf], axis=1)
    validation_binarized.drop(['label'], inplace=True, axis=1)
    # for training
    mlb_target = MultiLabelBinarizer()
    M = mlb_target.fit_transform(train.label.str.split('\t'))
    Mdf = pd.DataFrame(M, columns=LABELS_LIST)
    train.reset_index(inplace=True, drop=True)
    train_binarized = pd.concat([train, Mdf], axis=1)
    train_binarized.drop(['label'], inplace=True, axis=1)
    if save_csv:
        pd.DataFrame.to_csv(test_binarized, os.path.join(test_save_path, "test_ground_truth_binarized.csv"),
                            index=False)
        pd.DataFrame.to_csv(validation_binarized,
                            os.path.join(validation_save_path, "validation_ground_truth_binarized.csv"), index=False)
        pd.DataFrame.to_csv(train_binarized, os.path.join(train_save_path, "train_ground_truth_binarized.csv"),
                            index=False)
    return train, validation, test



def load_predictions_groundtruth(predictions_path, groundtruth_path):
    test_pred_prob = np.loadtxt(predictions_path, delimiter=',')
    test_classes = np.loadtxt(groundtruth_path, delimiter=',')
    return test_pred_prob, test_classes


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


def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Print mean and standard deviation.
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()