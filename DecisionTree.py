import math
import numpy as np
import pandas as pd


def run_algorithm(data, num_folds, max_depth, min_sample):
    data = normalize_data(data)

    fold_splits = k_fold_cross_validation_split(data, num_folds)

    training_set_accuracies = []
    test_set_accuracies = []

    for i in range(num_folds):
        fold_splits_copy = list(fold_splits)
        test_set = fold_splits_copy.pop(i)
        training_set = pd.concat(fold_splits_copy)

        label_categories = training_set.iloc[:, -1].unique()

        decision_tree = generate_splits(training_set, label_categories, max_depth, 0, min_sample)
        print(decision_tree)
        training_accuracy_rate = evaluate_performance(decision_tree, training_set)
        test_accuracy_rate = evaluate_performance(decision_tree, test_set)

        print ("Training Accuracy is: " + str(training_accuracy_rate))
        print ("Test Accuracy is: " + str(test_accuracy_rate))

        training_set_accuracies.append(training_accuracy_rate)
        test_set_accuracies.append(test_accuracy_rate)

        matrix_values = get_confusion_matrix_values(decision_tree, test_set)
        print(matrix_values)

    mean_training_error = sum(training_set_accuracies) / float(len(training_set_accuracies))
    mean_test_error = sum(test_set_accuracies) / float(len(test_set_accuracies))

    print ("Average Training Accuracy is: " + str(mean_training_error))
    print ("Average Test Accuracy is: " + str(mean_test_error))

def normalize_data(data):
    for column in data.iloc[:, :data.shape[1] - 1].columns:
        min = np.amin(data[column])
        max = np.amax(data[column])
        for i in range(len(data[column])):
            value = (data[column][i] - min) / float(max)
            data.at[i, column] = value

    return data


def k_fold_cross_validation_split(data, num_folds):
    data_copy = data.copy()
    data_copy = data_copy.sample(frac=1)
    fold_splits = []

    fold_size = data.shape[0] / num_folds

    for fold in range(num_folds):
        index_offset = fold * fold_size
        df = pd.DataFrame(data_copy.iloc[index_offset:index_offset+fold_size])
        fold_splits.append(df)

    return fold_splits


def generate_splits(bucket_data, label_categories, max_depth, cur_depth, min_sample):
    bucket_labels = bucket_data.iloc[:, -1]

    current_entropy = calculate_entropy(bucket_labels, label_categories)

    if max_depth == cur_depth or current_entropy < 0.3 or len(bucket_labels) < min_sample:
        return {'classification': bucket_labels.value_counts().idxmax()}

    best_option = {}

    for column in bucket_data.iloc[:, :bucket_data.shape[1] - 2]:
        idx = 0
        denominator = len(bucket_data[column]) / 10
        for row_value in bucket_data[column].sort_values():
            idx += 1
            if denominator > 0 and idx % denominator != 0: # Limit to 100 per column
                continue

            left = bucket_data.loc[bucket_data[column] < row_value]
            right = bucket_data.loc[bucket_data[column] >= row_value]

            left_labels = left.iloc[:, -1]
            left_entropy = calculate_entropy(left_labels, label_categories)
            left_proportion = left.shape[0] / float(bucket_data.shape[0])

            right_labels = right.iloc[:, -1]
            right_entropy = calculate_entropy(right_labels, label_categories)
            right_proportion = right.shape[0] / float(bucket_data.shape[0])

            weighted_entropy = left_entropy * left_proportion + right_entropy * right_proportion

            if not best_option:
                best_option['entropy'] = weighted_entropy
                best_option['col_index'] = column
                best_option['split_value'] = row_value
                best_option['left_data'] = left
                best_option['right_data'] = right

            if weighted_entropy < best_option['entropy']:
                best_option['entropy'] = weighted_entropy
                best_option['col_index'] = column
                best_option['split_value'] = row_value
                best_option['left_data'] = left
                best_option['right_data'] = right

    if best_option['left_data'].empty or best_option['right_data'].empty:
        return {'classification': bucket_labels.value_counts().idxmax()}

    decision_tree = {'left_branch': generate_splits(best_option['left_data'], label_categories, max_depth, cur_depth + 1, min_sample),
                     'right_branch': generate_splits(best_option['right_data'], label_categories, max_depth, cur_depth + 1, min_sample),
                     'col_index': best_option['col_index'], 'split_value': best_option['split_value']}

    return decision_tree

def get_confusion_matrix_values(decision_tree, data):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for idx, row in data.iterrows():
        predicted = predict_label(decision_tree, row)
        actual = row.iloc[-1]

        if predicted == 1 and actual == 1:
            TP += 1
        elif predicted == 1 and actual == 0:
            FP += 1
        elif predicted == 0 and actual == 1:
            FN += 1
        elif predicted == 0 and actual == 0:
            TN += 1

    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

def evaluate_performance(decision_tree, data):
    total_rows = data.shape[0]
    correct = 0
    for idx, row in data.iterrows():
        predicted = predict_label(decision_tree, row)
        actual = row.iloc[-1]

        if predicted == actual:
            correct += 1

    return correct / float(total_rows)


def predict_label(decision_tree, row):
    if 'classification' in decision_tree:
        return decision_tree['classification']

    split_value = decision_tree['split_value']
    column_index = decision_tree['col_index']

    if row[column_index] < split_value:
        return predict_label(decision_tree['left_branch'], row)
    else:
        return predict_label(decision_tree['right_branch'], row)


def calculate_entropy(labels, categories):
    entropy = 0
    total_data = len(labels)

    if total_data == 0:
        return entropy

    for category in categories:
        matching_labels = 0
        for label in labels:
            if label == category:
                matching_labels += 1

        category_prob = matching_labels / float(total_data)

        if category_prob > 0:
            entropy += category_prob * math.log(1 / category_prob, 2)

    return entropy

num_folds = 3
max_depth = 7
min_sample = 15

df = pd.read_csv('/Users/justinlittman/dev/courses/DS4420/HW1/spambase.csv', header=None)
run_algorithm(df, num_folds, max_depth, min_sample)

