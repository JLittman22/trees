import numpy as np
import pandas as pd


def run_algorithm(max_depth, min_sample, training, test):
    training = normalize_data(training)
    test = normalize_data(test)
    training_set_errors = []
    test_set_errors = []

    test_set = test
    training_set = training

    label_categories = training_set.iloc[:, -1].unique()

    decision_tree = generate_splits(training_set, label_categories, max_depth, 0, min_sample)
    print(decision_tree)
    training_error_rate = evaluate_performance(decision_tree, training_set)
    test_error_rate = evaluate_performance(decision_tree, test_set)

    training_set_errors.append(training_error_rate)
    test_set_errors.append(test_error_rate)

    mean_training_error = sum(training_set_errors) / float(len(training_set_errors))
    mean_test_error = sum(test_set_errors) / float(len(test_set_errors))

    print ("Average Training Error is: " + str(mean_training_error))
    print ("Average Test Error is: " + str(mean_test_error))


def normalize_data(data):
    for column in data.iloc[:, :data.shape[1] - 1].columns:
        min = np.amin(data[column])
        max = np.amax(data[column])
        for i in range(len(data[column])):
            value = (data[column][i] - min) / float(max)
            data.at[i, column] = value

    return data


def generate_splits(bucket_data, label_categories, max_depth, cur_depth, min_sample):
    bucket_labels = bucket_data.iloc[:, -1]

    current_variance = calculate_variance(bucket_labels)

    if max_depth == cur_depth or current_variance < 10 or len(bucket_labels) < min_sample:
        return {'classification': np.mean(bucket_labels)}

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

            if left.shape[0] == 0 or right.shape[0] == 0:
                continue

            left_labels = left.iloc[:, -1]
            left_variance = calculate_variance(left_labels)
            left_proportion = left.shape[0] / float(bucket_data.shape[0])

            right_labels = right.iloc[:, -1]
            right_variance = calculate_variance(right_labels)
            right_proportion = right.shape[0] / float(bucket_data.shape[0])

            weighted_variance = left_variance * left_proportion + right_variance * right_proportion

            if len(bucket_labels) < min_sample:
                return {'classification': np.mean(bucket_labels)}

            if not best_option:
                best_option['variance'] = weighted_variance
                best_option['col_index'] = column
                best_option['split_value'] = row_value
                best_option['left_data'] = left
                best_option['right_data'] = right

            if weighted_variance < best_option['variance']:
                best_option['variance'] = weighted_variance
                best_option['col_index'] = column
                best_option['split_value'] = row_value
                best_option['left_data'] = left
                best_option['right_data'] = right

    if best_option['left_data'].empty or best_option['right_data'].empty:
        return {'classification': np.mean(bucket_labels)}

    decision_tree = {'left_branch': generate_splits(best_option['left_data'], label_categories, max_depth, cur_depth + 1, min_sample),
                     'right_branch': generate_splits(best_option['right_data'], label_categories, max_depth, cur_depth + 1, min_sample),
                     'col_index': best_option['col_index'], 'split_value': best_option['split_value']}

    return decision_tree


def evaluate_performance(decision_tree, data):
    total_rows = data.shape[0]
    sum = 0
    for idx, row in data.iterrows():
        predicted = predict_label(decision_tree, row)
        actual = row.iloc[-1]

        sum += (predicted - actual) ** 2

    mse = sum / total_rows

    return mse


def predict_label(decision_tree, row):
    if 'classification' in decision_tree:
        return decision_tree['classification']

    split_value = decision_tree['split_value']
    column_index = decision_tree['col_index']

    if row[column_index] < split_value:
        return predict_label(decision_tree['left_branch'], row)
    else:
        return predict_label(decision_tree['right_branch'], row)


def calculate_variance(labels):
    n = len(labels)
    mean = np.mean(labels)
    variance = np.sum(((labels - mean) ** 2)) / n

    return variance

max_depth = 10
min_sample = 15

training_df = pd.read_csv('housing_train.txt', header=None, sep=' +', engine='python')
test_df = pd.read_csv('housing_test.txt', header=None, sep=' +', engine='python')
run_algorithm(max_depth, min_sample, training_df, test_df)

