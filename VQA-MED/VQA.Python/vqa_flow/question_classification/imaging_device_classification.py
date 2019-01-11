# coding: utf-8
import warnings

import pandas as pd
import numpy as np
import itertools
import random
import traceback
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pre_processing.known_find_and_replace_items import imaging_devices
from common.os_utils import File
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from vqa_flow.question_classification.classifier_info import ClassifierInfo


def get_classifier_data(df_arg):
    df = df_arg.copy()
    df['x'] = df.question_embedding

    df['y1'] = df.answer.isin(imaging_devices)  # .apply(lambda ans: )[][['question','answer']]
    df['y2'] = df.question.apply(lambda q: q.startswith('what shows'))
    df['y'] = df['y1'] | df['y2']

    df.loc[df.y == False, 'y'] = 0
    df.loc[df.y == True, 'y'] = 1

    return df[['question', 'y', 'imaging_device', 'answer', 'x']]


def get_data(data_path):
    with pd.HDFStore(data_path) as store:
        data = store['data']
    return data


def evaluate(clf, x_test, y_test):
    preds = clf.predict(x_test)
    # new_y_tes
    z = list(zip(preds, y_test))
    total = len(z)
    ok = len([tpl for tpl in z if tpl[0] == tpl[1]])
    acc = 1.0 * ok / total
    # print(acc)
    # preds
    return acc


def main(data_path):
    print(data_path)
    data = get_data(data_path)
    df = get_classifier_data(data)
    desc = df.y.describe()
    print(desc)

    skew_factor = sum(1 - df.y) / len(df.y)
    print(f'skew_factor: {skew_factor}')

    X_test, X_train, y_test, y_train = get_model_inputs(df)

    combs = get_hidden_layers_candidates()

    classifer_infos = []
    errors = []

    pbar = tqdm(combs)
    for hidden_layer in pbar:
        max_f1 = max(cinfo.f1 for cinfo in classifer_infos) if classifer_infos else None
        pbar.set_description(f'working on {hidden_layer}. max acc: {max_f1}')
        try:
            solver = 'adam'  # 'lbfgs'
            clf = MLPClassifier(solver=solver, alpha=1e-5, hidden_layer_sizes=hidden_layer, random_state=1)
            scores_train, scores_test = train_classifier(clf, X_train, y_train, X_test, y_test)

            curr_res = ClassifierInfo(classifier=clf, scores_train=scores_train, scores_test=scores_test)
            # curr_res.plot()

            classifer_infos.append(curr_res)

        except Exception as ex:
            errors.append(f'{ex}:\n{traceback.format_exc()}')
            raise
    if errors:
        print(errors)

    classifer_infos = sorted(classifer_infos, key=lambda info: info.precision, reverse=True)
    # path = 'C:\\Users\\Public\\Documents\\Data\\2018\\imaging_dvices_classifiers\\all_classifiers.pkl'
    # File.dump_pickle(classifer_infos, path)
    # print(f'saved all classifiers to: {path }')


    chosen = classifer_infos[1]
    print(f'best precision was: {chosen.scores_test["precision"][-1]}')

    clf = chosen.classifier

    save_path = save_classifier(clf)
    print(f'saved chosen classifier to: {save_path}')

    chosen.plot(title=f'chosen classifier: {clf.hidden_layer_sizes}',block=False)

    # plot_classifier_data(clf)

    predicted = clf.predict(X_test)
    z = list(zip(predicted, y_test))
    total = len(z)
    ok = len([tpl for tpl in z if tpl[0] == tpl[1]])
    print(1.0 * ok / total)

    # In[96]:


    for i, (expected, label) in list(enumerate(zip(y_test, predicted)))[:20]:
        question = df.question.loc[i]
        ans = df.answer.loc[i][:30]

        if label == expected:
            res = 'SUCCESS'
        else:
            res = 'FAIL'
        print(f'{i}. {question} => {res} (Predicted: {label}) Answer: {ans}')


def train_classifier(clf, X_train, y_train, X_test, y_test):
    # c = clf.fit(X_train, y_train)

    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 10
    N_BATCH = 128
    N_CLASSES = np.unique(y_train)

    score_funcs = {'accuracy':accuracy_score,
                   'f1': f1_score,
                   'precision': precision_score,
                   'recall': recall_score}

    train_vals = defaultdict(lambda: [])
    test_vals = defaultdict(lambda: [])

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            from sklearn.exceptions import UndefinedMetricWarning
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)

            # print(f'epoch: {epoch}/{N_EPOCHS}')
            # SHUFFLING
            random_perm = np.random.permutation(X_train.shape[0])
            mini_batch_index = 0
            while True:
                # MINI-BATCH
                indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
                clf.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
                mini_batch_index += N_BATCH

                if mini_batch_index >= N_TRAIN_SAMPLES:
                    break

            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)
            for name, score_func in score_funcs.items():
                score = score_func

                # SCORE TRAIN
                score_train = score_func(y_train, y_pred_train)
                train_vals[name].append(score_train)

                # SCORE TEST
                score_test = score_func(y_test, y_pred_test)
                test_vals[name].append(score_test)

            epoch += 1

    return train_vals, test_vals




def save_classifier(clf):
    path = 'C:\\Users\\Public\\Documents\\Data\\2018\\imaging_dvices_classifiers\\question_classifier.pickle'
    File.dump_pickle(clf, path)
    return path

#
# def plot_classifier_data(clf):
#     plt.ylabel('cost')
#     plt.xlabel('iterations')
#     plt.title("Learning rate =" + str(0.001))
#     plt.plot(clf.loss_curve_)
#     plt.show()


def get_hidden_layers_candidates()->list:
    combs_2 = list(itertools.combinations([5, 4, 3, 2], 2))
    combs_3 = list(itertools.combinations([5, 4, 3, 2], 3))
    combs_4 = list(itertools.combinations([5, 4, 3, 2], 4))
    combs = combs_2 + combs_3 + combs_4
    combs = [list(c) for c in combs if sum(c) <= 16]
    # combs = [list(c) for c in itertools.combinations([4, 6, 8, 5, 4, 7, 3], 5)]
    [random.shuffle(c) for c in combs]
    return combs


def _format_inputs(x:np.array, y:list)-> (np.array, list):
    new_x = np.asarray([v[0] for v in x])
    new_y = y#list(y)
    return new_x, new_y


def get_model_inputs(df:pd.DataFrame)->(np.array, np.array, list, list):
    X = df['x'].values
    y = df['y'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32, random_state=42)
    # In[8]:
    print(f'Train Length X, y: {len(X_train), len(y_train)}')
    print(f'Test Length X, y: {len(X_test), len(y_test)}')

    X_test, y_test = _format_inputs(X_test, y_test)
    X_train, y_train = _format_inputs(X_train, y_train)

    return X_test, X_train, y_test, y_train


if __name__ == '__main__':
    data_path = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\data\\model_input.h5'
    main(data_path)
