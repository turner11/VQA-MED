# coding: utf-8
import datetime
import time
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
import random
import traceback
from collections import defaultdict

from matplotlib.figure import Figure
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from common.constatns import vqa_models_folder
from data_access.api import DataAccess
from common.settings import data_access as data_api

from common.os_utils import File
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from tools.tabbed_plots import open_window
from vqa_flow.question_classification.classifier_info import ClassifierInfo
import logging
import vqa_logger

logger = logging.getLogger(__name__)

CLASS_NAME = 'Abnormality'


def get_classifier_data(df_arg: pd.DataFrame) -> (pd.DataFrame, dict):
    df = df_arg.drop_duplicates(subset=['question'])
    df.loc[:, 'x'] = df.question_embedding

    lst_classes = sorted(df.question_category.drop_duplicates().values, key=lambda s: s == CLASS_NAME)
    classes = {v: i for i, v in enumerate(lst_classes)}
    df['y'] = df.question_category.apply(lambda category: classes[category])
    return df[['question', 'question_category', 'y', 'x']], classes


def get_data(data_access: DataAccess) -> pd.DataFrame:
    df = data_access.load_processed_data(
        columns=['question', 'answer', 'question_category', 'answer_embedding', 'question_embedding'])
    return df


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


def main(data_access: DataAccess) -> None:
    print(data_access)
    data = get_data(data_access)

    ab_idx = ~(data.question_category == CLASS_NAME)
    data.loc[ab_idx, 'question_category'] = 'Else'

    df, classes = get_classifier_data(data)
    desc = df.y.describe()
    print(desc)

    X_test, X_train, y_test, y_train = get_model_inputs(df)

    combs = get_hidden_layers_candidates()

    classifer_infos = []
    errors = []

    pbar = tqdm(combs)
    for hidden_layer in pbar:
        max_f1 = max(cinfo.f1 for cinfo in classifer_infos) if classifer_infos else None
        pbar.set_description(f'working on {hidden_layer}. max acc: {max_f1:.3f}')
        try:
            solver = 'adam'  # 'lbfgs'
            clf = MLPClassifier(solver=solver, alpha=1e-5, hidden_layer_sizes=hidden_layer, random_state=1)
            scores_train, scores_test = train_classifier(clf, X_train, y_train, X_test, y_test)

            curr_res = ClassifierInfo(classifier=clf, scores_train=scores_train, scores_test=scores_test,
                                      classes=classes)
            # curr_res.get_figure(plot=True)

            classifer_infos.append(curr_res)

        except Exception as ex:
            errors.append(f'{ex}:\n{traceback.format_exc()}')
            raise
    if errors:
        print(errors)

    classifer_infos = sorted(classifer_infos, key=lambda info: info.precision, reverse=True)
    # figures = [info.get_figure() for info in classifer_infos]
    # open_window(figures)

    # proc = multiprocessing.Process(target=open_window, args=(figures,))
    # proc.start()

    # path = 'C:\\Users\\Public\\Documents\\Data\\2018\\imaging_dvices_classifiers\\all_classifiers.pkl'
    # File.dump_pickle(classifer_infos, path)
    # print(f'saved all classifiers to: {path }')

    chosen = classifer_infos[1]
    print(f'best precision was: {chosen.scores_test["precision"][-1]}')

    clf = chosen.classifier
    fig = chosen.get_figure(title=f'chosen classifier: {clf.hidden_layer_sizes}', plot=True)

    save_path = save_classifier(chosen, fig)
    print(f'saved chosen classifier to: {save_path}')

    # plot_classifier_data(clf)

    # predicted = clf.predict(X_test)
    x, y = _format_inputs(df.x, df.y)
    predicted = clf.predict(x)
    z = list(zip(predicted, y))
    total = len(z)
    ok = len([tpl for tpl in z if tpl[0] == tpl[1]])
    print(1.0 * ok / total)

    df_res = pd.DataFrame({'y_test': y, 'prediction': predicted}).reset_index(drop=True)
    df_res['question'] = df.reset_index().question
    df_res['is_correct'] = df_res.y_test == df_res.prediction
    df_res['label'] = df_res.apply(
        lambda
            row: f'{"SUCCESS" if row.is_correct else "FAIL"}. {row.question} => {"SUCCESS" if row.is_correct else "FAIL"} (Predicted: {row.y_test}) Answer: {row.prediction}',
        axis=1)

    abnormality_class = classes[CLASS_NAME]
    df_small = df_res[(df_res.prediction == abnormality_class) | (df_res.y_test == abnormality_class)]

    print(df_small.label.values)


def train_classifier(clf, X_train, y_train, X_test, y_test):
    # c = clf.fit(X_train, y_train)

    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 10
    N_BATCH = 128
    N_CLASSES = np.unique(y_train)

    AVERAGE = 'weighted'
    score_funcs = {'accuracy': accuracy_score,
                   'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average=AVERAGE),
                   'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average=AVERAGE),
                   'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average=AVERAGE),
                   }

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


def save_classifier(clf_info: ClassifierInfo, fig: Figure = None) -> str:
    now = time.time()
    ts = datetime.datetime.fromtimestamp(now).strftime('%Y%m%d_%H%M_%S')
    clf = clf_info.classifier
    folder = Path(vqa_models_folder) / f'question_classifier_{ts}'
    folder.mkdir()

    path = folder / 'question_classifier.pickle'
    File.dump_pickle(clf, str(path))
    if fig is not None:
        fig_path = str(folder / 'figure.jpg')
        fig.savefig(fig_path)

    return str(folder)


#
# def plot_classifier_data(clf):
#     plt.ylabel('cost')
#     plt.xlabel('iterations')
#     plt.title("Learning rate =" + str(0.001))
#     plt.plot(clf.loss_curve_)
#     plt.show()


def get_hidden_layers_candidates() -> list:
    combs_2 = list(itertools.combinations([5, 4, 3, 2], 2))
    combs_3 = list(itertools.combinations([5, 4, 3, 2], 3))
    combs_4 = list(itertools.combinations([6, 5, 4, 3, 2], 4))
    combs_5 = list(itertools.combinations(list(range(3, 11)), 5))

    combs = combs_2 + combs_3 + combs_4 + combs_5
    combs = [list(c) for c in combs]
    # combs = [list(c) for c in combs if sum(c) <= 32]
    # combs = [list(c) for c in itertools.combinations([4, 6, 8, 5, 4, 7, 3], 5)]
    [random.shuffle(c) for c in combs]
    return combs


def _format_inputs(x: np.array, y: list) -> (np.array, list):
    new_x = np.array([np.array(xi) for xi in x])  # x#x.reshape(-1,1)#np.asarray([v[0] for v in x])#x#x.reshape(1, -1)#
    new_y = y  # list(y)
    return new_x, new_y


def get_model_inputs(df: pd.DataFrame) -> (np.array, np.array, list, list):
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
    main(data_api)
