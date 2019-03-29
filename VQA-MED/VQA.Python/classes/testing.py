import itertools
from pathlib import Path

import tqdm
import pandas as pd

from common.os_utils import File

allow_multi_predictions = False

def main():
    folder = Path('D:\\Users\\avitu\\Downloads')

    # words_decoder.to_hdf(str(folder/'h5'), key='words_decoder', format='t')
    # df_data.to_pickle(str(folder / 'data.pkl'))
    # File.dump_pickle(predictions, str(folder/'predictions.pickle'))
    # File.dump_pickle(probabilities, str(folder / 'probabilities.pickle'))

    words_decoder = pd.read_hdf(str(folder / 'h5'), key='words_decoder', format='t')
    df_data = pd.read_pickle(str(folder / 'data.pkl'))
    predictions = File.load_pickle(str(folder / 'predictions.pickle'))
    probabilities = File.load_pickle(str(folder / 'probabilities.pickle'))




    # dictionary for creating a data frame
    cols_to_transfer = ['image_name', 'question', 'answer', 'path']
    df_dict = {col_name: df_data[col_name] for col_name in cols_to_transfer}
    df_data_light = pd.DataFrame(df_dict).reset_index()

    results = []
    pbar = tqdm.tqdm(enumerate(zip(predictions, probabilities)), total=len(predictions))
    for i, (curr_prediction, curr_probabilities) in pbar:
        pbar.set_description(f'Prediction: {str(curr_prediction)[:20]}; probabilities: {str(curr_probabilities)[:20]}')
        prediction_df = pd.DataFrame({'word_idx': curr_prediction,
                                      'prediction': list(words_decoder.iloc[curr_prediction].str.strip().values),
                                      'probabilities': curr_probabilities}
                                     ).sort_values(by='probabilities', ascending=False).reset_index(drop=True)

        if not allow_multi_predictions:
            prediction_df = prediction_df.head(1)

        curr_prediction_str = ' '.join([str(w) for w in list(prediction_df.prediction.values)])
        probabilities_str = ', '.join(['({:.3f})'.format(p) for p in list(prediction_df.probabilities.values)])

        light_pred_df = pd.DataFrame({
            'prediction': [curr_prediction_str],
            'probabilities': [probabilities_str]
        })
        results.append(light_pred_df)

    df_aggregated = pd.DataFrame({
        'prediction': list(itertools.chain.from_iterable([curr_df.prediction.values for curr_df in results])),
        'probabilities': [curr_df.probabilities.values for curr_df in results]
    })
    ret = df_data_light.merge(df_aggregated, how='outer', left_index=True, right_index=True)
    ret = ret.set_index('index')
    str(ret)

if __name__ == '__main__':
    main()
