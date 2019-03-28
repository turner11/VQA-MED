import logging
import re
from collections import Counter
import pandas as pd
import itertools
from tqdm import tqdm
from common.settings import validation_data, train_data
from pre_processing.known_find_and_replace_items import diagnosis

logger = logging.getLogger(__name__)


def _add_questions_categories(df):
    data_locations = [train_data, validation_data]
    dfs = [dl.get_image_names_category_info() for dl in data_locations]
    all_data = pd.concat(dfs)
    new_df = pd.merge(left=df, right=all_data
                       , left_on=['image_name', 'question'], right_on=['image_name', 'question']
                       , how='left')


    return new_df



def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches Data frame. Adds coulms for:
        1. diagnosis
        2. locations
        3. imaging_devices
        4. is question is about imaging device
    :type df: pd.DataFrame
    :returns df: pd.DataFrame
    """
    df: pd.DataFrame = df.copy()

    # add_diagnostics_columns
    _add_columns_by_search(df, new_columns_name='diagnosis' ,indicator_words=diagnosis, search_columns=['question', 'answer'])

    # add_locations_columns
    # _add_columns_by_search(df, new_columns_name='locations', indicator_words=locations, search_columns=['question', 'answer'])

    # _add_columns_by_search(df, new_columns_name='imaging_device', indicator_words=imaging_devices, search_columns=['processed_question', 'processed_answer'])
    # _consolidate_image_devices(df)

    df = _add_questions_categories(df)


    return df




def _add_columns_by_search(df, new_columns_name, indicator_words, search_columns):
    df_hit_miss = pd.DataFrame(index=df.index, columns=indicator_words)

    pbar = tqdm(indicator_words)
    for word in pbar:
        pbar.set_description(f'Looking for word: {word}')
        c_pattern = re.compile(r'\b{0}\b'.format(word), re.I)
        res = None
        for col in search_columns:
            # curr_res = df[col].apply(lambda s: has_word(word, s))
            curr_res = df[col].str.contains(c_pattern)

            if res is None:
                res = curr_res
            res = res | curr_res

        df_hit_miss[word] = res
        if not any(res):
            logger.warning("\nfound no matching for '{0}'".format(word))
        # if any(res):
        #     df_hit_miss[word] = res
        # else:
        #     logger.warning("\nfound no matching for '{0}'".format(word))
    df[new_columns_name] = df_hit_miss.apply(lambda row: ' '.join({col for col in df_hit_miss.columns if row[col]}), axis=1)

def _consolidate_image_devices(df):
    image_names = df.image_name.drop_duplicates().values

    pbar = tqdm(image_names)
    logger.info('consolidating image devices')
    for image_name in pbar:
        df_image = df[df.image_name == image_name]
        image_imaging_device = list(itertools.chain.from_iterable([d.split() for d in df_image.imaging_device.values]))
        consolidated = {d for d in image_imaging_device if d}

        if len(consolidated) > 1:
            #try majority
            cntr = Counter(image_imaging_device)
            (top1_val, top1_freq), (top2_val, top2_freq) = cntr.most_common(2)
            if top1_freq > top2_freq:
                consolidated = {top1_val}
            else:
                consolidated = {}

        if len(consolidated) == 0:
            consolidated = {'unknown'}

        assert len(consolidated) == 1, \
            f'got {len(consolidated)} non consolidated image devices. for example:\n{consolidated[:5]}'
        devices_string = consolidated.pop()
        df.loc[df.image_name == image_name, 'imaging_device'] = devices_string
        # df[df.image_name == image_name].imaging_device = result
        pbar.set_description(f'image device:\t{devices_string}'.ljust(25))
    count_unknown = len(df[df.imaging_device == 'unknown'].image_name.drop_duplicates())
    if count_unknown:
        count_total = len(df.image_name.drop_duplicates())
        logger.warning(f"Did not find an imaging device for {count_unknown}/{count_total}")
    return df
