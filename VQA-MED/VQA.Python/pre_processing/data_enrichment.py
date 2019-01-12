import logging
import pandas as pd
import itertools
from tqdm import tqdm

from pre_processing.known_find_and_replace_items import imaging_devices, diagnosis, locations

logger = logging.getLogger(__name__)


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
    _add_columns_by_search(df, indicator_words=diagnosis, search_columns=['question', 'answer'])
    # add_locations_columns
    _add_columns_by_search(df, indicator_words=locations, search_columns=['question', 'answer'])

    # add_imaging_columns
    for col in imaging_devices:
        df[col] = ''
    _add_columns_by_search(df, indicator_words=imaging_devices, search_columns=['question', 'answer'])
    _consolidate_image_devices(df)
    for col in imaging_devices:
        del df[col]

    return df







def _add_columns_by_search(df, indicator_words, search_columns):
    from common.utils import has_word
    for word in indicator_words:
        res = None
        for col in search_columns:
            curr_res = df[col].apply(lambda s: has_word(word, s))
            if res is None:
                res = curr_res
            res = res | curr_res
        if any(res):
            df[word] = res
        else:
            logger.warning("found no matching for '{0}'".format(word))


def _consolidate_image_devices(df):
    def get_imaging_device(r):

        data = [r[device] for device in imaging_devices]
        result = ' '.join([device for d, device in zip(data, imaging_devices) if d]).strip()
        if len(result) == 0:
            result = 'unknown'
        return result

    df['imaging_device'] = df.apply(get_imaging_device, axis=1)

    image_names = df.image_name.drop_duplicates().values

    pbar = tqdm(image_names)
    logger.info('consolidating image devices')
    for image_name in pbar:
        df_image = df[df.image_name == image_name]
        raw_image_imaging_device = df_image.imaging_device.drop_duplicates().values

        image_imaging_device = set(itertools.chain.from_iterable([d.split() for d in raw_image_imaging_device]))

        consolidated = {d for d in image_imaging_device}
        if 'unknown' in consolidated and len(consolidated) == 2:
            consolidated.remove('unknown')

        if len(consolidated) > 1:
            consolidated.clear()
            consolidated.add('unknown')

        assert len(consolidated) == 1, \
            f'got {len(consolidated)} non consolidated image devices. for example:\n{consolidated[:5]}'
        devices_string = ' '.join([str(d) for d in consolidated])
        df.loc[df.image_name == image_name, 'imaging_device'] = devices_string
        # df[df.image_name == image_name].imaging_device = result
        pbar.set_description(f'image device:\t{devices_string}'.ljust(25))
    return df
