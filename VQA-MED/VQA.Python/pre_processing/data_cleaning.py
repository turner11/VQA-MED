import re
import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    from pre_processing.known_find_and_replace_items import find_and_replace_collection

    find_and_replace_data = find_and_replace_collection

    def replace_func(val: str) -> str:
        if isinstance(val, str):
            new_val = val
            for tpl in find_and_replace_data:
                pattern = re.compile(tpl.orig, re.IGNORECASE)
                new_val = pattern.sub(repl=tpl.sub, string=new_val).strip().lower()
        elif np.isnan(val):
            new_val = ''
        else:
            new_val = val

        return new_val

    df['original_question'] = df['question']
    df['original_answer'] = df['answer']

    df['question'] = df['question'].apply(replace_func)
    df['answer'] = df['answer'].apply(replace_func)
    return df
