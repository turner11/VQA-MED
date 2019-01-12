from pre_processing.prepare_data import _apply_heavy_function
import pandas as pd
import dask.dataframe as dd


def test_dask_feature_extraction():
    # Arrange
    question_samples = [
        'what does mri show?',
        'where does axial section mri abdomen show hypoechoic mass?',
        'what do the arrows denote in the noncontrast ct scan image of pelvis?',
        'what was normal?',
        'what shows evidence of a contained rupture?',
        'what does preoperative ct demonstrate?',
        'what does the axial contracted CT section show?',
        'where does sagittal reformatted ct scan of the pelvis show a contrastfilled vagina?',
        'what does coronal ct scan demonstrate?',
        'what shows site and size of infarct in three study patients?',
        'what does coronal reformatted ct image demonstrate multiple bilateral nofs of?',
        'where does ct scan show enlarged lymph node?',
        'what shows complete healing of the lesion?',
        'what does mri brain show?',
        'where does ct show a cholecysto duodenal fistula and an impacted gallstone?',
        'what shows dilatation of the vertebrobasilar artery and the internal carotid arteries?',
    ]
    q_col = 'question'
    dask_rev_col = 'reveresed'
    apply_rev_col = 'double_reveresed'
    pd_data = {q_col:question_samples}

    df = pd.DataFrame(pd_data)
    ddata = dd.from_pandas(df, npartitions=8)
    def reverse(lst):
        return lst[::-1]

    # Act
    df[dask_rev_col]= _apply_heavy_function(dask_df=ddata, apply_func=reverse,column=q_col)
    df[apply_rev_col ] = df[q_col].apply(reverse)

    # Assert
    diffs = df[df[apply_rev_col] != df[apply_rev_col]]
    assert len(diffs) == 0, 'Got different results for reversing in spark'



def main():
    test_dask_feature_extraction()

if __name__ == '__main__':
    main()
