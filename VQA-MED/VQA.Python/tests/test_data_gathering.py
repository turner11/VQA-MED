import os
import pytest
from common.functions import normalize_data_strucrture
from parsers.VQA18 import Vqa18Base
from tests import image_folder

CSV_ROW_TEMPLATE = '{0}	test_image	question {0}?	answer {0}'


def get_csv_text(row_count):
    return '\n'.join([CSV_ROW_TEMPLATE.format(i) for i in range(row_count)])


@pytest.mark.parametrize("expected_length", [0, 1, 10, 100])
def test_data_length(expected_length):
    # Arrange
    csv = get_csv_text(expected_length)

    # Act & Assert
    df = Vqa18Base.get_instance(csv).data
    assert len(df) == expected_length

    normalized_data = normalize_data_strucrture(df, 'test', image_folder)
    assert len(normalized_data) == expected_length

    paths = normalized_data.path.drop_duplicates().values
    non_existing_path = [p for p in paths if not os.path.exists(p)]
    assert len(non_existing_path) == 0, f'Got missing paths:\n{non_existing_path }'


def main():
    d = 30
    test_data_length(d)


if __name__ == '__main__':
    main()
