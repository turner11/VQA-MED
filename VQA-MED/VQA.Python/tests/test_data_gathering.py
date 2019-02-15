import os
import pytest
import pandas as pd
from common.functions import normalize_data_structure
from common.settings import train_data, validation_data
from parsers.data_loader import DataLoader
from tests.conftest import image_folder

CSV_ROW_TEMPLATE = 'test_image|asking some question{0}?|this is answer #{0}'


def get_csv_text(row_count):
    return '\n'.join([CSV_ROW_TEMPLATE.format(i) for i in range(row_count)])


def __validate_loaded_data(normalized_data: pd.DataFrame) -> None:
    paths = normalized_data.path.drop_duplicates().values
    non_existing_path = [p for p in paths if not os.path.exists(p)]
    assert len(non_existing_path) == 0, f'Got missing paths:\n{non_existing_path}'


@pytest.mark.parametrize("expected_length", [1, 10, 50, 100])
def test_data_length(expected_length):
    # Arrange
    raw_text = get_csv_text(expected_length)

    # Act & Assert
    df = DataLoader.get_data(raw_text)
    assert len(df) == expected_length

    normalized_data = normalize_data_structure(df, 'test', image_folder)
    assert len(normalized_data) == expected_length

    __validate_loaded_data(normalized_data)


@pytest.mark.parametrize("data_arg", [train_data, validation_data])
def test_data_load(data_arg):
    data_location = data_arg.qa_path
    df = DataLoader.get_data(data_location)
    normalized_data = normalize_data_structure(df, data_arg.tag, data_arg.images_folder)
    __validate_loaded_data(normalized_data)


def main():
    test_data_load(train_data)
    d = 1
    test_data_length(d)


if __name__ == '__main__':
    main()
