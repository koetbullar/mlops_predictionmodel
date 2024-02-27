import pytest

from prediction_model.config import config
from prediction_model.predict import generate_predictions
from prediction_model.preprocessing.data_handling import load_dataset

# Fixtures --> functions before execution of test function
# --> ensure single_prediction run


@pytest.fixture()
def single_prediction():
    test_dataset = load_dataset(config.TEST_FILE)
    single_row = test_dataset[:1]
    result = generate_predictions(single_row)
    return result


# output from predict is not none
def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None


# output from predict is str data type
def test_single_pred_str_type(single_prediction):
    assert isinstance(single_prediction.get('prediction')[0],str)


# output is 'Y' for an example data
def test_single_pred_validate(single_prediction):
    assert single_prediction.get('prediction')[0] == 'Y'
