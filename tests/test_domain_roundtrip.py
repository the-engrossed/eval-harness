import pytest
from domain_fixtures import agent_outputs, answer_key, population
from pydantic import BaseModel


@pytest.mark.parametrize(
    "model",
    [population(), answer_key(), *agent_outputs()],
    ids=lambda m: type(m).__name__,
)
def test_json_round_trip_is_lossless(model: BaseModel) -> None:
    assert type(model).model_validate_json(model.model_dump_json()) == model
