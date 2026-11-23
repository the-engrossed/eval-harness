import json

import pytest
from domain_fixtures import invoice_data, step_record
from pydantic import ValidationError

from auditlab.domain.population import Invoice


@pytest.mark.parametrize("bad", [4500.0, 4500.5, "4500", True, -1])
def test_invoice_amount_rejects_non_int_or_negative(bad: object) -> None:
    with pytest.raises(ValidationError, match="amount_cents"):
        Invoice.model_validate(invoice_data(amount_cents=bad))


@pytest.mark.parametrize("bad", ["4500.0", "4500.5", '"4500"', "true"])
def test_invoice_amount_rejects_floats_in_json(bad: str) -> None:
    raw = json.dumps(invoice_data(amount_cents=0))
    raw = raw.replace('"amount_cents": 0', f'"amount_cents": {bad}')
    with pytest.raises(ValidationError, match="amount_cents"):
        Invoice.model_validate_json(raw)


def test_step_record_cost_micros_rejects_float() -> None:
    with pytest.raises(ValidationError, match="cost_micros"):
        step_record(cost_micros=1.5)
    with pytest.raises(ValidationError, match="cost_micros"):
        step_record(cost_micros=10.0)
