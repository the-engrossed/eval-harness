import pytest

from auditlab.money import cost_micros, format_cents, format_usd, mean_micros


def test_format_cents() -> None:
    assert format_cents(2_750_000) == "$27,500.00"
    assert format_cents(5) == "$0.05"
    assert format_cents(0) == "$0.00"
    assert format_cents(-123_456) == "-$1,234.56"


def test_cost_micros_uses_the_rates_it_is_given() -> None:
    # 1M in at 2_000_000 micros/1M + 500k out at 8_000_000 micros/1M = 2_000_000 + 4_000_000
    assert (
        cost_micros(
            1_000_000,
            500_000,
            input_micros_per_million=2_000_000,
            output_micros_per_million=8_000_000,
        )
        == 6_000_000
    )


def test_cost_micros_rounds_down_to_whole_micros() -> None:
    assert cost_micros(1, 0, input_micros_per_million=999_999, output_micros_per_million=0) == 0


def test_negative_tokens_or_rates_raise() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        cost_micros(-1, 0, input_micros_per_million=1, output_micros_per_million=1)
    with pytest.raises(ValueError, match="non-negative"):
        cost_micros(1, 1, input_micros_per_million=-1, output_micros_per_million=1)


def test_format_usd() -> None:
    assert format_usd(12_500_000) == "$12.500000"
    assert format_usd(-1) == "-$0.000001"


def test_mean_micros() -> None:
    assert mean_micros((100, 200, 300)) == 200
    with pytest.raises(RuntimeError, match="empty"):
        mean_micros(())
