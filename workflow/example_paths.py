"""Repository-relative locations shared by the order example's build and validation tools."""
from pathlib import Path

ORDER_EXAMPLE = Path("example/order-fill-rate-spread")
SAMPLE_TICKS = ORDER_EXAMPLE / "data" / "sample_ticks.csv"
EXPECTED_OUTPUT = ORDER_EXAMPLE / "data" / "factor_output.csv"
