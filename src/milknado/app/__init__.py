"""Application layer — command-level policy wiring domains to the CLI."""

from milknado.app.easy_cheese_execution import (
    AppliedMilknadoPlan,
    apply_milknado_plan_to_graph,
)
from milknado.app.easy_cheese_outcomes import (
    MilknadoCriterionOutcome,
    encode_milknado_physical_outcome,
)
from milknado.app.easy_cheese_results import collect_milknado_results

__all__ = [
    "AppliedMilknadoPlan",
    "MilknadoCriterionOutcome",
    "apply_milknado_plan_to_graph",
    "collect_milknado_results",
    "encode_milknado_physical_outcome",
]
