"""FinQA rollout loop with deterministic textual termination."""
from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop


class FinQASingleTurnAgentLoop(SingleTurnAgentLoop):
    async def run(self, sampling_params: dict[str, Any], **kwargs):
        params = dict(sampling_params)
        params["stop"] = ["End: FINQA_END"]
        params["include_stop_str_in_output"] = True
        return await super().run(params, **kwargs)
