"""Optional offline Hugging Face ReAct runtime for the CUDA controller."""
import json
import logging
import os


def transformers_dialog(system, initial, turns, model, decode, dispatch, done, decode_failure):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from transformers.agents import ReactJsonAgent, Tool
        from transformers.agents.agents import AgentError, AgentGenerationError
        from transformers.agents.llm_engine import get_clean_message_list
    except ImportError as exc:
        raise RuntimeError("transformers backend requires the local Transformers 4.41.2 agents runtime") from exc

    class CudaTools(Tool):
        name = "cuda_tools"
        description = "Dispatch one or more actions to the offline CUDA controller."
        inputs = {"actions": {"type": "any", "description": "Ordered CUDA actions"}}
        output_type = str

        def forward(self, actions):
            return json.dumps(dispatch(step, actions), ensure_ascii=False)

    def engine(memory, **_):
        # This runtime appends copies of all previous tool arguments in a reminder.
        # The original actions and observations already contain the complete CUDA edits.
        history = [m for m in memory[1:] if not (m["role"] == "user" and
                   m["content"].startswith("Reminder: you are working towards solving the following task:"))]
        messages = get_clean_message_list(history, {"tool-response": "user"})
        return "Action:\n" + model(step, memory[0]["content"], messages)

    runtime = ReactJsonAgent(tools=[CudaTools()], llm_engine=engine, system_prompt=system,
                             tool_parser=lambda reply: ("cuda_tools", {"actions": decode(reply)}),
                             add_base_tools=False)
    runtime.logger.setLevel(logging.CRITICAL)
    runtime.initialize_for_run(json.dumps(initial, ensure_ascii=False))
    for step in range(1, turns + 1):
        try:
            runtime.step()
        except AgentGenerationError as exc:
            raise RuntimeError(str(exc)) from exc
        except AgentError as exc:
            decode_failure(step, exc)
            runtime.logs[-1]["error"] = exc
        if done():
            return
