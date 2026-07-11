from io import StringIO

from rich.console import Console

from ghostwheel.bootstrap import build_application
from ghostwheel.compaction import HistoryCompactor
from ghostwheel.config import Settings
from ghostwheel.pydantic_runner import PydanticAgentRunner


def test_build_application_wires_isolated_compaction_runner(tmp_path) -> None:
    config = Settings(_env_file=None).resolve()
    console = Console(file=StringIO(), force_terminal=False)

    application = build_application(config, console, cwd=tmp_path)
    try:
        chat_runner = application.session._runner
        compactor = application.session._compactor

        assert isinstance(chat_runner, PydanticAgentRunner)
        assert isinstance(compactor, HistoryCompactor)
        assert isinstance(compactor._runner, PydanticAgentRunner)
        assert compactor._runner is not chat_runner
        assert compactor._runner._agent is not chat_runner._agent
        assert compactor._runner._deps is None
        assert compactor._runner._event_sink is None
        assert compactor._runner._agent._function_toolset.tools == {}
        assert (
            compactor._token_counter is application.session.history_policy.token_counter
        )
        assert compactor._summary_token_limit == 2_048
        assert compactor._input_token_budget == 13_824
        assert application.session.estimated_context_tokens > 256
        assert application.session.context_tokens_estimated is True
    finally:
        application.close()
