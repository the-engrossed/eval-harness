import pytest

from auditlab.agent.prompts import PROMPTS_DIR, load_prompt


def test_prompts_dir_is_inside_the_agent_package() -> None:
    assert PROMPTS_DIR.parts[-3:] == ("auditlab", "agent", "prompts")


def test_missing_prompt_fails_loudly_with_its_path() -> None:
    with pytest.raises(FileNotFoundError, match=r"v0[/\\]no_such_node\.md"):
        load_prompt("v0", "no_such_node")
