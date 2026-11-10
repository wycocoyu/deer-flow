"""Opt-in merge of adjacent ``role=user`` Chat Completions messages (#6751).

DeerFlow's context middlewares add hidden user-role messages next to the real
user turn (slash skill activation, durable context data, memory, client and
language context). The OpenAI API accepts consecutive user messages, but some
OpenAI-compatible backends whose chat template requires strict user/assistant
alternation reject them with HTTP 400. ``merge_consecutive_user_messages`` lets
an operator merge them at the wire boundary for such a model only.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages

from deerflow.agents.middlewares.durable_context_middleware import DurableContextMiddleware
from deerflow.agents.middlewares.dynamic_context_middleware import DynamicContextMiddleware
from deerflow.agents.middlewares.skill_activation_middleware import SkillActivationMiddleware
from deerflow.agents.middlewares.system_message_coalescing_middleware import SystemMessageCoalescingMiddleware
from deerflow.config.app_config import AppConfig
from deerflow.config.model_config import ModelConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.config.skills_config import SkillsConfig
from deerflow.models import factory as factory_module
from deerflow.models.openai_response import GuardedChatOpenAI
from deerflow.models.patched_openai import PatchedChatOpenAI


def _model(cls=GuardedChatOpenAI, **kwargs):
    return cls(model="strict-model", api_key="sk-test", base_url="https://provider.invalid/v1", **kwargs)


def _roles(payload: dict) -> list[str]:
    return [message["role"] for message in payload["messages"]]


def _has_adjacent_user(payload: dict) -> bool:
    roles = _roles(payload)
    return any(a == b == "user" for a, b in zip(roles, roles[1:], strict=False))


def test_default_keeps_consecutive_user_messages():
    payload = _model()._get_request_payload([SystemMessage(content="sys"), HumanMessage(content="a"), HumanMessage(content="b")])

    assert _roles(payload) == ["system", "user", "user"]


def test_enabled_merges_string_contents_in_order():
    payload = _model(merge_consecutive_user_messages=True)._get_request_payload([SystemMessage(content="sys"), HumanMessage(content="context"), HumanMessage(content="question")])

    assert payload["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "context\n\nquestion"},
    ]


def test_enabled_merges_multimodal_parts_in_order():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    payload = _model(merge_consecutive_user_messages=True)._get_request_payload([HumanMessage(content="context"), HumanMessage(content=[{"type": "text", "text": "look"}, image])])

    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "context"}, {"type": "text", "text": "look"}, image]},
    ]


def test_enabled_never_merges_across_assistant_or_tool_messages():
    payload = _model(merge_consecutive_user_messages=True)._get_request_payload(
        [
            HumanMessage(content="u1"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "call_1", "type": "tool_call"}]),
            ToolMessage(content="r", tool_call_id="call_1"),
            AIMessage(content="answer"),
            HumanMessage(content="u2"),
            HumanMessage(content="u3"),
        ]
    )

    assert _roles(payload) == ["user", "assistant", "tool", "assistant", "user"]
    assert payload["messages"][-1]["content"] == "u2\n\nu3"


def test_enabled_keeps_user_messages_with_different_names_apart():
    payload = _model(merge_consecutive_user_messages=True)._get_request_payload([HumanMessage(content="a", name="alice"), HumanMessage(content="b", name="bob")])

    assert _roles(payload) == ["user", "user"]


def test_flag_is_inherited_by_patched_openai_adapters():
    payload = _model(PatchedChatOpenAI, merge_consecutive_user_messages=True)._get_request_payload([HumanMessage(content="a"), HumanMessage(content="b")])

    assert payload["messages"] == [{"role": "user", "content": "a\n\nb"}]


def test_factory_forwards_config_flag_without_leaking_into_payload():
    config = AppConfig(
        models=[
            ModelConfig(
                name="strict",
                display_name="strict",
                description=None,
                use="langchain_openai:ChatOpenAI",
                model="strict-model",
                supports_thinking=False,
                supports_vision=False,
                merge_consecutive_user_messages=True,
            )
        ],
        sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"),
    )

    model = factory_module.create_chat_model(name="strict", app_config=config, attach_tracing=False, api_key="sk-test")
    payload = model._get_request_payload([HumanMessage(content="a"), HumanMessage(content="b")])

    assert isinstance(model, GuardedChatOpenAI)
    assert model.merge_consecutive_user_messages is True
    assert "merge_consecutive_user_messages" not in model.model_kwargs
    assert "merge_consecutive_user_messages" not in payload
    assert _roles(payload) == ["user"]


@tool
def sample_read_file(path: str) -> str:
    """Read a file."""
    return "ok"


def _skill_follow_up_payload(tmp_path: Path, *, merge: bool) -> dict:
    """Run the real context middlewares on a slash-skill follow-up turn and serialize the request."""
    skill_dir = tmp_path / "custom" / "demo-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo-skill\ndescription: Demo skill\n---\n\n# Demo Skill\n", encoding="utf-8")
    app_config = AppConfig(
        models=[ModelConfig(name="m", display_name="m", description=None, use="langchain_openai:ChatOpenAI", model="strict-model", supports_thinking=False, supports_vision=False)],
        sandbox=SandboxConfig(use="test"),
        skills=SkillsConfig(path=str(tmp_path)),
    )
    token = "owner-token"
    dynamic = DynamicContextMiddleware(agent_name="lead_agent", app_config=app_config, memory_enabled=True)
    dynamic._build_full_reminder = lambda runtime, query: ("<system-reminder>date</system-reminder>", "<memory>\n- preference\n</memory>")
    activation = SkillActivationMiddleware(available_skills={"demo-skill"}, app_config=app_config, slash_source_owner_token=token)
    durable = DurableContextMiddleware(skills_container_path="/mnt/skills", entry_decisions_owner_token=token)
    coalescing = SystemMessageCoalescingMiddleware()
    chat_model = _model(merge_consecutive_user_messages=merge)
    runtime = SimpleNamespace(context={"locale": "zh-CN"})

    first_turn = [HumanMessage(content="/demo-skill first question", id="msg-1")]
    history = add_messages(first_turn, dynamic.before_agent({"messages": first_turn}, runtime)["messages"])
    history = [
        *history,
        AIMessage(content="", id="ai-1", tool_calls=[{"name": "sample_read_file", "args": {"path": "/mnt/skills/custom/demo-skill/SKILL.md"}, "id": "call_1", "type": "tool_call"}]),
        ToolMessage(content="# Demo Skill", tool_call_id="call_1", name="sample_read_file", id="tool-1"),
        AIMessage(content="first answer", id="ai-2"),
        HumanMessage(content="/demo-skill follow-up question", id="msg-2"),
    ]
    state = {"messages": history, "summary_text": "Earlier summary", **(durable.before_model({"messages": history}, runtime) or {})}
    request = ModelRequest(model=chat_model, system_message=SystemMessage(content="static prompt"), messages=history, tools=[sample_read_file], runtime=runtime, state=state)
    captured: dict = {}

    def terminal(req: ModelRequest) -> AIMessage:
        captured["payload"] = chat_model._get_request_payload([req.system_message, *req.messages], **chat_model.bind_tools(req.tools).kwargs)
        return AIMessage(content="ok")

    dynamic.wrap_model_call(request, lambda r1: activation.wrap_model_call(r1, lambda r2: durable.wrap_model_call(r2, lambda r3: coalescing.wrap_model_call(r3, terminal))))
    return captured["payload"]


def test_skill_follow_up_payload_has_adjacent_user_messages_by_default(tmp_path: Path):
    payload = _skill_follow_up_payload(tmp_path, merge=False)

    assert _has_adjacent_user(payload)


def test_skill_follow_up_payload_alternates_roles_when_enabled(tmp_path: Path):
    payload = _skill_follow_up_payload(tmp_path, merge=True)
    messages = payload["messages"]

    assert not _has_adjacent_user(payload)
    assert _roles(payload) == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert messages[2]["tool_calls"][0]["id"] == "call_1"
    assert [tool["function"]["name"] for tool in payload["tools"]] == ["sample_read_file"]
    final_user = messages[-1]["content"]
    # The hidden skill activation stays ahead of the real request it introduces.
    assert final_user.index("<slash_skill_activation>") < final_user.index("/demo-skill follow-up question")
