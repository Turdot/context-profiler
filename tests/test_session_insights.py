from context_profiler.diagnostics import diagnose_result
from context_profiler.models import APIRequest, BlockType, ContentBlock, Message, Role, Session
from context_profiler.profiler import profile_session
from context_profiler.session_insights import analyze_session_insights


def _request(index: int, blocks: list[ContentBlock], total_padding: int = 0) -> APIRequest:
    messages = [Message(role=Role.USER, blocks=blocks, index=0)]
    if total_padding:
        messages.append(
            Message(
                role=Role.ASSISTANT,
                blocks=[ContentBlock(BlockType.TEXT, "padding", token_count=total_padding)],
                index=1,
            )
        )
    return APIRequest(messages=messages, request_index=index, source_format="openai")


def test_session_insights_find_long_lived_context_blocks():
    carried = ContentBlock(
        BlockType.TOOL_RESULT,
        "large stable tool output",
        token_count=1200,
        tool_name="get_component_info",
    )
    session = Session(requests=[
        _request(0, [carried]),
        _request(1, [carried, ContentBlock(BlockType.TEXT, "new", token_count=50)]),
        _request(2, [carried, ContentBlock(BlockType.TEXT, "newer", token_count=50)]),
    ])

    insights = analyze_session_insights(session)

    hotspot = insights["carryover_hotspots"][0]
    assert hotspot["tool_name"] == "get_component_info"
    assert hotspot["first_request_index"] == 0
    assert hotspot["carried_request_count"] == 2
    assert hotspot["carried_tokens"] == 2400
    assert hotspot["source_block_id"] == "r0:m0:b0"
    assert hotspot["label"] == "tool result: get_component_info"

    system_session = Session(requests=[
        _request(0, [ContentBlock(BlockType.TEXT, "stable system", token_count=1200)]),
        _request(1, [ContentBlock(BlockType.TEXT, "stable system", token_count=1200)]),
    ])
    system_session.requests[0].messages[0].role = Role.SYSTEM
    system_session.requests[1].messages[0].role = Role.SYSTEM
    assert analyze_session_insights(system_session)["carryover_hotspots"] == []


def test_session_insights_flags_budget_pressure_and_compression_drop():
    session = Session(requests=[
        _request(0, [ContentBlock(BlockType.TEXT, "small", token_count=10_000)]),
        _request(1, [ContentBlock(BlockType.TEXT, "near budget", token_count=55_000)]),
        _request(2, [ContentBlock(BlockType.TEXT, "compressed", token_count=30_000)]),
    ])

    insights = analyze_session_insights(session)

    event_types = [event["type"] for event in insights["budget_events"]]
    assert "budget_pressure" in event_types
    assert "compression_opportunity" in event_types


def test_session_insights_track_artifact_lifecycle_and_diagnosis_hints():
    carried = ContentBlock(BlockType.TEXT, "stable context", token_count=1200)
    session = Session(requests=[
        _request(0, [carried, ContentBlock(BlockType.TOOL_USE, "read src/Button.tsx", token_count=100, tool_name="ReadFile")]),
        _request(1, [carried, ContentBlock(BlockType.TOOL_USE, "edit src/Button.tsx", token_count=100, tool_name="ApplyPatch")]),
        _request(2, [carried, ContentBlock(BlockType.TOOL_USE, "read src/Button.tsx again", token_count=100, tool_name="ReadFile")]),
    ])

    insights = analyze_session_insights(session)
    assert insights["artifact_lifecycles"][0]["artifact_key"] == "src/Button.tsx"
    assert insights["artifact_lifecycles"][0]["request_indices"] == [0, 1, 2]
    assert insights["artifact_lifecycles"][0]["source_block_id"] == "r0:m0:b1"

    cdn_session = Session(requests=[
        _request(0, [ContentBlock(BlockType.TOOL_RESULT, "<script src=\"https://cdn.example.com/cdn.js\"></script>", token_count=100)]),
        _request(1, [ContentBlock(BlockType.TOOL_RESULT, "<script src=\"https://cdn.example.com/cdn.js\"></script>", token_count=100)]),
    ])
    cdn_insights = analyze_session_insights(cdn_session)
    assert cdn_insights["artifact_lifecycles"] == []

    diagnosis = diagnose_result(profile_session(session), session=session)
    hint_types = [hint["type"] for hint in diagnosis["diff_hints"]]
    assert "token_carryover_hotspot" in hint_types
    assert "possible_artifact_lifecycle_churn" in hint_types


def test_session_insights_detect_tool_result_artifact_duplication():
    before = '{"success":true,"data":{"component_id":"cmp-1","content":"alpha beta gamma delta epsilon"}}'
    after = '{"success":true,"data":{"component_id":"cmp-1","content":"alpha beta gamma delta changed"}}'
    session = Session(requests=[
        _request(0, [ContentBlock(BlockType.TOOL_RESULT, before, token_count=800, tool_name="get_component_info")]),
        _request(1, [ContentBlock(BlockType.TOOL_RESULT, after, token_count=850, tool_name="update_component")]),
        _request(2, [ContentBlock(BlockType.TOOL_RESULT, before, token_count=800, tool_name="get_component_info")]),
    ])

    insights = analyze_session_insights(session)
    duplication = insights["artifact_duplications"][0]

    assert duplication["artifact_key"] == "component:cmp-1"
    assert duplication["occurrences"] == 3
    assert duplication["tools"] == ["get_component_info", "update_component"]
    assert duplication["source_block_id"] == "r0:m0:b0"
    assert duplication["redundant_tokens"] > 0

    diagnosis = diagnose_result(profile_session(session), session=session)
    assert "tool_result_artifact_duplication" in [hint["type"] for hint in diagnosis["diff_hints"]]


def test_session_insights_build_propagation_links_for_spawn_and_carry():
    base = "src/Card.tsx " + " ".join(["alpha beta gamma delta"] * 80)
    edited = "src/Card.tsx " + " ".join(["alpha beta gamma delta"] * 70 + ["changed layout spacing"] * 10)
    edited_2 = "src/Card.tsx " + " ".join(["alpha beta gamma delta"] * 68 + ["changed color palette"] * 12)
    edited_3 = "src/Card.tsx " + " ".join(["alpha beta gamma delta"] * 66 + ["changed copy text"] * 14)
    exact = ContentBlock(BlockType.TOOL_RESULT, "stable carried payload", token_count=900, tool_name="ReadFile")
    session = Session(requests=[
        _request(0, [
            ContentBlock(BlockType.TOOL_RESULT, base, token_count=1200, tool_name="ReadFile"),
            exact,
        ]),
        _request(1, [
            ContentBlock(BlockType.TOOL_USE, edited, token_count=1250, tool_name="ApplyPatch"),
            ContentBlock(BlockType.TOOL_USE, edited_2, token_count=1230, tool_name="ApplyPatch"),
            ContentBlock(BlockType.TOOL_USE, edited_3, token_count=1210, tool_name="ApplyPatch"),
            exact,
        ]),
    ])

    propagation = analyze_session_insights(session)["propagation"]
    link_types = {link["type"] for link in propagation["links"]}

    assert "carry" in link_types
    assert "spawn" in link_types
    spawn = next(link for link in propagation["links"] if link["type"] == "spawn")
    assert spawn["source_block_id"] == "r0:m0:b0"
    assert spawn["target_block_id"] == "r1:m0:b0"
    assert spawn["similarity"] >= 0.65

    accumulation = next(link for link in propagation["links"] if link["type"] == "accumulation")
    accumulation_node = next(node for node in propagation["nodes"] if node["id"] == accumulation["target"])
    assert accumulation["source_block_id"] == "r0:m0:b0"
    assert accumulation["target_request_index"] == 1
    assert accumulation_node["copies"] == 3
    assert accumulation_node["member_block_ids"] == ["r1:m0:b0", "r1:m0:b1", "r1:m0:b2"]


def test_session_insights_keeps_spawn_links_when_accumulation_is_present():
    carry_blocks = [
        ContentBlock(
            BlockType.TOOL_RESULT,
            f"stable carried payload {idx}",
            token_count=5000,
            tool_name="get_component_info",
        )
        for idx in range(30)
    ]
    source_blocks = [
        ContentBlock(
            BlockType.TOOL_RESULT,
            f"component-{idx} " + " ".join(["alpha beta gamma delta"] * 80),
            token_count=1200,
            tool_name="get_component_info",
        )
        for idx in range(30)
    ]
    target_blocks = [
        ContentBlock(
            BlockType.TOOL_USE,
            f"component-{idx} " + " ".join(["alpha beta gamma delta"] * 70 + [f"edited copy {target_idx}"] * 10),
            token_count=1250,
            tool_name="update_component",
        )
        for idx in range(30)
        for target_idx in range(3)
    ]
    session = Session(requests=[
        _request(0, [*carry_blocks, *source_blocks]),
        _request(1, [*carry_blocks, *target_blocks]),
    ])

    propagation = analyze_session_insights(session)["propagation"]
    spawn_links = [link for link in propagation["links"] if link["type"] == "spawn"]
    link_types = {link["type"] for link in propagation["links"]}

    assert "spawn" in link_types
    assert len(spawn_links) >= 40
    assert any(link["source_block_id"] == "r0:m0:b30" for link in spawn_links)

