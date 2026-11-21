import json
import pytest
from pathlib import Path
import borrow


@pytest.fixture(autouse=True)
def setup_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    borrow.STATE["history"].clear()
    borrow.STATE["thought"] = ""
    borrow.STATE["piped"] = ""


# -----------------------------------------------------------------------------
# 1. Continuation support
# -----------------------------------------------------------------------------
def test_continuation_assistant_text_no_duplication(monkeypatch):
    call_log = []

    def mock_stream_chat(messages, *args, **kwargs):
        call_log.append(list(messages))
        if len(call_log) == 1:
            return ("Part 1 of the explanation...", "length", [])
        return (" Part 2 completing the explanation.", "stop", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)

    res = borrow.run_agent("Explain architecture")

    assert len(call_log) == 2
    # Second call should only have the current turn's assistant text in continuation prompt
    second_call_messages = call_log[1]
    last_assistant_msg = next((m for m in reversed(second_call_messages) if m.get("role") == "assistant"), None)
    assert last_assistant_msg is not None
    assert last_assistant_msg["content"] == "Part 1 of the explanation..."
    assert second_call_messages[-1]["content"] == borrow.CONTINUE_PROMPT

    # Final merged text should have both parts without duplicating Part 1
    assert "Part 1 of the explanation... Part 2 completing the explanation." in res
    assert res.count("Part 1") == 1

    # Session history should only have the original user prompt and final merged answer
    assert len(borrow.STATE["history"]) == 2
    assert borrow.STATE["history"][0]["content"] == "Explain architecture"
    assert "Part 1" in borrow.STATE["history"][1]["content"]
    assert "Part 2" in borrow.STATE["history"][1]["content"]


def test_continuation_post_tool_call_response(monkeypatch, tmp_path):
    call_log = []

    def mock_stream_chat(messages, *args, **kwargs):
        call_log.append(list(messages))
        if len(call_log) == 1:
            # Turn 1: Model calls a tool with some commentary
            return ("I will create the file.", "stop", [{
                "id": "c1",
                "type": "function",
                "function": {"name": "write_file", "arguments": json.dumps({"path": "sample.txt", "content": "hello"})}
            }])
        elif len(call_log) == 2:
            # Turn 2: Response following tool output is cut off
            return ("Here is what was done: created sample.txt with", "length", [])
        else:
            # Turn 3: Continuation of Turn 2
            return (" content 'hello'. All done.", "stop", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)

    res = borrow.run_agent("Create sample.txt and explain it", auto_approve=True)

    assert (tmp_path / "sample.txt").exists()
    assert (tmp_path / "sample.txt").read_text() == "hello"
    assert len(call_log) == 3

    # Check continuation message in call 3: should NOT duplicate turn 1's "I will create the file."
    call3_messages = call_log[2]
    last_asst = next((m for m in reversed(call3_messages) if m.get("role") == "assistant"), None)
    assert last_asst["content"] == "Here is what was done: created sample.txt with"
    assert "I will create the file." not in last_asst["content"]

    # Final text should be the completed response
    assert "Here is what was done: created sample.txt with content 'hello'. All done." in res


def test_continuation_respects_max_continues_limit(monkeypatch):
    call_count = [0]

    def mock_stream_chat(messages, *args, **kwargs):
        call_count[0] += 1
        # Always return length cutoff
        return (f"Chunk {call_count[0]}...", "length", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)

    # With max_continues=2, it should call stream_chat initial + 2 continues = 3 times total
    res = borrow.run_agent("Write endless stream", max_continues=2)
    assert call_count[0] == 3
    assert "Chunk 1..." in res
    assert "Chunk 2..." in res
    assert "Chunk 3..." in res


# -----------------------------------------------------------------------------
# 2. Tool-call ID consistency & multiple calls
# -----------------------------------------------------------------------------
def test_missing_and_empty_tool_call_ids_normalized(monkeypatch):
    call_log = []

    def mock_stream_chat(messages, *args, **kwargs):
        call_log.append(list(messages))
        if len(call_log) == 1:
            return ("", "stop", [
                {"id": "", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}},
                {"id": None, "type": "function", "function": {"name": "read_file", "arguments": '{"path": "b.txt"}'}}
            ])
        return ("Done reading", "stop", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)
    monkeypatch.setattr(borrow, "execute_tool", lambda name, args, **kw: "mock content")

    borrow.run_agent("read a and b")

    assert len(call_log) == 2
    turn2_messages = call_log[1]
    asst_msg = next(m for m in turn2_messages if m.get("role") == "assistant" and m.get("tool_calls"))
    tool_msgs = [m for m in turn2_messages if m.get("role") == "tool"]

    assert len(asst_msg["tool_calls"]) == 2
    assert len(tool_msgs) == 2

    # Verify every call ID is a non-empty string and matches the corresponding tool message ID
    tc_id_0 = asst_msg["tool_calls"][0]["id"]
    tc_id_1 = asst_msg["tool_calls"][1]["id"]
    assert tc_id_0 and isinstance(tc_id_0, str)
    assert tc_id_1 and isinstance(tc_id_1, str)
    assert tc_id_0 != tc_id_1

    assert tool_msgs[0]["tool_call_id"] == tc_id_0
    assert tool_msgs[1]["tool_call_id"] == tc_id_1


def test_duplicate_tool_call_ids_disambiguated(monkeypatch):
    call_log = []

    def mock_stream_chat(messages, *args, **kwargs):
        call_log.append(list(messages))
        if len(call_log) == 1:
            return ("", "stop", [
                {"id": "call_dup", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "1.txt"}'}},
                {"id": "call_dup", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "2.txt"}'}}
            ])
        return ("Done", "stop", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)
    monkeypatch.setattr(borrow, "execute_tool", lambda name, args, **kw: "mock content")

    borrow.run_agent("read both")

    turn2_messages = call_log[1]
    asst_msg = next(m for m in turn2_messages if m.get("role") == "assistant" and m.get("tool_calls"))
    tool_msgs = [m for m in turn2_messages if m.get("role") == "tool"]

    id1 = asst_msg["tool_calls"][0]["id"]
    id2 = asst_msg["tool_calls"][1]["id"]

    assert id1 != id2, "Duplicate IDs must be disambiguated"
    assert tool_msgs[0]["tool_call_id"] == id1
    assert tool_msgs[1]["tool_call_id"] == id2


def test_multiple_tool_calls_executed_in_order(tmp_path, monkeypatch):
    def mock_stream_chat(messages, *args, **kwargs):
        if len(messages) <= 2:
            return ("", "stop", [
                {"id": "c1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "first.txt", "content": "1"})}},
                {"id": "c2", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "second.txt", "content": "2"})}}
            ])
        return ("Both written", "stop", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)

    borrow.run_agent("write first and second", auto_approve=True)

    assert (tmp_path / "first.txt").read_text() == "1"
    assert (tmp_path / "second.txt").read_text() == "2"


# -----------------------------------------------------------------------------
# 3. Tool execution exception handling & validation
# -----------------------------------------------------------------------------
def test_execute_tool_validates_tool_name():
    assert "Unknown tool 'fake_tool'" in borrow.execute_tool("fake_tool", {})
    assert "Invalid or missing tool name" in borrow.execute_tool("", {})
    assert "Invalid or missing tool name" in borrow.execute_tool(None, {})


def test_execute_tool_validates_argument_type():
    assert "must be a dictionary" in borrow.execute_tool("read_file", "not_a_dict")
    assert "must be a dictionary" in borrow.execute_tool("write_file", [1, 2, 3])


def test_execute_tool_validates_required_fields():
    assert "missing required argument" in borrow.execute_tool("read_file", {})
    assert "missing required argument" in borrow.execute_tool("edit_file", {"path": "a.py"})
    assert "missing required arguments" in borrow.execute_tool("write_file", {"path": "a.py"})
    assert "missing required argument" in borrow.execute_tool("search_files", {})
    assert "missing required argument" in borrow.execute_tool("run_command", {})


def test_execute_tool_validates_field_types():
    assert "must be an integer" in borrow.execute_tool("read_file", {"path": "a.py", "start_line": "invalid_num"})
    assert "cannot be greater than" in borrow.execute_tool("read_file", {"path": "a.py", "start_line": 10, "end_line": 5})
    assert "must be a string" in borrow.execute_tool("edit_file", {"path": "a.py", "target": 123, "replacement": "x"})
    assert "must be a string" in borrow.execute_tool("write_file", {"path": "a.py", "content": 456})
    assert "must be a string" in borrow.execute_tool("run_command", {"command": ["ls"]})


def test_unexpected_tool_exception_caught_without_crashing_loop(monkeypatch):
    call_log = []

    def mock_stream_chat(messages, *args, **kwargs):
        call_log.append(messages)
        if len(call_log) == 1:
            return ("", "stop", [{
                "id": "c1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "test.txt"}'}
            }])
        return ("Recovered from error", "stop", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)

    # Force tool_read_file to raise an unexpected runtime error
    def crashing_read(*args, **kwargs):
        raise RuntimeError("Simulated unexpected disk corruption")

    monkeypatch.setattr(borrow, "tool_read_file", crashing_read)

    res = borrow.run_agent("Read file")
    assert "Recovered from error" in res
    second_turn_tools = [m for m in call_log[1] if m.get("role") == "tool"]
    assert len(second_turn_tools) == 1
    assert "unexpected exception" in second_turn_tools[0]["content"]
    assert "Simulated unexpected disk corruption" in second_turn_tools[0]["content"]


def test_failed_tool_call_does_not_prevent_independent_calls_in_same_turn(tmp_path, monkeypatch):
    call_log = []

    def mock_stream_chat(messages, *args, **kwargs):
        call_log.append(messages)
        if len(call_log) == 1:
            return ("", "stop", [
                {"id": "bad_call", "type": "function", "function": {"name": "unknown_tool", "arguments": "{}"}},
                {"id": "good_call", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "success.txt", "content": "written"})}}
            ])
        return ("Handled both", "stop", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)

    borrow.run_agent("run both", auto_approve=True)

    # Valid call should still execute despite first call failing
    assert (tmp_path / "success.txt").exists()
    assert (tmp_path / "success.txt").read_text() == "written"

    tool_msgs = [m for m in call_log[1] if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert "Unknown tool 'unknown_tool'" in tool_msgs[0]["content"]
    assert "Successfully wrote" in tool_msgs[1]["content"]


# -----------------------------------------------------------------------------
# 4. Reasoning-channel tool-call rejection
# -----------------------------------------------------------------------------
def test_reasoning_channel_tool_calls_never_executed(tmp_path, monkeypatch):
    def mock_stream_chat(messages, *args, **kwargs):
        # Inject reasoning content in STATE["thought"] containing a tool call
        borrow.STATE["thought"] = '<tool_call>{"name": "write_file", "arguments": {"path": "danger.txt", "content": "bad"}}</tool_call>'
        # Model actual response content has NO tool call
        return ("I considered writing a file, but decided not to.", "stop", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)

    res = borrow.run_agent("Do your work")

    # The file from thought MUST NOT be created
    assert not (tmp_path / "danger.txt").exists()
    assert "decided not to" in res


def test_valid_textual_tool_call_executed(tmp_path, monkeypatch):
    def mock_stream_chat(messages, *args, **kwargs):
        if len(messages) <= 2:
            text = 'Let me write the file:\n<tool_call>{"name": "write_file", "arguments": {"path": "from_tag.txt", "content": "created via tag"}}</tool_call>'
            return (text, "stop", [])
        return ("Finished writing from tag", "stop", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)

    borrow.run_agent("write from tag", auto_approve=True)

    assert (tmp_path / "from_tag.txt").exists()
    assert (tmp_path / "from_tag.txt").read_text() == "created via tag"


def test_textual_tool_call_inside_markdown_code_block_ignored(tmp_path, monkeypatch):
    def mock_stream_chat(messages, *args, **kwargs):
        text = (
            "Here is an example of calling write_file:\n"
            "```xml\n"
            '<tool_call>{"name": "write_file", "arguments": {"path": "example.txt", "content": "example"}}</tool_call>\n'
            "```\n"
            "Hope this explanation helps!"
        )
        return (text, "stop", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)

    res = borrow.run_agent("Show me how write_file works")

    assert not (tmp_path / "example.txt").exists()
    assert "Hope this explanation helps!" in res


# -----------------------------------------------------------------------------
# 5. Maximum agent turns
# -----------------------------------------------------------------------------
def test_max_turns_termination_with_assistant_text(monkeypatch):
    turn_counter = [0]

    def mock_stream_chat(messages, *args, **kwargs):
        turn_counter[0] += 1
        return (f"Working on step {turn_counter[0]}", "stop", [{
            "id": f"c_{turn_counter[0]}",
            "type": "function",
            "function": {"name": "run_command", "arguments": f'{{"command": "echo {turn_counter[0]}"}}'}
        }])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)
    monkeypatch.setattr(borrow, "execute_tool", lambda name, args, **kw: "done")

    res = borrow.run_agent("Long task", auto_approve=True, max_turns=3)

    assert turn_counter[0] == 3
    # Result must explicitly state that max turns was reached and task was stopped
    assert "Stopped: reached the 3-step limit before completing the task." in res
    assert "Working on step 3" not in res or "Last planned action" in res

    # History must accurately reflect the stopped status
    assert len(borrow.STATE["history"]) == 2
    assert "Stopped: reached the 3-step limit" in borrow.STATE["history"][1]["content"]


def test_max_turns_termination_without_assistant_text(monkeypatch):
    turn_counter = [0]

    def mock_stream_chat(messages, *args, **kwargs):
        turn_counter[0] += 1
        return ("", "stop", [{
            "id": f"c_{turn_counter[0]}",
            "type": "function",
            "function": {"name": "run_command", "arguments": f'{{"command": "echo {turn_counter[0]}"}}'}
        }])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)
    monkeypatch.setattr(borrow, "execute_tool", lambda name, args, **kw: "done")

    res = borrow.run_agent("Silent loop", auto_approve=True, max_turns=2)

    assert "Stopped: reached the 2-step limit before completing the task." in res


# -----------------------------------------------------------------------------
# 6. Stream interruption & Network failures
# -----------------------------------------------------------------------------
def test_stream_interrupted_does_not_execute_partial_tool_calls(tmp_path, monkeypatch):
    def mock_stream_chat(messages, *args, **kwargs):
        # Stream was cut off mid-way
        return ("partial narration", "interrupted", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)

    res = borrow.run_agent("Do something risky")

    assert "Stopped: Generation stream interrupted" in res or "Interrupted" in res
    assert len(borrow.STATE["history"]) == 2
    assert "Stopped" in borrow.STATE["history"][1]["content"] or "Interrupted" in borrow.STATE["history"][1]["content"]


def test_keyboard_interrupt_stops_cleanly(monkeypatch):
    def mock_stream_chat(messages, *args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)

    res = borrow.run_agent("Cancel this")

    assert "Interrupted" in res
    assert len(borrow.STATE["history"]) == 2
    assert "Interrupted" in borrow.STATE["history"][1]["content"]


# -----------------------------------------------------------------------------
# 7. Repeating failed tool loop detection
# -----------------------------------------------------------------------------
def test_repeating_failed_tool_loop_detected(monkeypatch):
    call_count = [0]

    def mock_stream_chat(messages, *args, **kwargs):
        call_count[0] += 1
        return (f"Try {call_count[0]}", "stop", [{
            "id": f"call_{call_count[0]}",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "nonexistent.py"}'}
        }])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)

    res = borrow.run_agent("Read file")

    # Should stop after detecting repeating failure loop on turn 3
    assert call_count[0] == 3
    assert "Stopped: Detected repeating tool loop" in res


# -----------------------------------------------------------------------------
# 8. Intermediate commentary separation
# -----------------------------------------------------------------------------
def test_intermediate_commentary_separated_from_final_output(monkeypatch):
    step = [0]

    def mock_stream_chat(messages, *args, **kwargs):
        step[0] += 1
        if step[0] == 1:
            return ("Step 1: I will inspect the code.", "stop", [{
                "id": "c1",
                "type": "function",
                "function": {"name": "run_command", "arguments": '{"command": "git status"}'}
            }])
        elif step[0] == 2:
            return ("Step 2: Now I will run the tests.", "stop", [{
                "id": "c2",
                "type": "function",
                "function": {"name": "run_command", "arguments": '{"command": "pytest"}'}
            }])
        else:
            return ("All tests have passed successfully. The changes are verified.", "stop", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)
    monkeypatch.setattr(borrow, "execute_tool", lambda name, args, **kw: "success")

    final_res = borrow.run_agent("Run workflow", auto_approve=True)

    # Final response must NOT prepend Step 1 and Step 2 commentary
    assert final_res == "All tests have passed successfully. The changes are verified."
    assert "Step 1: I will inspect" not in final_res
    assert "Step 2: Now I will" not in final_res


# -----------------------------------------------------------------------------
# 9. Recovery, Context Pruning & History Preservation
# -----------------------------------------------------------------------------
def test_failed_tool_call_followed_by_successful_recovery(tmp_path, monkeypatch):
    code_file = tmp_path / "app.py"
    code_file.write_text("def run():\n    return False\n", encoding="utf-8")

    step = [0]
    call_log = []

    def mock_stream_chat(messages, *args, **kwargs):
        step[0] += 1
        call_log.append(messages)
        if step[0] == 1:
            # Model mistakenly tries an incorrect target string
            return ("Fixing the function", "stop", [{
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "arguments": json.dumps({"path": str(code_file), "target": "def run():\n    return None", "replacement": "def run():\n    return True"})
                }
            }])
        elif step[0] == 2:
            # Model receives the error, corrects the target string, and retries
            return ("Target was incorrect, updating with correct target", "stop", [{
                "id": "c2",
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "arguments": json.dumps({"path": str(code_file), "target": "def run():\n    return False", "replacement": "def run():\n    return True"})
                }
            }])
        else:
            return ("The function now returns True. Issue resolved.", "stop", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)

    res = borrow.run_agent("Fix app.py", auto_approve=True)

    assert "The function now returns True. Issue resolved." in res
    assert code_file.read_text(encoding="utf-8") == "def run():\n    return True\n"
    # Verify the first tool response communicated the error to the model
    turn2_messages = call_log[1]
    tool_msgs = [m for m in turn2_messages if m.get("role") == "tool"]
    assert "target string not found" in tool_msgs[0]["content"]


def test_context_pruning_preserves_assistant_tool_message_sequences(monkeypatch):
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Run analysis"},
        {"role": "assistant", "content": "Step 1", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "Line 1\nLine 2\nLine 3\n" + ("x" * 2000)},
        {"role": "assistant", "content": "Step 2", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "Line 1\nLine 2\nLine 3\n" + ("y" * 2000)},
    ]

    borrow.STATE["n_ctx"] = 1000
    borrow.prune_tool_outputs_if_needed(messages)

    # Verify message sequence and pairing is completely preserved
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "tool"
    assert messages[4]["role"] == "assistant"
    assert messages[5]["role"] == "tool"

    assert messages[2]["tool_calls"][0]["id"] == "c1"
    assert messages[3]["tool_call_id"] == "c1"
    assert messages[4]["tool_calls"][0]["id"] == "c2"
    assert messages[5]["tool_call_id"] == "c2"

    assert "[... output shortened to save context ...]" in messages[3]["content"]


def test_preservation_of_history_after_partial_failure(tmp_path, monkeypatch):
    step = [0]

    def mock_stream_chat(messages, *args, **kwargs):
        step[0] += 1
        if step[0] == 1:
            return ("Creating file", "stop", [{
                "id": "c1",
                "type": "function",
                "function": {"name": "write_file", "arguments": json.dumps({"path": "file1.txt", "content": "data1"})}
            }])
        else:
            # Step 2 connection drops mid-stream
            return ("Working on part 2...", "interrupted", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)

    res = borrow.run_agent("Write multiple files", auto_approve=True)

    # First file was written and is safely preserved
    assert (tmp_path / "file1.txt").exists()
    assert (tmp_path / "file1.txt").read_text() == "data1"

    # History preserved and notes the executed tools
    assert len(borrow.STATE["history"]) == 2
    assert borrow.STATE["history"][0]["content"] == "Write multiple files"
    asst_history = borrow.STATE["history"][1]["content"]
    assert "Interrupted" in asst_history or "write_file" in asst_history

