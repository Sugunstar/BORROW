import os
import pytest
from pathlib import Path
import borrow


import json
import os
import pytest
from pathlib import Path
import borrow


@pytest.fixture(autouse=True)
def setup_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_tool_read_file_basic(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n", encoding="utf-8")
    out = borrow.tool_read_file(str(f), start_line=2, end_line=4)
    assert "2 | line 2" in out
    assert "3 | line 3" in out
    assert "4 | line 4" in out
    assert "1 | line 1" not in out
    assert "5 | line 5" not in out


def test_tool_read_file_binary_rejected(tmp_path):
    bin_f = tmp_path / "bin.dat"
    bin_f.write_bytes(b"\x00\x01\x02\x03")
    out = borrow.tool_read_file(str(bin_f))
    assert "binary file" in out


def test_tool_edit_file_exact_replacement(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 10\ny = 20\n", encoding="utf-8")
    res = borrow.tool_edit_file(str(f), target="y = 20", replacement="y = 42", auto_approve=True)
    assert "Successfully updated" in res
    assert f.read_text(encoding="utf-8") == "x = 10\ny = 42\n"


def test_tool_edit_file_missing_target(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 10\n", encoding="utf-8")
    res = borrow.tool_edit_file(str(f), target="nonexistent", replacement="foo", auto_approve=True)
    assert "target string not found" in res


def test_tool_edit_file_ambiguous_target(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("var = 1\nvar = 1\n", encoding="utf-8")
    res = borrow.tool_edit_file(str(f), target="var = 1", replacement="var = 2", auto_approve=True)
    assert "occurs 2 times" in res


def test_tool_write_file_new_and_overwrite(tmp_path):
    f = tmp_path / "subdir" / "new_file.txt"
    res = borrow.tool_write_file(str(f), content="hello world", auto_approve=True)
    assert "Successfully wrote" in res
    assert f.read_text(encoding="utf-8") == "hello world"

    res_overwrite = borrow.tool_write_file(str(f), content="hello universe", auto_approve=True)
    assert "Successfully wrote" in res_overwrite
    assert f.read_text(encoding="utf-8") == "hello universe"


def test_tool_list_directory(tmp_path):
    (tmp_path / "test1.py").write_text("a", encoding="utf-8")
    (tmp_path / "dir1").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "git_internal").write_text("b", encoding="utf-8")

    out = borrow.tool_list_directory(str(tmp_path))
    assert "test1.py" in out
    assert "[dir]  dir1/" in out
    assert ".git" not in out


def test_tool_search_files(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "target.py").write_text("def unique_function_name(): pass\n", encoding="utf-8")
    out = borrow.tool_search_files("unique_function_name", path=str(tmp_path))
    assert "unique_function_name" in out
    assert "target.py:1" in out


def test_tool_run_command():
    out = borrow.tool_run_command("python3 -c 'print(\"agent command test\")'", auto_approve=True)
    assert "Exit code: 0" in out
    assert "agent command test" in out


# -----------------------------------------------------------------------------
# Tests for safe_path and boundary confinement
# -----------------------------------------------------------------------------
def test_safe_path_valid(tmp_path):
    p = borrow.safe_path("sample.py")
    assert p == (tmp_path / "sample.py").resolve()
    sub = borrow.safe_path("subdir/nested.txt")
    assert sub == (tmp_path / "subdir/nested.txt").resolve()
    assert borrow.safe_path(".") == tmp_path.resolve()


def test_safe_path_outside_refused(tmp_path):
    with pytest.raises(PermissionError, match="outside the project folder"):
        borrow.safe_path("../outside.txt")
    with pytest.raises(PermissionError, match="outside the project folder"):
        borrow.safe_path("/etc/passwd")


def test_safe_path_forbidden_files(tmp_path):
    with pytest.raises(PermissionError, match=r"\.env"):
        borrow.safe_path(".env")
    with pytest.raises(PermissionError, match=r"\.env"):
        borrow.safe_path("sub/.env")
    with pytest.raises(PermissionError, match=r"\.env"):
        borrow.safe_path(".env.local")
    with pytest.raises(PermissionError, match=r"~\/\.ssh"):
        borrow.safe_path("~/.ssh")
    with pytest.raises(PermissionError, match=r"~\/\.ssh"):
        borrow.safe_path("~/.ssh/id_rsa")
    with pytest.raises(PermissionError, match=r"~\/\.borrow\.json"):
        borrow.safe_path("~/.borrow.json")
    with pytest.raises(PermissionError, match=r"~\/\.borrow\.json"):
        borrow.safe_path(".borrow.json")


def test_file_tools_refuse_outside(tmp_path):
    assert "Access denied" in borrow.tool_read_file("/etc/passwd")
    assert "Access denied" in borrow.tool_edit_file("/etc/passwd", "a", "b", auto_approve=True)
    assert "Access denied" in borrow.tool_write_file("/etc/passwd", "x", auto_approve=True)
    assert "Access denied" in borrow.tool_list_directory("../")
    assert "Access denied" in borrow.tool_search_files("test", "../")


def test_file_tools_refuse_sensitive_files(tmp_path):
    assert "Access denied" in borrow.tool_read_file(".env")
    assert "Access denied" in borrow.tool_write_file(".env", "SECRET=1", auto_approve=True)
    assert "Access denied" in borrow.tool_read_file("~/.ssh/id_rsa")
    assert "Access denied" in borrow.tool_read_file(".borrow.json")
    assert "Access denied" in borrow.tool_read_file("~/.borrow.json")


# -----------------------------------------------------------------------------
# Tests for cut-off tool calls and 12-step limit
# -----------------------------------------------------------------------------
def test_cutoff_tool_call_not_run_with_empty_args(monkeypatch, tmp_path):
    call_log = []

    def mock_stream_chat(messages, *args, **kwargs):
        call_log.append(messages)
        if len(call_log) == 1:
            # Model response was cut off (finish == "length") with incomplete arguments JSON
            return ("", "length", [{
                "id": "call_cut_1",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "cutoff.txt", "content": "incomplete data...'
                }
            }])
        else:
            return ("Understood, writing in smaller parts.", "stop", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)

    borrow.run_agent("write a file")

    # Verify tool was NOT executed with empty arguments and file was not created
    assert not (tmp_path / "cutoff.txt").exists()
    assert not (tmp_path / "").is_file()

    # Verify tool message communicated the cut-off message to the model
    second_turn_messages = call_log[1]
    tool_messages = [m for m in second_turn_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "that was cut off, write the file in smaller parts" in tool_messages[0]["content"]


def test_invalid_json_tool_call_not_run(monkeypatch, tmp_path):
    call_log = []

    def mock_stream_chat(messages, *args, **kwargs):
        call_log.append(messages)
        if len(call_log) == 1:
            return ("", "stop", [{
                "id": "call_bad_json",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": "invalid json {{"
                }
            }])
        return ("Done", "stop", [])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)

    borrow.run_agent("write file")

    second_turn_messages = call_log[1]
    tool_messages = [m for m in second_turn_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "that was cut off, write the file in smaller parts" in tool_messages[0]["content"]


def test_twelve_step_limit_printed(monkeypatch, capsys):
    step_count = [0]

    def mock_stream_chat(messages, *args, **kwargs):
        step_count[0] += 1
        return (f"step {step_count[0]}", "stop", [{
            "id": f"call_{step_count[0]}",
            "type": "function",
            "function": {"name": "run_command", "arguments": '{"command": "echo step"}'}
        }])

    monkeypatch.setattr(borrow, "stream_chat", mock_stream_chat)
    monkeypatch.setattr(borrow, "is_pure_greeting", lambda text: False)
    monkeypatch.setattr(borrow, "execute_tool", lambda name, args, **kw: "step output")

    borrow.run_agent("keep going", auto_approve=True, max_turns=12)

    captured = capsys.readouterr().out
    assert "Reached the 12-step limit." in captured


# -----------------------------------------------------------------------------
# Tests for context pruning
# -----------------------------------------------------------------------------
def test_prune_tool_outputs_when_context_nears_limit(monkeypatch):
    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Read files and analyze."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "First long output line 1\nline 2\nline 3\n" + ("x" * 2000)},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2"}]},
        {"role": "tool", "tool_call_id": "c2", "content": "Second long output line 1\nline 2\nline 3\n" + ("y" * 2000)},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c3"}]},
        {"role": "tool", "tool_call_id": "c3", "content": "Latest output line 1\nline 2\n" + ("z" * 500)},
    ]

    borrow.STATE["n_ctx"] = 1500  # Low threshold: threshold = min(1125, 17500) = 1125 tokens (~4500 chars)

    borrow.prune_tool_outputs_if_needed(messages)

    # Older tool output (c1) should have been shortened
    assert "[... output shortened to save context ...]" in messages[3]["content"]
    assert "First long output line 1" in messages[3]["content"]
    assert len(messages[3]["content"]) < 300

