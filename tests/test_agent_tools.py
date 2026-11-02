import os
import pytest
from pathlib import Path
import borrow


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
