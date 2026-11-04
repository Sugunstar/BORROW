#!/usr/bin/env python3
"""Comprehensive test suite for borrow CLI."""
import http.server
import json
import os
import shutil
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import borrow


class TestCommandLineParsing(unittest.TestCase):
    def test_split_line(self):
        self.assertEqual(borrow.split_line("hello world"), ["hello", "world"])
        self.assertEqual(borrow.split_line('hello "world foo"'), ["hello", "world foo"])
        with self.assertRaises(borrow.UserError):
            borrow.split_line('unclosed "quote')

    def test_parse_args_valid_flags(self):
        args = borrow.parse_args("code", "build a web app -o app.py -t 0.5 -m 4096")
        self.assertEqual(args["text"], "build a web app")
        self.assertEqual(args["out"], "app.py")
        self.assertEqual(args["temp"], 0.5)
        self.assertEqual(args["max_tokens"], 4096)

    def test_parse_args_help_flag(self):
        args = borrow.parse_args("code", "--help")
        self.assertTrue(args.get("help"))
        args_short = borrow.parse_args("code", "-h")
        self.assertTrue(args_short.get("help"))

    def test_parse_args_invalid_flag_value(self):
        with self.assertRaises(borrow.UserError):
            borrow.parse_args("code", "test -t notanumber")
        with self.assertRaises(borrow.UserError):
            borrow.parse_args("code", "test -m notanumber")

    def test_parse_args_unknown_flag(self):
        with self.assertRaises(borrow.UserError):
            borrow.parse_args("code", "test --nonexistent")

    def test_parse_args_flag_missing_arg(self):
        with self.assertRaises(borrow.UserError):
            borrow.parse_args("code", "test -o")

    def test_parse_args_chat_with_real_file(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tf:
            tf.write(b"print('hello')")
            tf_path = tf.name

        try:
            line = f"explain @{tf_path} please"
            args = borrow.parse_args_chat(line)
            self.assertIn(tf_path, args["files"])
            self.assertIn(tf_path, args["text"])
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)

    def test_parse_args_chat_ignores_annotations_when_not_file(self):
        line = "how does @Override work in java?"
        args = borrow.parse_args_chat(line)
        self.assertEqual(args["files"], [])
        self.assertEqual(args["text"], "how does @Override work in java?")


class TestFileBlockHandling(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_read_file_block_success(self):
        test_file = Path(self.temp_dir) / "sample.py"
        test_file.write_text("def foo():\n    return 42\n", encoding="utf-8")
        block = borrow.read_file_block(str(test_file))
        self.assertIn("File: " + str(test_file), block)
        self.assertIn("def foo():", block)

    def test_read_file_block_missing_file(self):
        missing = Path(self.temp_dir) / "missing.txt"
        with self.assertRaises(borrow.UserError) as ctx:
            borrow.read_file_block(str(missing))
        self.assertIn("File not found", str(ctx.exception))

    def test_read_file_block_directory_error(self):
        with self.assertRaises(borrow.UserError) as ctx:
            borrow.read_file_block(self.temp_dir)
        self.assertIn("is a directory, not a file", str(ctx.exception))

    def test_read_file_block_truncation(self):
        huge_file = Path(self.temp_dir) / "huge.txt"
        huge_file.write_text("x" * (borrow.MAX_FILE_CHARS + 500), encoding="utf-8")
        block = borrow.read_file_block(str(huge_file))
        self.assertIn("[truncated to 60000 characters]", block)


class TestCodeExtractionAndContinuation(unittest.TestCase):
    def test_first_block_basic(self):
        text = "Here is code:\n```python\nprint('hello')\n```\nExplanation."
        self.assertEqual(borrow.first_block(text), "print('hello')\n")

    def test_first_block_with_explanation_before(self):
        text = "Root cause:\n1. Off by one error\n\n```python\ndef fixed():\n    return 42\n```"
        self.assertEqual(borrow.first_block(text), "def fixed():\n    return 42\n")

    def test_first_block_windows_crlf(self):
        text = "Here is code:\r\n```python\r\nprint('hello')\r\n```\r\nExplanation."
        self.assertEqual(borrow.first_block(text), "print('hello')\r\n")

    def test_first_block_unclosed_fence(self):
        text = "Here is cut-off code:\n```javascript\nconsole.log('hi');"
        self.assertEqual(borrow.first_block(text), "console.log('hi');")

    def test_merge_continuation_repeated_fence(self):
        prev = "```python\ndef test():\n    return True\n"
        more = "```python\ndef test2():\n    return False\n```"
        merged = borrow.merge_continuation(prev, more)
        self.assertEqual(prev.count("```"), 1)
        # Should strip the second opening ```python fence
        self.assertNotIn("```python\ndef test2()", merged)
        self.assertIn("def test2():", merged)

    def test_merge_continuation_bare_fence(self):
        prev = "```\nline 1\n"
        more = "```\nline 2\n```"
        merged = borrow.merge_continuation(prev, more)
        self.assertEqual(merged, "```\nline 1\nline 2\n```")


class TestOutputSaving(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_output_creates_subdirectories(self):
        target = Path(self.temp_dir) / "nested" / "sub" / "code.py"
        code_text = "```python\nx = 10\n```"
        borrow.save_output(str(target), code_text, as_code=True)
        self.assertTrue(target.is_file())
        self.assertEqual(target.read_text(encoding="utf-8"), "x = 10\n")


class TestConfigAndPermissions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.orig_config = borrow.CONFIG_PATH
        borrow.CONFIG_PATH = Path(self.temp_dir) / ".borrow.json"

    def tearDown(self):
        borrow.CONFIG_PATH = self.orig_config
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_config(self):
        borrow.STATE["url"] = "https://test.trycloudflare.com"
        borrow.STATE["key"] = "test-secret-key"
        borrow.save_config()

        self.assertTrue(borrow.CONFIG_PATH.is_file())
        # Check permissions on POSIX systems (should be 0600)
        if os.name != "nt":
            mode = stat.S_IMODE(os.stat(borrow.CONFIG_PATH).st_mode)
            self.assertEqual(mode, 0o600)

        # Clear state and reload
        borrow.STATE["url"] = ""
        borrow.STATE["key"] = ""
        with patch.dict(os.environ, {}, clear=True):
            borrow.load_config()
            self.assertEqual(borrow.STATE["url"], "https://test.trycloudflare.com")
            self.assertEqual(borrow.STATE["key"], "test-secret-key")


class TestHistoryTrimming(unittest.TestCase):
    def setUp(self):
        borrow.STATE["history"] = []

    def test_trim_history_exceeding_budget(self):
        borrow.STATE["history"] = [
            {"role": "user", "content": "a" * 40_000},
            {"role": "assistant", "content": "b" * 40_000},
            {"role": "user", "content": "c" * 100},
            {"role": "assistant", "content": "d" * 100},
        ]
        borrow.trim_history()
        self.assertEqual(len(borrow.STATE["history"]), 2)
        self.assertEqual(borrow.STATE["history"][0]["content"], "c" * 100)

    def test_trim_history_with_extra_chars(self):
        borrow.STATE["history"] = [
            {"role": "user", "content": "a" * 30_000},
            {"role": "assistant", "content": "b" * 30_000},
            {"role": "user", "content": "c" * 100},
            {"role": "assistant", "content": "d" * 100},
        ]
        borrow.trim_history(extra_chars=20_000)
        self.assertEqual(len(borrow.STATE["history"]), 2)
        self.assertEqual(borrow.STATE["history"][0]["content"], "c" * 100)


class MockServerHandler(http.server.BaseHTTPRequestHandler):
    last_request_body = None

    def log_message(self, format, *args):
        pass  # Suppress logging in tests

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            MockServerHandler.last_request_body = body
            messages = body.get("messages", [])

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            # Check if this is a continuation request
            is_continue = any(m.get("content") == borrow.CONTINUE_PROMPT for m in messages)

            if not is_continue:
                # Stream chunk with thinking, then partial code ending with length limit
                chunks = [
                    {"choices": [{"delta": {"reasoning_content": "Analyzing..."}, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": "```python\ndef first_part():\n"}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "length"}]},
                    "[DONE]",
                ]
            else:
                # Continuation response
                chunks = [
                    {"choices": [{"delta": {"content": "    return 100\n```"}, "finish_reason": "stop"}]},
                    "[DONE]",
                ]

            for chunk in chunks:
                if chunk == "[DONE]":
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
        else:
            self.send_response(404)
            self.end_headers()


class TestMockServerStreaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["BORROW_TEST"] = "1"
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), MockServerHandler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.daemon = True
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("BORROW_TEST", None)
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        borrow.STATE["url"] = f"http://127.0.0.1:{self.port}"
        borrow.STATE["key"] = "test-key"
        borrow.STATE["history"] = []

    def test_health_check(self):
        self.assertTrue(borrow.health())

    def test_ask_with_auto_continuation(self):
        reply = borrow.ask("code", "write a function")
        self.assertIsNotNone(reply)
        self.assertIn("def first_part():", reply)
        self.assertIn("return 100", reply)
        self.assertEqual(len(borrow.STATE["history"]), 2)

    def test_reasoning_is_kept_for_think_and_model_name_is_cleaned(self):
        borrow.ask("code", "write a function")
        self.assertEqual(borrow.STATE["thought"], "Analyzing...")
        self.assertGreater(borrow.STATE["ctx_used"], 0)
        self.assertEqual(borrow.pretty_model("/tmp/m/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"),
                         "Qwen3-Coder-30B-A3B-Instruct")
        self.assertEqual(borrow.pretty_model("qwen"), "qwen")

    def test_stream_chat_sampling_parameters(self):
        borrow.ask("code", "test params")
        body = MockServerHandler.last_request_body
        self.assertIsNotNone(body)
        self.assertEqual(body.get("temperature"), 0.2)
        self.assertEqual(body.get("top_p"), 0.8)
        self.assertEqual(body.get("top_k"), 20)
        self.assertEqual(body.get("repeat_penalty"), 1.05)

    def test_cmd_generic_with_out_silent(self):
        with tempfile.TemporaryDirectory() as td:
            out_file = Path(td) / "generated.py"
            args = borrow.parse_args("code", f"write something -o {out_file}")
            borrow.cmd_generic("code", args)
            self.assertTrue(out_file.is_file())
            content = out_file.read_text(encoding="utf-8")
            self.assertIn("def first_part():", content)
            self.assertIn("return 100", content)


class TestCompact(unittest.TestCase):
    def setUp(self):
        self.saved = dict(borrow.STATE)
        borrow.STATE["history"] = [{"role": r, "content": f"{r} {i}"} for i in range(4) for r in ("user", "assistant")]

    def tearDown(self):
        borrow.STATE.clear()
        borrow.STATE.update(self.saved)

    def test_compact_keeps_last_exchange_and_summarizes_the_rest(self):
        seen = {}
        def fake(messages, **kw):
            seen["prompt"] = messages[-1]["content"]
            return "- user wants X", "stop"
        with patch.object(borrow, "stream_chat", side_effect=fake):
            self.assertTrue(borrow.compact(focus="the auth bug"))
        h = borrow.STATE["history"]
        self.assertEqual(len(h), 4)
        self.assertIn("- user wants X", h[0]["content"])
        self.assertEqual([m["content"] for m in h[2:]], ["user 3", "assistant 3"])
        self.assertIn("user 0", seen["prompt"])
        self.assertNotIn("user 3", seen["prompt"])
        self.assertIn("the auth bug", seen["prompt"])

    def test_failed_compact_leaves_history_alone(self):
        before = list(borrow.STATE["history"])
        with patch.object(borrow, "stream_chat", return_value=(None, None)):
            self.assertFalse(borrow.compact())
        self.assertEqual(borrow.STATE["history"], before)

    def test_auto_compact_only_when_context_is_nearly_full(self):
        borrow.STATE.update(n_ctx=32768, ctx_used=10000)
        with patch.object(borrow, "compact") as c:
            borrow.maybe_compact()
            c.assert_not_called()
            borrow.STATE["ctx_used"] = 27000                  # 82%
            borrow.maybe_compact()
            c.assert_called_once()
            borrow.STATE["history"] = borrow.STATE["history"][:4]   # just compacted: don't loop
            borrow.maybe_compact()
            c.assert_called_once()


class TestPromptLook(unittest.TestCase):
    def test_input_fragments_colours_command_flags_and_files(self):
        styles = dict((t, s) for s, t in borrow.input_fragments("/review @app.py -o out.txt"))
        self.assertEqual(styles["/review"], "class:cmd")
        self.assertEqual(styles["@app.py"], "class:file")
        self.assertEqual(styles["-o"], "class:flag")
        self.assertEqual(borrow.input_fragments("/con")[0][0], "class:cmd")       # half-typed is fine
        self.assertEqual(borrow.input_fragments("/nope")[0][0], "class:cmd-bad")
        self.assertEqual(borrow.input_fragments("/review", first=False)[0][0], "")  # only line 1 is a command
        self.assertEqual("".join(t for _, t in borrow.input_fragments("a  b @c")), "a  b @c")

    def test_caret_blinks_but_stays_solid_while_typing(self):
        self.assertTrue(borrow.caret_on(100.0, typed_at=100.0))     # just typed: visible
        self.assertFalse(borrow.caret_on(100.6, typed_at=100.0))    # first off phase
        self.assertTrue(borrow.caret_on(101.1, typed_at=100.0))     # back on

    def test_ctx_meter(self):
        self.assertEqual(borrow.ctx_meter(0, 100), "─" * 10)
        self.assertEqual(borrow.ctx_meter(50, 100), "━" * 5 + "─" * 5)
        self.assertEqual(borrow.ctx_meter(500, 100), "━" * 10)

    def test_spinner_rotates_thinking_words(self):
        s = borrow.Spinner("Thinking · step 2…")
        seen = set()
        for k in range(4):
            s.t0 = borrow.time.time() - k * 2.5 - 0.1
            seen.add(next(w for w in s.words if w in s.render(0)))
        self.assertGreater(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
