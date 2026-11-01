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


class MockServerHandler(http.server.BaseHTTPRequestHandler):
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
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), MockServerHandler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.daemon = True
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
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


if __name__ == "__main__":
    unittest.main()
