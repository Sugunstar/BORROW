#!/usr/bin/env python3
"""Tests for agent skills discovery, parsing, tool calling, and CLI commands."""
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.document import Document

import borrow


class TestSkillsSystem(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp_dir)

    def tearDown(self):
        os.chdir(self.old_cwd)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_parse_skill_md_with_frontmatter(self):
        skill_dir = Path(self.tmp_dir) / "demo-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            "---\nname: demo-skill\ndescription: A demo skill for testing.\n---\n# Demo\nDo something useful.",
            encoding="utf-8"
        )
        parsed = borrow.parse_skill_md(skill_file)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["name"], "demo-skill")
        self.assertEqual(parsed["description"], "A demo skill for testing.")
        self.assertIn("Do something useful.", parsed["body"])

    def test_parse_skill_md_fallback(self):
        skill_dir = Path(self.tmp_dir) / "no-frontmatter"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("# Heading Description\nSome instructions here.", encoding="utf-8")
        parsed = borrow.parse_skill_md(skill_file)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["name"], "no-frontmatter")
        self.assertEqual(parsed["description"], "Heading Description")

    def test_parse_skill_md_nonexistent(self):
        self.assertIsNone(borrow.parse_skill_md(Path(self.tmp_dir) / "ghost" / "SKILL.md"))

    def test_discover_skills_precedence(self):
        # Create global skill
        fake_home = Path(self.tmp_dir) / "fake_home"
        global_skill_dir = fake_home / ".borrow" / "skills" / "shared-skill"
        global_skill_dir.mkdir(parents=True)
        (global_skill_dir / "SKILL.md").write_text(
            "---\nname: shared-skill\ndescription: Global version.\n---\nGlobal instructions",
            encoding="utf-8"
        )

        # Create workspace skill with the same name
        ws_skill_dir = Path(self.tmp_dir) / ".borrow" / "skills" / "shared-skill"
        ws_skill_dir.mkdir(parents=True)
        (ws_skill_dir / "SKILL.md").write_text(
            "---\nname: shared-skill\ndescription: Workspace version.\n---\nWorkspace instructions",
            encoding="utf-8"
        )

        with patch("pathlib.Path.home", return_value=fake_home):
            skills = borrow.discover_skills(self.tmp_dir)
            self.assertIn("shared-skill", skills)
            # Workspace skill must override global skill
            self.assertEqual(skills["shared-skill"]["scope"], "workspace")
            self.assertEqual(skills["shared-skill"]["description"], "Workspace version.")

    def test_get_skills_prompt(self):
        ws_skill_dir = Path(self.tmp_dir) / ".agents" / "skills" / "test-skill"
        ws_skill_dir.mkdir(parents=True)
        (ws_skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: Prompt test skill.\n---\nBody",
            encoding="utf-8"
        )
        fake_home = Path(self.tmp_dir) / "fake_home"
        with patch("pathlib.Path.home", return_value=fake_home):
            prompt = borrow.get_skills_prompt(self.tmp_dir)
            self.assertIn("test-skill [workspace]", prompt)
            self.assertIn("Prompt test skill.", prompt)
            self.assertIn("load_skill", prompt)

    def test_tool_load_skill_success(self):
        ws_skill_dir = Path(self.tmp_dir) / ".borrow" / "skills" / "tool-skill"
        scripts_dir = ws_skill_dir / "scripts"
        scripts_dir.mkdir(parents=True)
        (ws_skill_dir / "SKILL.md").write_text(
            "---\nname: tool-skill\ndescription: Tool skill desc.\n---\nRun step 1 and step 2.",
            encoding="utf-8"
        )
        (scripts_dir / "verify.sh").write_text("#!/bin/bash\nexit 0", encoding="utf-8")

        fake_home = Path(self.tmp_dir) / "fake_home"
        with patch("pathlib.Path.home", return_value=fake_home):
            out = borrow.tool_load_skill("tool-skill")
            self.assertIn("Skill: tool-skill", out)
            self.assertIn("Run step 1 and step 2.", out)
            self.assertIn("scripts/: verify.sh", out)

    def test_tool_load_skill_not_found(self):
        fake_home = Path(self.tmp_dir) / "fake_home"
        with patch("pathlib.Path.home", return_value=fake_home):
            out = borrow.tool_load_skill("nonexistent-skill")
            self.assertIn("Error: Skill 'nonexistent-skill' not found.", out)

    def test_tool_load_skill_path_traversal(self):
        out = borrow.tool_load_skill("../../secrets")
        self.assertIn("Error: Invalid skill name", out)

    def test_cmd_skill_init_and_listing(self):
        borrow.cmd_skill_init("scaffolded-skill")
        skill_file = Path(self.tmp_dir) / ".borrow" / "skills" / "scaffolded-skill" / "SKILL.md"
        self.assertTrue(skill_file.exists())

        # Test invalid name
        with self.assertRaises(borrow.UserError):
            borrow.cmd_skill_init("invalid/name")

        # Test duplicate init
        with self.assertRaises(borrow.UserError):
            borrow.cmd_skill_init("scaffolded-skill")

        # Test cmd_skills listing
        fake_home = Path(self.tmp_dir) / "fake_home"
        with patch("pathlib.Path.home", return_value=fake_home):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                borrow.cmd_skills()
                self.assertIn("scaffolded-skill", mock_out.getvalue())

    def test_cmd_skill_view_instructions(self):
        ws_skill_dir = Path(self.tmp_dir) / ".borrow" / "skills" / "view-me"
        ws_skill_dir.mkdir(parents=True)
        (ws_skill_dir / "SKILL.md").write_text(
            "---\nname: view-me\ndescription: View description.\n---\nDetailed view body",
            encoding="utf-8"
        )
        fake_home = Path(self.tmp_dir) / "fake_home"
        with patch("pathlib.Path.home", return_value=fake_home):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                args = borrow.parse_args("skill", "view-me")
                borrow.cmd_skill(args)
                output = mock_out.getvalue()
                self.assertIn("Skill: view-me", output)
                self.assertIn("Detailed view body", output)

    def test_completer_skills(self):
        ws_skill_dir = Path(self.tmp_dir) / ".borrow" / "skills" / "deploy-prod"
        ws_skill_dir.mkdir(parents=True)
        (ws_skill_dir / "SKILL.md").write_text(
            "---\nname: deploy-prod\ndescription: Deploy to production.\n---\nBody",
            encoding="utf-8"
        )
        fake_home = Path(self.tmp_dir) / "fake_home"
        with patch("pathlib.Path.home", return_value=fake_home):
            completer = borrow.make_completer()
            doc = Document("/skill ")
            completions = [c.text for c in completer.get_completions(doc, None)]
            self.assertIn("deploy-prod ", completions)

            doc_filter = Document("/skill dep")
            filtered = [c.text for c in completer.get_completions(doc_filter, None)]
            self.assertEqual(filtered, ["deploy-prod "])


if __name__ == "__main__":
    unittest.main()
