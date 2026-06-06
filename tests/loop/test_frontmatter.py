"""Tests for ralphify._frontmatter — YAML frontmatter parsing and serialization."""

import pytest

from milknado.loop._frontmatter import (
    RALPH_MARKER,
    _extract_frontmatter_block,
    parse_frontmatter,
    serialize_frontmatter,
)


class TestRalphMarker:
    def test_marker_value(self):
        assert RALPH_MARKER == "RALPH.md"


class TestExtractFrontmatterBlock:
    def test_valid_frontmatter(self):
        text = "---\nagent: claude\n---\nHello"
        fm, body = _extract_frontmatter_block(text)
        assert fm == "agent: claude"
        assert body == "Hello"

    def test_no_frontmatter(self):
        text = "Just a body"
        fm, body = _extract_frontmatter_block(text)
        assert fm == ""
        assert body == "Just a body"

    def test_no_closing_delimiter(self):
        text = "---\nagent: claude\nNo closing"
        fm, body = _extract_frontmatter_block(text)
        assert fm == ""
        assert body == text

    def test_empty_string(self):
        fm, body = _extract_frontmatter_block("")
        assert fm == ""
        assert body == ""

    def test_only_delimiters(self):
        text = "---\n---"
        fm, body = _extract_frontmatter_block(text)
        assert fm == ""
        assert body == ""

    def test_multiline_frontmatter(self):
        text = "---\nagent: claude\ncommands:\n  - name: test\n    run: pytest\n---\nPrompt here"
        fm, body = _extract_frontmatter_block(text)
        assert "agent: claude" in fm
        assert "commands:" in fm
        assert body == "Prompt here"

    def test_body_stripped(self):
        text = "---\nagent: claude\n---\n\n  Hello  \n\n"
        _, body = _extract_frontmatter_block(text)
        assert body == "Hello"

    def test_indented_opening_delimiter_not_treated_as_frontmatter(self):
        """Leading whitespace before the opening '---' means no frontmatter —
        consistent with the closing delimiter which also requires column 0."""
        text = "  ---\nagent: claude\n---\nBody"
        fm, body = _extract_frontmatter_block(text)
        assert fm == ""
        assert body == text

    def test_indented_triple_dash_not_treated_as_closing_delimiter(self):
        """An indented '---' inside a YAML block scalar must not be mistaken
        for the closing frontmatter delimiter.  Only unindented '---' at
        column 0 should close the frontmatter block."""
        text = "---\nagent: claude\nnotes: |\n  first\n  ---\n  third\n---\nBody"
        fm, body = _extract_frontmatter_block(text)
        assert "notes:" in fm
        assert "---" in fm  # the indented --- is part of frontmatter
        assert body == "Body"


class TestParseFrontmatter:
    def test_basic_parsing(self):
        text = "---\nagent: claude\n---\nDo the thing"
        fm, body = parse_frontmatter(text)
        assert fm == {"agent": "claude"}
        assert body == "Do the thing"

    def test_no_frontmatter(self):
        text = "Just a prompt"
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert body == "Just a prompt"

    def test_commands_list(self):
        text = "---\nagent: claude\ncommands:\n  - name: tests\n    run: pytest\n---\nPrompt"
        fm, _ = parse_frontmatter(text)
        assert fm["commands"] == [{"name": "tests", "run": "pytest"}]

    def test_args_list(self):
        text = "---\nagent: claude\nargs:\n  - dir\n  - focus\n---\nPrompt"
        fm, _ = parse_frontmatter(text)
        assert fm["args"] == ["dir", "focus"]

    def test_html_comments_stripped_from_body(self):
        text = "---\nagent: claude\n---\nBefore <!-- hidden --> after"
        _, body = parse_frontmatter(text)
        assert body == "Before  after"

    def test_multiline_html_comment_stripped(self):
        text = "---\nagent: claude\n---\nBefore <!--\nmultiline\ncomment\n--> after"
        _, body = parse_frontmatter(text)
        assert body == "Before  after"

    def test_html_comment_inside_fenced_code_block_preserved(self):
        """HTML comments inside fenced code blocks must NOT be stripped —
        they are part of the code example, not hidden notes."""
        text = (
            "---\nagent: claude\n---\n"
            "Here is code:\n\n"
            "```html\n"
            "<div><!-- keep me --><span>hi</span></div>\n"
            "```\n\n"
            "<!-- strip me -->\n"
            "Done."
        )
        _, body = parse_frontmatter(text)
        assert "<!-- keep me -->" in body
        assert "<!-- strip me -->" not in body

    def test_html_comment_inside_tilde_fence_preserved(self):
        """Tilde-fenced code blocks must also protect HTML comments."""
        text = "---\nagent: claude\n---\n~~~\n<!-- preserved -->\n~~~\n<!-- removed -->"
        _, body = parse_frontmatter(text)
        assert "<!-- preserved -->" in body
        assert "<!-- removed -->" not in body

    def test_html_comment_inside_four_backtick_fence_preserved(self):
        """Four-backtick fences (used to embed ``` inside code) must
        protect HTML comments inside them.

        When a single ``` appears inside a ```` fence, the comment
        between the ``` and the closing ```` must not be stripped."""
        text = (
            "---\nagent: claude\n---\n"
            "````\n"
            "```\n"
            "code\n"
            "<!-- keep this comment -->\n"
            "````\n"
            "<!-- strip this comment -->"
        )
        _, body = parse_frontmatter(text)
        assert "<!-- keep this comment -->" in body
        assert "<!-- strip this comment -->" not in body

    def test_html_comment_after_longer_closing_fence_stripped(self):
        """When the closing fence has more backticks than the opening, the
        extra backticks must NOT be treated as a new opening fence.

        In CommonMark, a closing fence need only have *at least* as many
        backticks as the opening.  A 6-backtick line closes a 3-backtick
        fence; everything after is normal text and comments should be
        stripped.

        Regression: the backreference in the fence regex consumed only as
        many backticks as the opening, leaving the extras to be
        misinterpreted as a second opening fence when followed by another
        line of 3+ backticks later in the text."""
        text = "---\nagent: claude\n---\n```\ncode\n``````\n<!-- strip this -->\n```\nmore text"
        _, body = parse_frontmatter(text)
        assert "<!-- strip this -->" not in body

    def test_html_comment_after_inline_backticks_inside_fence_preserved(self):
        """When a ``` fence contains inline ``` characters on a content
        line, the inline backticks must NOT be treated as the closing
        fence.  HTML comments that follow the inline backticks but are
        still inside the real fence must be preserved."""
        text = (
            "---\nagent: claude\n---\n"
            "```\n"
            "code with ``` inline\n"
            "<!-- keep this -->\n"
            "```\n"
            "<!-- strip this -->"
        )
        _, body = parse_frontmatter(text)
        assert "<!-- keep this -->" in body
        assert "<!-- strip this -->" not in body

    def test_invalid_yaml_raises_value_error(self):
        text = "---\n: invalid: yaml: [unclosed\n---\nBody"
        with pytest.raises(ValueError, match="Invalid YAML"):
            parse_frontmatter(text)

    def test_empty_frontmatter_returns_empty_dict(self):
        text = "---\n---\nBody"
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert body == "Body"

    def test_yaml_comment_only_frontmatter_returns_empty_dict(self):
        text = "---\n# just a YAML comment\n---\nBody"
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert body == "Body"

    def test_block_scalar_with_triple_dash_parsed_correctly(self):
        """A YAML block scalar containing '---' must not break frontmatter parsing."""
        text = "---\nagent: claude\nnotes: |\n  first\n  ---\n  third\n---\nBody"
        fm, body = parse_frontmatter(text)
        assert fm["agent"] == "claude"
        assert "---" in fm["notes"]
        assert body == "Body"

    def test_utf8_bom_does_not_break_frontmatter(self):
        """Files saved with a UTF-8 BOM (common on Windows) must still
        have their frontmatter parsed correctly."""
        text = "\ufeff---\nagent: claude\n---\nDo the thing"
        fm, body = parse_frontmatter(text)
        assert fm == {"agent": "claude"}
        assert body == "Do the thing"

    def test_non_dict_frontmatter_raises_value_error(self):
        text = "---\n- item1\n- item2\n---\nBody"
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            parse_frontmatter(text)

    def test_scalar_frontmatter_raises_value_error(self):
        text = "---\njust a string\n---\nBody"
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            parse_frontmatter(text)


class TestSerializeFrontmatter:
    def test_roundtrip(self):
        original_fm = {"agent": "claude"}
        original_body = "Do the thing"
        serialized = serialize_frontmatter(original_fm, original_body)
        fm, body = parse_frontmatter(serialized)
        assert fm == original_fm
        assert body == original_body

    def test_empty_frontmatter(self):
        result = serialize_frontmatter({}, "Just a body")
        assert result == "Just a body"
        assert "---" not in result

    def test_includes_delimiters(self):
        result = serialize_frontmatter({"agent": "claude"}, "Prompt")
        assert result.startswith("---\n")
        assert "\n---\n" in result

    def test_body_preserved(self):
        result = serialize_frontmatter({"agent": "claude"}, "My prompt text")
        assert result.endswith("My prompt text")

    def test_roundtrip_empty_frontmatter_body_starts_with_delimiter(self):
        """serialize then parse must round-trip when frontmatter is empty and
        the body starts with '---' (which looks like a frontmatter block)."""
        original_body = "---\nagent: fake\n---\nreal body"
        serialized = serialize_frontmatter({}, original_body)
        fm, body = parse_frontmatter(serialized)
        assert fm == {}
        assert body == original_body

    def test_roundtrip_empty_frontmatter_body_is_double_delimiter(self):
        """Empty frontmatter with a body that is exactly '---\\n---' must
        round-trip without the body being consumed as empty frontmatter."""
        original_body = "---\n---"
        serialized = serialize_frontmatter({}, original_body)
        fm, body = parse_frontmatter(serialized)
        assert fm == {}
        assert body == original_body
