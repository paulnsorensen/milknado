from __future__ import annotations

import re

_QUOTED_START = re.compile(
    r"(?i)((?:[\"']?(?:api[_-]?key|password|token|client[_-]?secret|secret|"
    + r"authorization)[\"']?[ \t]*[:=][ \t]*|bearer[ \t]+))([\"'])"
)
_AUTH_HEADER = re.compile(r"(?im)(\bauthorization[ \t]*[:=][ \t]*)[^\r\n]*")
_SECRET = re.compile(
    r"(?i)(\b(?:bearer[ \t]+|api[_-]?key[ \t]*[=:][ \t]*|"
    + r"password[ \t]*[=:][ \t]*|token[ \t]*[=:][ \t]*|"
    + r"client[_-]?secret[ \t]*[=:][ \t]*|secret[ \t]*[=:][ \t]*))"
    + r"(?![\"'])[^\s,;]+"
    + r"|\b(?:sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9_]{8,})\b"
)


def _redact_quoted(value: str) -> str:
    parts: list[str] = []
    position = 0
    while match := _QUOTED_START.search(value, position):
        parts.append(value[position : match.end()])
        quote = match.group(2)
        cursor = match.end()
        while cursor < len(value):
            if value[cursor] == "\\":
                cursor += 2
            elif value[cursor] == quote:
                break
            else:
                cursor += 1
        parts.append("[REDACTED]")
        if cursor >= len(value):
            position = cursor
            break
        parts.append(quote)
        position = cursor + 1
    parts.append(value[position:])
    return "".join(parts)


def redact_control_text(value: str) -> str:
    quoted = _redact_quoted(value)
    headers = _AUTH_HEADER.sub(lambda match: match.group(1) + "[REDACTED]", quoted)
    return _SECRET.sub(lambda match: (match.group(1) or "") + "[REDACTED]", headers)
