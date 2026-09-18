from __future__ import annotations

import re

from .common import from_timecode

TIMECODE = re.compile(r"\\ts\{(\d{1,2}:\d{2}(?::\d{2})?)\}")

_THEOREM_NAMES = {
    "definition": "Определение",
    "theorem": "Теорема",
    "statement": "Утверждение",
    "lemma": "Лемма",
    "proposition": "Предложение",
    "example": "Пример",
    "note": "Замечание",
    "remark": "Замечание",
    "proof": "Доказательство",
    "bropilotnote": "Примечание",
}

_DISPLAY_ENVS = ("equation", "equation*", "align", "align*", "gather", "gather*",
                 "multline", "multline*")


def timecodes(latex: str) -> list[tuple[str, float]]:
    seen: dict[str, float] = {}
    for tc in TIMECODE.findall(latex or ""):
        seen.setdefault(tc, from_timecode(tc))
    return list(seen.items())


def _braced(text: str, start: int) -> tuple[str, int]:
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "\\":
            continue
        if ch == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    return text[start + 1 :], len(text)


def _replace_command(text: str, name: str, fn) -> str:
    pattern = re.compile(r"\\" + re.escape(name) + r"\*?\s*\{")
    out, pos = [], 0
    while True:
        m = pattern.search(text, pos)
        if not m:
            out.append(text[pos:])
            return "".join(out)
        arg, end = _braced(text, m.end() - 1)
        out.append(text[pos : m.start()])
        out.append(fn(arg))
        pos = end


def _math_env(match: re.Match) -> str:
    env, body = match.group(1), match.group(2).strip()
    body = re.sub(r"\\label\{[^}]*\}", "", body)
    if env.startswith(("align", "multline")):
        body = f"\\begin{{aligned}}{body}\\end{{aligned}}"
    elif env.startswith("gather"):
        body = f"\\begin{{gathered}}{body}\\end{{gathered}}"
    return f"\n$$\n{body}\n$$\n"


def _lists(text: str) -> str:
    lines_out: list[str] = []
    stack: list[str] = []
    counters: list[int] = []
    for raw in text.split("\n"):
        line = raw
        while True:
            m = re.search(r"\\begin\{(itemize|enumerate)\}(\[[^\]]*\])?|\\end\{(itemize|enumerate)\}|\\item(\[[^\]]*\])?\s*", line)
            if not m:
                break
            before, after = line[: m.start()], line[m.end():]
            if before.strip():
                lines_out.append(before.rstrip())
            if m.group(1):
                stack.append(m.group(1))
                counters.append(0)
                lines_out.append("")
                line = after
            elif m.group(3):
                if stack:
                    stack.pop()
                    counters.pop()
                lines_out.append("")
                line = after
            else:
                indent = "   " * max(0, len(stack) - 1)
                if stack and stack[-1] == "enumerate":
                    counters[-1] += 1
                    bullet = f"{counters[-1]}."
                else:
                    bullet = "-"
                label = m.group(4)
                prefix = f"{indent}{bullet} " + (f"**{label[1:-1]}** " if label else "")
                nxt = re.search(r"\\(begin|end)\{(itemize|enumerate)\}|\\item", after)
                item_text = after[: nxt.start()] if nxt else after
                lines_out.append(prefix + item_text.strip())
                line = after[nxt.start():] if nxt else ""
        if line.strip() or not stack:
            if stack and line.strip():
                lines_out[-1] = lines_out[-1] + " " + line.strip()
            else:
                lines_out.append(line)
    return "\n".join(lines_out)


_KATEX_MACROS = {
    "eps": r"\varepsilon", "R": r"\mathbb{R}", "E": r"\mathbb{E}",
    "sign": r"\operatorname{sign}", "argmin": r"\operatorname*{arg\,min}",
    "argmax": r"\operatorname*{arg\,max}", "Var": r"\operatorname{Var}",
    "cov": r"\operatorname{cov}", "tr": r"\operatorname{tr}",
}
_MACRO_RE = re.compile(r"\\(" + "|".join(_KATEX_MACROS) + r")(?![A-Za-z])")


def katex_math(formula: str) -> str:
    return _MACRO_RE.sub(lambda m: _KATEX_MACROS[m.group(1)], formula)


_MATH = re.compile(r"\$\$.*?\$\$|(?<!\\)\$.*?(?<!\\)\$", re.S)


def latex_to_markdown(latex: str, *, heading_level: int = 3, timecode_fmt: str = "`{tc}`") -> str:
    text = (latex or "").replace("\r\n", "\n")
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)

    envs = "|".join(re.escape(e) for e in _DISPLAY_ENVS)
    text = re.sub(r"\\begin\{(" + envs + r")\}(.*?)\\end\{\1\}", _math_env, text, flags=re.S)
    text = re.sub(r"\\\[(.*?)\\\]", lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n", text, flags=re.S)
    text = re.sub(r"\\\((.*?)\\\)", lambda m: f"${m.group(1).strip()}$", text, flags=re.S)
    shelf: list[str] = []

    def shelve(m: re.Match) -> str:
        shelf.append(m.group(0))
        return f"\x01{len(shelf) - 1}\x01"

    text = _MATH.sub(shelve, text)

    h = "#" * heading_level
    text = _replace_command(text, "section", lambda a: f"\n{h} {a.strip()}\n")
    text = _replace_command(text, "subsection", lambda a: f"\n{h}# {a.strip()}\n")
    text = _replace_command(text, "subsubsection", lambda a: f"\n**{a.strip()}**\n")
    text = _replace_command(text, "paragraph", lambda a: f"\n**{a.strip()}** ")

    def _timecode(m: re.Match) -> str:
        if not timecode_fmt:
            return ""
        tc = m.group(1)
        return timecode_fmt.format(tc=tc, sec=int(from_timecode(tc))) + " "

    text = TIMECODE.sub(_timecode, text)
    text = _replace_command(text, "bropilotfigure", lambda a: f"\n> 🖼 *Рисунок с доски: {a.strip()}*\n")
    text = _replace_command(text, "bropilotunsure", lambda a: f"{a}⁽?⁾")
    text = _replace_command(text, "textbf", lambda a: f"**{a}**")
    for cmd in ("emph", "textit"):
        text = _replace_command(text, cmd, lambda a: f"*{a}*")
    text = _replace_command(text, "texttt", lambda a: f"`{a}`")
    text = _replace_command(text, "underline", lambda a: a)
    text = re.sub(r"\\(label|ref|eqref|cite|index)\{[^}]*\}", "", text)

    def theorem(m: re.Match) -> str:
        name = _THEOREM_NAMES.get(m.group(1), m.group(1))
        title = f" ({m.group(2)[1:-1]})" if m.group(2) else ""
        return f"\n\n**{name}{title}.** "

    names = "|".join(_THEOREM_NAMES)
    text = re.sub(r"\\begin\{(" + names + r")\}(\[[^\]]*\])?", theorem, text)
    text = re.sub(r"\\end\{proof\}", " ∎\n\n", text)
    text = re.sub(r"\\end\{(" + names + r")\}", "\n\n", text)

    text = _lists(text)

    text = text.replace("\\\\", "  \n").replace("~", " ").replace("---", "—").replace("--", "–")
    text = text.replace("<<", "«").replace(">>", "»")
    text = re.sub(r"\\(noindent|medskip|bigskip|smallskip|newline|par|centering)\b", "", text)
    text = re.sub(r"\\(small|footnotesize|large|Large|normalsize)\b\s*", "", text)
    text = re.sub(r"\\begin\{center\}|\\end\{center\}", "", text)
    for esc in ("%", "&", "_", "#"):
        text = text.replace("\\" + esc, esc)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"\x01(\d+)\x01", lambda m: katex_math(shelf[int(m.group(1))]), text)
    return text.strip()


def lecture_to_markdown(doc, *, video_url: str | None = None) -> str:
    from .common import to_timecode, youtube_url_at

    def fmt_tc(tc: str) -> str:
        if video_url:
            link = youtube_url_at(video_url, from_timecode(tc))
            if link:
                return f"[⏱ {tc}]({link})"
        return f"`⏱ {tc}`"

    parts = [f"# {doc.title or 'Конспект лекции'}", ""]
    if doc.duration:
        parts.append(f"*Длительность записи: {to_timecode(doc.duration)}. "
                     "Конспект собран автоматически и может содержать ошибки распознавания.*")
        parts.append("")
    parts.append("## Содержание")
    for s in doc.segments:
        parts.append(f"- {fmt_tc(to_timecode(s.span.start))} {s.title or 'Раздел'}")
    parts.append("")
    for s in doc.segments:
        if not s.body:
            continue
        body = s.body
        md = TIMECODE.sub(lambda m: "\x00" + m.group(1) + "\x00", body)
        md = latex_to_markdown(md, heading_level=2, timecode_fmt="{tc}")
        md = re.sub(r"\x00([\d:]+)\x00", lambda m: fmt_tc(m.group(1)), md)
        parts.append(md)
        parts.append("")
    return "\n".join(parts).strip() + "\n"
