from __future__ import annotations

import re
from dataclasses import dataclass, field

FORBIDDEN = re.compile(
    r"\\(documentclass|usepackage|begin\{document\}|end\{document\}|input|include|write18|immediate)\b"
)

ALLOWED_ENVIRONMENTS = {
    "align", "align*", "aligned", "array", "cases", "center", "description",
    "definition", "enumerate", "equation", "equation*", "example", "figure",
    "gather", "gather*", "itemize", "lemma", "matrix", "bmatrix", "pmatrix",
    "vmatrix", "note", "proof", "proposition", "quote", "remark", "split",
    "statement", "tabular", "theorem", "verbatim",
}

_ENV = re.compile(r"\\(begin|end)\{([^}]+)\}")
_VERB = re.compile(r"\\verb\|[^|]*\||\\begin\{verbatim\}.*?\\end\{verbatim\}", re.DOTALL)
_ESCAPED = re.compile(r"\\[\\{}$&#%_]")

_BARE_ENV_COMMAND = re.compile(
    r"(?<!\\)\\(note|remark|example)\b[ \t]*\n?(?!\{)(.+?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)


@dataclass
class Report:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _strip_noise(text: str) -> str:
    text = _VERB.sub(" ", text)
    text = re.sub(r"(?<!\\)%.*?$", "", text, flags=re.MULTILINE)
    return _ESCAPED.sub(" ", text)


def check(fragment: str) -> Report:
    report = Report()
    probe = _strip_noise(fragment)

    if FORBIDDEN.search(probe):
        report.fail("fragment contains preamble-only commands")

    if probe.count("{") != probe.count("}"):
        report.fail(f"unbalanced braces ({probe.count('{')} open, {probe.count('}')} close)")

    inline = len(re.findall(r"(?<!\$)\$(?!\$)", probe))
    if inline % 2:
        report.fail("odd number of `$` delimiters")

    if probe.count(r"\[") != probe.count(r"\]"):
        report.fail("unbalanced display-math delimiters")

    stack: list[str] = []
    for kind, env in _ENV.findall(probe):
        base = env.strip()
        if base not in ALLOWED_ENVIRONMENTS:
            report.warn(f"unexpected environment `{base}`")
        if kind == "begin":
            stack.append(base)
        elif not stack or stack.pop() != base:
            report.fail(f"mismatched \\end{{{base}}}")
    if stack:
        report.fail(f"unclosed environment(s): {', '.join(stack)}")

    return report


def sanitise(fragment: str) -> str:
    text = fragment.strip()

    text = re.sub(r"^```(?:latex|tex)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    body = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", text, re.DOTALL)
    if body:
        text = body.group(1).strip()
    text = FORBIDDEN.sub("", text)

    text = re.sub(r"\$\$(.+?)\$\$", r"\\[\1\\]", text, flags=re.DOTALL)

    text = _BARE_ENV_COMMAND.sub(
        lambda m: f"\\begin{{{m.group(1)}}}\n{m.group(2).strip()}\n\\end{{{m.group(1)}}}",
        text,
    )

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def repair_or_escape(fragment: str) -> tuple[str, Report]:
    cleaned = sanitise(fragment)
    report = check(cleaned)
    if report.ok:
        return cleaned, report
    escaped = cleaned.replace("\\", "\\textbackslash ")
    for ch in "{}$&#_%":
        escaped = escaped.replace(ch, "\\" + ch)
    fallback = (
        "\\begin{quote}\\small\\itshape\n"
        "[автоматическая проверка LaTeX не пройдена: "
        + "; ".join(report.errors)
        + "]\\par\n"
        + escaped
        + "\n\\end{quote}"
    )
    return fallback, report
