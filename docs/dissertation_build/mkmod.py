"""Derive chN_mod.js (a module assemble.js can require) from the standalone chN.js.

The standalone script writes a per-chapter .docx; the module form drops the
Document/Packer tail and delegates the chapter title to lib.CH so every chapter
in the assembled document gets identical front matter and a page break.
"""
import re, sys, pathlib

def convert(src: str) -> str:
    m = re.search(
        r"c\.push\(new Paragraph\(\{ spacing: \{ after: \d+ \},\n"
        r"  children: \[new TextRun\(\{ text: 'Chapter (\d+)'.*?\n"
        r"c\.push\(new Paragraph\(\{ spacing: \{ after: 360 \},\n"
        r"  children: \[new TextRun\(\{ text: '([^']+)'.*?\n", src)
    assert m, "chapter title block not found"
    src = src[:m.start()] + f"c.push(...CH({m.group(1)}, '{m.group(2)}'));\n" + src[m.end():]
    i = src.index("const doc = new Document({")
    src = src[:i] + "\nmodule.exports = c;\n"
    return "const { CH } = require('./lib');\n" + src

for n in sys.argv[1:]:
    s = pathlib.Path(f"ch{n}.js").read_text()
    pathlib.Path(f"ch{n}_mod.js").write_text(convert(s))
    print(f"ch{n}_mod.js written")
