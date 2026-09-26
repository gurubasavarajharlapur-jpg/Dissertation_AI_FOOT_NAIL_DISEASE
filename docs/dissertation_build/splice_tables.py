"""Build the condensed chapters by pulling table definitions from the long ones.

The condensed chapters carry the argument; the tables carry the measured
results. Copying the table source verbatim out of the long chapter, rather than
retyping it, is what guarantees that shortening the dissertation cannot alter a
number. A missing or duplicated table id fails the build.
"""
import re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def tables_in(source: str) -> dict[str, str]:
    """Map table number -> the exact `cap(...)` + `T(...)` push statements."""
    out = {}
    for m in re.finditer(r"\w+\.push\(cap\('([\d.]+)',", source):
        num = m.group(1)
        start = m.start()
        # the T(...) push that follows the caption, to its closing "));"
        t_start = re.search(r"\w+\.push\(T\(", source[start:]).start() + start
        depth, i = 0, t_start
        while True:
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        end = source.index(";", i) + 1
        # ch5 numbers figures and tables in separate sequences, so a caption
        # number can legitimately appear twice; the table caption comes first.
        if num not in out:
            out[num] = source[start:end].replace("\nch1.push(", "\nc.push(").replace("\nch6.push(", "\nc.push(").replace("\nch7.push(", "\nc.push(")
        if not out[num].lstrip().startswith(("c.push", "ch1.push", "ch6.push", "ch7.push")):
            raise SystemExit(f"table {num} extracted badly")
    return out


def build(short_path: Path, long_path: Path, out_path: Path) -> None:
    short = short_path.read_text()
    available = tables_in(long_path.read_text())
    used = set()

    def swap(m):
        num, renumber, data_only = m.group(1), m.group(2), bool(m.group(0).startswith("//@TABLEDATA"))
        if num not in available:
            raise SystemExit(f"{short_path.name}: no table {num} in {long_path.name}")
        used.add(num)
        block = available[num]
        for old_name in ("ch1.push(", "ch6.push(", "ch7.push("):
            block = block.replace(old_name, "c.push(")
        if data_only:
            # only the T(...) push; the caller supplies its own caption
            return block[re.search(r"c\.push\(T\(", block).start():]
        if renumber:
            head, rest = block.split("',", 1)
            block = head[:head.rindex("'")] + "'" + renumber + "'," + rest
        return block

    out = re.sub(r"^//@TABLE(?:DATA)? ([\d.]+)(?: AS ([\d.A-Z]+))?$", swap, short, flags=re.M)
    if "//@TABLE" in out:
        raise SystemExit(f"{short_path.name}: an @TABLE marker was not replaced")
    out_path.write_text(out)
    print(f"{out_path.name}: {len(used)} tables spliced "
          f"({', '.join(sorted(used, key=lambda s: [int(p) for p in s.split('.')]))})")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        if n == "app":   # appendices draw their tables from the long Chapter 5
            build(HERE / "appendicess.js", HERE / "ch5.js", HERE / "appendices.js")
        else:
            build(HERE / f"ch{n}s.js", HERE / f"ch{n}.js", HERE / f"ch{n}_mod.js")
