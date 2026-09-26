"""Splice the author's graded Chapter 2 into the assembled dissertation.

Chapter 2 was written and submitted separately and must not be retyped: doing so
would lose its 15 tables, its 9 figures and its citation formatting. So the
chapter is merged at the OOXML level instead. Its body elements are copied out of
the source .docx, its image relationships are remapped onto fresh ids in the
target, its media files are copied across, and its two heading levels are shifted
down one so that its sections sit under Chapter 2 in the table of contents rather
than alongside the chapter titles. Nothing in the chapter's own text, tables or
figures is altered.

Run after assemble.js. Writes Dissertation_COMPLETE.docx.
"""
import re, shutil, zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).resolve().parent
SRC = HERE / "ch2src" / "unpacked"
DRAFT = HERE / "Dissertation_FULL_DRAFT.docx"
WORK = HERE / "merge" / "mine"
OUT = HERE / "Dissertation_COMPLETE.docx"

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
IMAGE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"

# Every prefix the source may use inside the copied elements, so that serialising
# them yields the prefixes Word expects rather than ET's ns0/ns1 invention.
for prefix, uri in {
    "w": W[1:-1], "r": R[1:-1],
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "wp14": "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing",
    "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}.items():
    ET.register_namespace(prefix, uri)


RUN_IN_HEADING = "2.5.3 Artificial Intelligence in Medical Image Analysis"


def split_run_in_heading(kids):
    """Give heading 2.5.3 its own paragraph.

    In the source document that heading is the last run of the paragraph
    holding the Figure 2.5 label and image, so Word does not see it as a
    heading and it is missing from the contents page. The run is moved into a
    new paragraph styled like its siblings 2.5.1, 2.5.2 and 2.5.4; the figure
    label and image stay where they are. Nothing is retyped, so the author's
    text and formatting are unchanged.
    """
    for i, k in enumerate(kids):
        if k.tag != W + "p":
            continue
        if RUN_IN_HEADING not in "".join(t.text or "" for t in k.iter(W + "t")):
            continue
        runs = k.findall(W + "r")
        head = [r for r in runs
                if RUN_IN_HEADING in "".join(t.text or "" for t in r.iter(W + "t"))]
        if len(head) != 1 or head[0] is runs[0]:
            return kids            # already a heading of its own, or not this shape
        para = ET.Element(W + "p")
        style = ET.SubElement(ET.SubElement(para, W + "pPr"), W + "pStyle")
        style.set(W + "val", "Heading2")   # remapped to Heading3 with the others
        k.remove(head[0])
        para.append(head[0])
        return kids[:i + 1] + [para] + kids[i + 1:]
    raise SystemExit("heading 2.5.3 not found where expected")


def chapter2_elements():
    """The source body children that make up Chapter 2, adjusted for insertion."""
    body = ET.parse(SRC / "word" / "document.xml").getroot().find(W + "body")
    kids = list(body)

    def text_of(el):
        return "".join(t.text or "" for t in el.iter(W + "t")).strip()

    kids = split_run_in_heading(kids)
    start = next(i for i, k in enumerate(kids)
                 if k.tag == W + "p" and text_of(k) == "Chapter 2: Literature Review")
    end = next(i for i, k in enumerate(kids) if k.tag == W + "sectPr")
    out = []
    for k in kids[start + 1:end]:          # the target supplies the chapter title
        style = k.find(f"{W}pPr/{W}pStyle")
        name = style.get(W + "val") if style is not None else ""
        if k.tag == W + "p" and name.startswith("Heading") and not text_of(k):
            continue                        # blank spacer headings; they would
                                            # otherwise appear as empty TOC rows
        # Chapter 2 numbers its sections at Heading1/Heading2; the rest of the
        # dissertation uses Heading2/Heading3 for the same depth.
        for ps in k.iter(W + "pStyle"):
            if ps.get(W + "val") == "Heading2":
                ps.set(W + "val", "Heading3")
        for ps in k.iter(W + "pStyle"):
            if ps.get(W + "val") == "Heading1":
                ps.set(W + "val", "Heading2")
        for rs in k.iter(W + "rStyle"):
            if rs.get(W + "val") == "Heading2Char":
                rs.set(W + "val", "Heading3Char")
        out.append(k)
    return out


def source_image_targets():
    rels = ET.parse(SRC / "word" / "_rels" / "document.xml.rels").getroot()
    return {r.get("Id"): r.get("Target") for r in rels
            if r.get("Type") == IMAGE_REL}


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    with zipfile.ZipFile(DRAFT) as z:
        z.extractall(WORK)

    elements = chapter2_elements()
    images = source_image_targets()

    # Fresh relationship ids after the ones docx-js already used.
    rels_path = WORK / "word" / "_rels" / "document.xml.rels"
    rels_xml = rels_path.read_text()
    used = [int(n) for n in re.findall(r'Id="rId(\d+)"', rels_xml)]
    next_id = max(used) + 1

    # Only the images Chapter 2 actually uses. The source document also contains
    # the project proposal, whose figures must not be dragged in with it.
    referenced = set()
    for el in elements:
        for node in el.iter():
            for attr, value in node.attrib.items():
                if attr.startswith(R) and value in images:
                    referenced.add(value)

    media_dir = WORK / "word" / "media"
    media_dir.mkdir(exist_ok=True)
    added, mapping = [], {}
    for old_id in sorted(referenced, key=lambda i: int(i[3:])):
        name = Path(images[old_id]).name
        shutil.copy(SRC / "word" / images[old_id], media_dir / name)
        new_id = f"rId{next_id}"; next_id += 1
        mapping[old_id] = new_id
        added.append(f'<Relationship Id="{new_id}" Type="{IMAGE_REL}" '
                     f'Target="media/{name}"/>')

    for el in elements:
        for node in el.iter():
            for attr, value in list(node.attrib.items()):
                if attr.startswith(R) and value in mapping:
                    node.set(attr, mapping[value])

    rels_path.write_text(rels_xml.replace("</Relationships>",
                                          "".join(added) + "</Relationships>"))

    # docProps/app.xml carries a stale page/word count; Word recalculates, and
    # leaving a wrong number in the file is worse than leaving none.
    doc_path = WORK / "word" / "document.xml"
    doc = doc_path.read_text()
    chunk = "".join(ET.tostring(el, encoding="unicode") for el in elements)

    # Remove the two placeholder paragraphs and insert the chapter in their place.
    box = re.search(r"<w:p\b[^>]*>(?:(?!</w:p>).)*?PASTE YOUR SUBMITTED CHAPTER 2 HERE"
                    r".*?</w:p>", doc, re.S)
    assert box, "placeholder instruction box not found"
    summary = re.search(r"<w:p\b[^>]*>(?:(?!</w:p>).)*?Chapter 2 reviews the clinical "
                        r"background.*?</w:p>", doc, re.S)
    assert summary, "placeholder summary paragraph not found"
    assert summary.start() >= box.end(), "placeholder paragraphs out of order"
    doc = doc[:box.start()] + chunk + doc[summary.end():]
    doc_path.write_text(doc)

    # Copy in any style definitions Chapter 2 refers to that the target lacks.
    styles_path = WORK / "word" / "styles.xml"
    target_styles = ET.parse(styles_path); troot = target_styles.getroot()
    have = {s.get(W + "styleId") for s in troot.findall(W + "style")}
    want = set()
    for el in elements:
        for node in el.iter():
            if node.tag in (W + "pStyle", W + "rStyle", W + "tblStyle"):
                want.add(node.get(W + "val"))
    src_styles = {s.get(W + "styleId"): s
                  for s in ET.parse(SRC / "word" / "styles.xml").getroot().findall(W + "style")}
    copied = []
    for style_id in sorted(want - have):
        if style_id in src_styles:
            troot.append(src_styles[style_id]); copied.append(style_id)
    if copied:
        target_styles.write(styles_path, xml_declaration=True, encoding="UTF-8")
    print(f"styles copied from Chapter 2: {copied or 'none needed'}")
    print(f"styles referenced but undefined in either document: "
          f"{sorted(want - have - set(src_styles)) or 'none'}")

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(WORK.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(WORK).as_posix())
    print(f"written {OUT.name}: {OUT.stat().st_size // 1024} KB, "
          f"{len(elements)} Chapter 2 elements, {len(added)} images")


if __name__ == "__main__":
    main()
