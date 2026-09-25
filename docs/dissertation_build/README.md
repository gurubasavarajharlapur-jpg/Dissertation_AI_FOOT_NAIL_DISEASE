# Dissertation document build

Generates `Dissertation_COMPLETE.docx` — Chapters 1 and 3 to 7 written from the
measured results in `../findings_and_analysis.md`, with the author's separately
submitted and graded Chapter 2 spliced in at the OOXML level so that its tables,
figures and citation formatting survive intact.

Chapter 2 is not in this repository. It is the author's graded coursework
document and has to be supplied at build time.

## Build

```bash
npm install                        # docx-js only
node assemble.js                   # -> Dissertation_FULL_DRAFT.docx
python3 merge_ch2.py               # -> Dissertation_COMPLETE.docx
```

`merge_ch2.py` expects the Chapter 2 source unpacked at `ch2src/unpacked/`:

```bash
mkdir -p ch2src/unpacked && cd ch2src/unpacked
unzip -o ../../"<the graded Chapter 2>.docx"
```

Open the result in Word and update the Table of Contents field (Ctrl+A, F9).
The TOC is a field, so it is empty until Word populates it.

## Layout

| File | Contents |
|---|---|
| `lib.js` | Shared docx helpers: paragraph, heading, table, caption, figure placeholder, the orange "to complete" box |
| `ch1.js`, `ch6.js`, `ch7.js` | Chapters that export their block array directly |
| `ch3.js`, `ch4.js`, `ch5.js` | Chapters that also build standalone per-chapter .docx files |
| `mkmod.py` | Derives `chN_mod.js` from `chN.js` — the same content with the per-chapter document tail removed, which is what `assemble.js` requires |
| `assemble.js` | Front matter, table of contents, all chapters, references, appendices |
| `merge_ch2.py` | Splices Chapter 2 in: copies its body elements, remaps its image relationships, copies its media, and shifts its two heading levels down one so its sections sit under Chapter 2 in the TOC |

Edit `chN.js`, never `chN_mod.js` — the latter is generated and will be
overwritten. After editing, run `python3 mkmod.py 3 4 5` before `assemble.js`.

Every figure quoted in these chapters comes from `../findings_and_analysis.md`,
which is the single source of truth for measured results. Change a number there
first, then here.
