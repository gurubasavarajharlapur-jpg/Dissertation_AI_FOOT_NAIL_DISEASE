# Dissertation document build

Generates `Dissertation_COMPLETE.docx`: Chapters 1 and 3 to 7 written from the
measured results in `../findings_and_analysis.md`, six figures drawn from those
same results, and the author's separately submitted and graded Chapter 2 spliced
in at the OOXML level so its tables, figures and citation formatting survive.

Chapter 2 is not in this repository. It is graded coursework and is supplied at
build time.

## Build

```bash
npm install                              # docx-js only
pip install matplotlib pillow            # figures
python3 make_figures.py                  # -> figs/*.png  (committed, so optional)
python3 splice_tables.py 1 3 4 5 6 7 app # -> chN_mod.js, appendices.js
node assemble.js                         # -> Dissertation_FULL_DRAFT.docx
python3 merge_ch2.py                     # -> Dissertation_COMPLETE.docx
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
| `lib.js` | Shared docx helpers: paragraph, heading, table, caption, figure, embedded image |
| `ch1s.js` … `ch7s.js` | The chapters as written. `//@TABLE n` markers stand in for tables |
|  `ch1.js`, `ch3.js` … `ch7.js` | The earlier long-form chapters, kept **only** as the source of the table definitions the markers pull in |
| `appendicess.js` | Appendices A to E, using `//@TABLEDATA n` for tables moved out of the chapters |
| `splice_tables.py` | Copies each table's `cap(...)` + `T(...)` source verbatim into the marker's place |
| `make_figures.py` | Draws the six figures from the recorded results |
| `assemble.js` | Front matter, table of contents, chapters, references, appendices |
| `merge_ch2.py` | Splices Chapter 2 in: copies its body, remaps image relationships, copies media, shifts its heading levels down one so its sections sit under Chapter 2 in the TOC |

**Why the splicer exists.** The chapters were cut from 33,000 words to 20,600 to
meet a word limit. Retyping a table during a cut like that is how a number gets
changed by accident, so the tables are copied from the long chapters
mechanically and a missing or duplicated table id fails the build. Edit
`chNs.js`, never `chN_mod.js` — the latter is generated.

Every figure quoted in Chapter 5 comes from `../findings_and_analysis.md`, which
is the single source of truth for measured results. Change a number there first.

## Word count

Chapters 1 to 7 total about 20,600 words, of which Chapter 2 is 10,846. The
front matter, references and appendices add roughly 2,700 more.
