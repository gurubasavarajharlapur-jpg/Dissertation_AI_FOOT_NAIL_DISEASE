const { Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow,
        TableCell, WidthType, ShadingType, PageBreak } = require('docx');
const ACCENT = '1F4E79', GREY = '595959';
const P = (t, o = {}) => new Paragraph({ spacing: { after: o.after ?? 160, line: 300 },
  alignment: o.left ? undefined : AlignmentType.JUSTIFIED,
  children: [new TextRun({ text: t, size: 22, italics: o.i, bold: o.b, font: 'Calibri' })] });
const TODO = t => new Paragraph({ spacing: { before: 120, after: 160, line: 300 },
  shading: { type: ShadingType.CLEAR, fill: 'FFF4E5' },
  children: [ new TextRun({ text: 'CHECK:  ', bold: true, size: 20, color: 'B26500', font: 'Calibri' }),
              new TextRun({ text: t, size: 20, italics: true, font: 'Calibri' }) ] });
const CH = (n, t) => ([
  new Paragraph({ spacing: { after: 280 }, pageBreakBefore: true,
    children: [new TextRun({ text: `Chapter ${n}`, bold: true, size: 36, color: ACCENT, font: 'Calibri' })] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { after: 360 },
    children: [new TextRun({ text: t, bold: true, size: 36, color: ACCENT, font: 'Calibri' })] }),
]);
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 340, after: 150 },
  children: [new TextRun({ text: t, font: 'Calibri' })] });
const H3 = t => new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 260, after: 120 },
  children: [new TextRun({ text: t, font: 'Calibri' })] });
const cap = (n, t) => new Paragraph({ spacing: { before: 240, after: 90 },
  children: [ new TextRun({ text: `Table ${n}.  `, bold: true, size: 20, font: 'Calibri' }),
              new TextRun({ text: t, size: 20, font: 'Calibri' }) ] });
const fcap = (n, t) => new Paragraph({ spacing: { before: 90, after: 200 }, alignment: AlignmentType.CENTER,
  children: [ new TextRun({ text: `Figure ${n}.  `, bold: true, size: 20, font: 'Calibri' }),
              new TextRun({ text: t, size: 20, font: 'Calibri' }) ] });
const FIG = t => new Paragraph({ spacing: { before: 180, after: 40 }, alignment: AlignmentType.CENTER,
  shading: { type: ShadingType.CLEAR, fill: 'F2F2F2' },
  children: [ new TextRun({ text: `[ INSERT FIGURE — ${t} ]`, size: 20, color: GREY, italics: true, font: 'Calibri' }) ] });
const cell = (t, w, o = {}) => new TableCell({ width: { size: w, type: WidthType.DXA },
  shading: o.head ? { type: ShadingType.CLEAR, fill: 'E8EEF4' } : undefined,
  margins: { top: 55, bottom: 55, left: 100, right: 100 },
  children: [new Paragraph({ spacing: { after: 0 },
    children: [new TextRun({ text: t, bold: o.head, size: 18, font: 'Calibri' })] })] });
const T = (widths, rows) => new Table({ columnWidths: widths,
  width: { size: widths.reduce((a,b)=>a+b,0), type: WidthType.DXA },
  rows: rows.map((r,i)=> new TableRow({ tableHeader: i===0,
    children: r.map((cc,j)=> cell(String(cc), widths[j], { head: i===0 })) })) });
const BREAK = () => new Paragraph({ children: [new PageBreak()] });
module.exports = { P, TODO, CH, H2, H3, cap, fcap, FIG, T, BREAK, ACCENT, GREY };
