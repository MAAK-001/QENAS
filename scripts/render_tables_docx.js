// Renders results/tables/tables_spec.json (built by scripts/paper_tables.py) into results/tables/QENAS Tables.docx.
// Needs the npm package "docx" (9.x): run with NODE_PATH pointing to a node_modules folder that contains it.
// Layout: A4 portrait, Times New Roman, IEEE-style three-line tables (top rule, header rule, bottom rule).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, BorderStyle, AlignmentType,
  VerticalAlign, Footer, PageNumber, HeadingLevel,
} = require("docx");

const DIR = path.join(__dirname, "..", "results", "tables");
const spec = JSON.parse(fs.readFileSync(path.join(DIR, "tables_spec.json"), "utf8"));
const OUT = path.join(DIR, "QENAS Tables.docx");

const FONT = "Times New Roman";
const BODY = 20, TAB = 18, NOTE = 16;                 // half-points: 10, 9 and 8 pt
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1134;   // A4, 2 cm margins
const CONTENT_W = PAGE_W - 2 * MARGIN;

const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const THICK = { style: BorderStyle.SINGLE, size: 12, color: "000000" };
const THIN = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const HAIR = { style: BorderStyle.SINGLE, size: 2, color: "9A9A9A" };

function text(t, size, o = {}) {
  return new TextRun({ text: String(t), size, font: FONT, bold: !!o.b, italics: !!o.i });
}

function widths(t) {
  const n = t.header.length;
  const rel = t.widths && t.widths.length === n ? t.widths : Array(n).fill(1);
  const tot = rel.reduce((a, b) => a + b, 0);
  const w = rel.map((x) => Math.floor((x / tot) * CONTENT_W));
  w[n - 1] += CONTENT_W - w.reduce((a, b) => a + b, 0);
  return w;
}

function makeTable(t) {
  const w = widths(t);
  const n = w.length;
  const cell = (d, k, borders, header) => {
    const span = d.span || 1;
    const width = w.slice(k, k + span).reduce((a, b) => a + b, 0);
    const align = d.al === "l" ? AlignmentType.LEFT : AlignmentType.CENTER;
    return new TableCell({
      width: { size: width, type: WidthType.DXA }, columnSpan: span > 1 ? span : undefined, borders,
      verticalAlign: header ? VerticalAlign.BOTTOM : VerticalAlign.CENTER,
      margins: { top: 30, bottom: 30, left: 60, right: 60 },
      children: [new Paragraph({ alignment: align, spacing: { after: 0, line: 240 },
        children: [text(d.t, TAB, { b: header ? true : d.b, i: d.i })] })],
    });
  };
  const head = new TableRow({ tableHeader: true, cantSplit: true, children: t.header.map((d, k) =>
    cell(d, k, { top: THICK, bottom: THIN, left: NONE, right: NONE }, true)) });
  const last = t.rows.length - 1;
  const body = t.rows.map((r, i) => {
    const isSub = r.length === 1 && (r[0].span || 1) === n;
    const borders = { top: isSub && i > 0 ? HAIR : NONE, bottom: i === last ? THICK : NONE, left: NONE, right: NONE };
    let k = 0;
    const cells = r.map((d) => { const x = cell(d, k, borders, false); k += d.span || 1; return x; });
    return new TableRow({ cantSplit: true, children: cells });
  });
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: w, rows: [head, ...body] });
}

const children = [
  new Paragraph({ heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: spec.title, font: FONT, size: 32, bold: true })] }),
];
for (const p of spec.intro) {
  children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 },
    children: [text(p, BODY)] }));
}
for (const t of spec.tables) {
  children.push(new Paragraph({ keepNext: true, keepLines: true, spacing: { before: 360, after: 60 },
    children: [text(`Table ${t.num}. `, BODY, { b: true }), text(t.title, BODY, { b: true })] }));
  children.push(new Paragraph({ keepNext: true, keepLines: true, alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 100 }, children: [text(t.desc, TAB)] }));
  children.push(makeTable(t));
  t.notes.forEach((nt, i) => children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED,
    spacing: { before: i === 0 ? 60 : 0, after: i === t.notes.length - 1 ? 120 : 20 }, children: [text(nt, NOTE)] })));
}

const doc = new Document({
  creator: "QENAS study", title: spec.title,
  styles: { default: { document: { run: { font: FONT, size: BODY } } } },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN,
      left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ children: [PageNumber.CURRENT], size: NOTE, font: FONT })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("written", OUT); });
