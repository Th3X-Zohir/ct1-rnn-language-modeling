// Shared helpers for building the CT1 report with docx-js.
const fs = require("fs");
const path = require("path");
const D = require("docx");
const {
  Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType,
  HeadingLevel, BorderStyle, PageBreak, LevelFormat, TableLayoutType, VerticalAlign,
} = D;

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "results", "figures");
const TAB = path.join(ROOT, "results", "tables");
const BODY_FONT = "Times New Roman";
const HEAD_FONT = "Arial";
const CONTENT_W = 9026; // A4 width 11906 - 2 * 1440 margins (DXA)

let figNo = 0, tabNo = 0;

// inline markup: **bold**, *italic*, `code`
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Courier New", size: 20, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({
  children: runs(text, opts.run || {}), alignment: opts.align ?? AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 120, line: 300 }, ...(opts.para || {}),
});
const headings = [];   // collected for the static table of contents
const H1 = (t) => { headings.push({ text: t, level: 1 }); return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] }); };
const H2 = (t) => { headings.push({ text: t, level: 2 }); return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] }); };
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const bullet = (t, level = 0) => new Paragraph({
  numbering: { reference: "bullets", level }, children: runs(t), spacing: { after: 60, line: 290 },
  alignment: AlignmentType.LEFT,
});
// Passing a reference starts a NEW list (fresh instance -> numbering restarts at 1).
let listInstance = 0;
const numbered = (t, ref) => {
  if (ref) listInstance += 1;
  return new Paragraph({
    numbering: { reference: "numbers", level: 0, instance: listInstance }, children: runs(t),
    spacing: { after: 60, line: 290 },
  });
};

// Static table of contents: entries [{text, level, page}] -> dot-leader lines with right-aligned page numbers.
const { TabStopType, TabStopPosition, Tab } = D;
const tocLines = (entries) => entries.map((e) => new Paragraph({
  spacing: { after: e.level === 1 ? 60 : 30 }, indent: { left: e.level === 1 ? 0 : 400 },
  tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W, leader: "dot" }],
  children: [
    new TextRun({ text: e.text, bold: e.level === 1, size: e.level === 1 ? 22 : 21 }),
    new TextRun({ children: [new Tab()] }),
    new TextRun({ text: String(e.page ?? "00"), bold: e.level === 1, size: e.level === 1 ? 22 : 21 }),
  ],
}));
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });
const equation = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 },
  children: [new TextRun({ text: t, italics: true, font: "Cambria Math" })],
});

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function figure(name, caption, widthIn = 6.2) {
  const file = path.join(FIG, `${name}.png`);
  const { w, h } = pngSize(file);
  const W = Math.round(widthIn * 96), H = Math.round(W * h / w);
  figNo += 1;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: W, height: H },
        altText: { title: caption, description: caption, name } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figure ${figNo}. `, bold: true, size: 20 }), ...runs(caption, { size: 20 })],
    }),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "8C8C8C" };
const borders = { top: border, bottom: border, left: border, right: border };

function table(header, rows, caption, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CONTENT_W / total;
  const w = widths.map((x) => Math.floor(x * scale));
  w[w.length - 1] += CONTENT_W - w.reduce((a, b) => a + b, 0);
  const fs_ = opts.size ?? 18;
  const cell = (txt, i, isHead, shade) => new TableCell({
    borders, width: { size: w[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    shading: isHead ? { fill: "1F4E79", type: ShadingType.CLEAR, color: "auto" }
      : shade ? { fill: shade, type: ShadingType.CLEAR, color: "auto" } : undefined,
    children: [new Paragraph({
      alignment: i === 0 && !opts.centerFirst ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(txt), { size: fs_, bold: isHead, color: isHead ? "FFFFFF" : undefined }),
    })],
  });
  if (caption) tabNo += 1;
  const out = [];
  if (caption) out.push(new Paragraph({
    keepNext: true, spacing: { before: 160, after: 80 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: `Table ${tabNo}. `, bold: true, size: 20 }), ...runs(caption, { size: 20 })],
  }));
  out.push(new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: w, layout: TableLayoutType.FIXED,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r, ri) => new TableRow({
        cantSplit: true,
        children: r.map((c, i) => cell(c, i, false, (opts.highlight && opts.highlight(r, ri)) ? "E2EFDA"
          : ri % 2 ? "F5F7FA" : undefined)),
      })),
    ],
  }));
  out.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  return out;
}

function csv(name) {
  const txt = fs.readFileSync(path.join(TAB, `${name}.csv`), "utf8").trim().split(/\r?\n/);
  const split = (l) => l.match(/("([^"]|"")*"|[^,]*)(,|$)/g).slice(0, -1).map((s) => s.replace(/,$/, "").replace(/^"|"$/g, ""));
  const head = split(txt[0]);
  return txt.slice(1).map((l) => Object.fromEntries(split(l).map((v, i) => [head[i], v])));
}
const json = (rel) => JSON.parse(fs.readFileSync(path.join(ROOT, rel), "utf8"));

const STYLES = {
  default: { document: { run: { font: BODY_FONT, size: 23 } } },
  paragraphStyles: [
    { id: "Title", name: "Title", basedOn: "Normal", run: { font: HEAD_FONT, size: 40, bold: true, color: "1F4E79" },
      paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 120 }, keepNext: true } },
    { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { font: HEAD_FONT, size: 30, bold: true, color: "1F4E79" },
      paragraph: { spacing: { before: 320, after: 140 }, keepNext: true, outlineLevel: 0 } },
    { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { font: HEAD_FONT, size: 25, bold: true, color: "2E75B6" },
      paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
    { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { font: HEAD_FONT, size: 22, bold: true, color: "404040" },
      paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 2 } },
  ],
};
const NUMBERING = {
  config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 1080, hanging: 270 } } } } ] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    { reference: "refs", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 600, hanging: 600 } } } }] },
  ],
};

module.exports = { headings, D, P, H1, H2, H3, bullet, numbered, tocLines, pageBreak, equation, figure, table, csv, json, runs,
  STYLES, NUMBERING, CONTENT_W, ROOT, BODY_FONT, HEAD_FONT };
