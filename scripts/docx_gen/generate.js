/**
 * 百度 Unlimited-OCR 深度拆解报告 — Word 生成器
 * 架构：R1 封面配方（DM-1 深青科技配色）+ 三节页码（封面无页码/目录罗马/正文阿拉伯）
 * 规范：docx skill — Profile A 正式字体（SimSun 正文/SimHei 标题/TNR 西文）、1.3x 行距、
 *       Horizontal-Only 表格、图片等比缩放、TOC 三步流程
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, PageNumber, NumberFormat,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType,
  SectionType, TableOfContents, TableLayoutType, LevelFormat,
} = require("/home/z/.npm-global/lib/node_modules/docx");

const parts = [
  require("./content_part1.js"),
  require("./content_part2.js"),
  require("./content_part3.js"),
];
const META = parts[0].meta;
const BLOCKS = parts.flatMap((p) => p.blocks);

/* ---------- DM-1 Deep Cyan palette (tech/AI report) ---------- */
const P = {
  bg: "162235",
  accent: "37DCF2",
  cover: { titleColor: "FFFFFF", subtitleColor: "B0B8C0", metaColor: "90989F", footerColor: "687078" },
  table: { headerBg: "1B6B7A", headerText: "FFFFFF", accentLine: "1B6B7A", innerLine: "C8DDE2", surface: "EDF3F5" },
  heading: "162235",
  body: "000000",
  secondary: "595959",
};

const F_BODY = { ascii: "Times New Roman", eastAsia: "SimSun", hAnsi: "Times New Roman" };
const F_HEAD = { ascii: "Times New Roman", eastAsia: "SimHei", hAnsi: "Times New Roman" };
const F_CODE = { ascii: "Courier New", eastAsia: "Microsoft YaHei", hAnsi: "Courier New" };

/* ---------- Cover: shared border constants ---------- */
const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: NB, bottom: NB, left: NB, right: NB };
const allNoBorders = { top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB };

/* ---------- Cover: width-aware title layout (CJK=1.0, ASCII=0.55) ---------- */
const CJK_RE = /[\u2E80-\u9FFF\uF900-\uFAFF\uFF00-\uFF60\u3000-\u303F]/;
function charW(ch, pt) { return CJK_RE.test(ch) ? pt * 20 : pt * 11; }
function textW(str, pt) { let w = 0; for (const ch of str) w += charW(ch, pt); return w; }

function splitTitleLines(title, maxWidthTwips, pt) {
  if (textW(title, pt) <= maxWidthTwips) return [title];
  const breakAfter = new Set([..."，。、；：！？", ..."的与和及之在于为", ..."-_—–·/", ..." \t"]);
  const lines = [];
  let remaining = title;
  while (textW(remaining, pt) > maxWidthTwips) {
    let acc = 0, cut = remaining.length;
    for (let i = 0; i < remaining.length; i++) {
      acc += charW(remaining[i], pt);
      if (acc > maxWidthTwips) { cut = i; break; }
    }
    if (cut <= 0) cut = 1;
    let breakAt = -1;
    const minCut = Math.max(1, Math.floor(cut * 0.6));
    for (let i = cut; i >= minCut; i--) {
      if (i < remaining.length && breakAfter.has(remaining[i - 1])) { breakAt = i; break; }
    }
    if (breakAt === -1) breakAt = cut;
    lines.push(remaining.slice(0, breakAt).trim());
    remaining = remaining.slice(breakAt).trim();
    if (!remaining) break;
  }
  if (remaining) lines.push(remaining);
  if (lines.length > 1 && lines[lines.length - 1].length <= 2) {
    const last = lines.pop();
    lines[lines.length - 1] += last;
  }
  return lines;
}

function calcTitleLayout(title, maxWidthTwips, preferredPt = 40, minPt = 24) {
  let titlePt = preferredPt, lines;
  while (titlePt >= minPt) {
    lines = splitTitleLines(title, maxWidthTwips, titlePt);
    const fits = lines.every((l) => textW(l, titlePt) <= maxWidthTwips);
    if (lines.length <= 3 && fits) break;
    titlePt -= 2;
  }
  if (!lines || lines.length > 3) {
    lines = splitTitleLines(title, maxWidthTwips, minPt);
    titlePt = minPt;
  }
  return { titlePt, titleLines: lines };
}

function calcCoverSpacing(params) {
  const {
    titleLineCount = 1, titlePt = 36, hasSubtitle = false,
    hasEnglishLabel = false, metaLineCount = 0,
    fixedHeight = 800, pageHeight = 16838,
    marginTop = 0, marginBottom = 0,
  } = params;
  const SAFETY = 1200;
  const usableHeight = pageHeight - marginTop - marginBottom - SAFETY;
  const titleHeight = titleLineCount * (titlePt * 23 + 200);
  const subtitleHeight = hasSubtitle ? 12 * 23 + 600 : 0;
  const englishLabelHeight = hasEnglishLabel ? 9 * 23 + 600 : 0;
  const metaHeight = metaLineCount * (10 * 23 + 100);
  const implicitParaHeight = 3 * 300;
  const contentHeight = titleHeight + subtitleHeight + englishLabelHeight + metaHeight + fixedHeight + implicitParaHeight;
  const remainingSpace = usableHeight - contentHeight;
  const safeRemaining = Math.max(remainingSpace, 400);
  const FOOTER_MIN = 800;
  const rawTop = Math.floor(safeRemaining * 0.45);
  const rawBottom = Math.floor(safeRemaining * 0.45);
  const bottomSpacing = Math.max(rawBottom, FOOTER_MIN);
  const topSpacing = Math.max(rawTop - Math.max(0, FOOTER_MIN - rawBottom), 400);
  const midSpacing = Math.max(safeRemaining - topSpacing - bottomSpacing, 0);
  return { topSpacing, midSpacing, bottomSpacing };
}

/* ---------- Recipe R1: Pure Paragraph Cover (Left-Aligned) ---------- */
function buildCoverR1(config) {
  const C = config.palette;
  const padL = 1200, padR = 800;
  const availableWidth = 11906 - padL - padR - 300;
  const { titlePt, titleLines } = calcTitleLayout(config.title, availableWidth, 40, 24);
  const titleSize = titlePt * 2;
  const spacing = calcCoverSpacing({
    titleLineCount: titleLines.length, titlePt,
    hasSubtitle: !!config.subtitle, hasEnglishLabel: !!config.englishLabel,
    metaLineCount: (config.metaLines || []).length,
    fixedHeight: 400,
  });
  const accentLeft = { style: BorderStyle.SINGLE, size: 8, color: C.accent, space: 12 };
  const children = [];

  children.push(new Paragraph({ spacing: { before: spacing.topSpacing } }));

  if (config.englishLabel) {
    children.push(new Paragraph({
      indent: { left: padL, right: padR }, spacing: { after: 500 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.accent, space: 8 } },
      children: [new TextRun({
        text: config.englishLabel.split("").join("  "),
        size: 18, color: C.accent, font: { ascii: "Arial", eastAsia: "SimHei" }, characterSpacing: 40,
      })],
    }));
  }

  for (let i = 0; i < titleLines.length; i++) {
    children.push(new Paragraph({
      indent: { left: padL },
      spacing: { after: i < titleLines.length - 1 ? 100 : 300, line: Math.ceil(titlePt * 23), lineRule: "atLeast" },
      children: [new TextRun({
        text: titleLines[i], size: titleSize, bold: true,
        color: C.titleColor, font: { eastAsia: "SimHei", ascii: "Arial" },
      })],
    }));
  }

  if (config.subtitle) {
    children.push(new Paragraph({
      indent: { left: padL }, spacing: { after: 800 },
      children: [new TextRun({
        text: config.subtitle, size: 24, color: C.subtitleColor,
        font: { eastAsia: "Microsoft YaHei", ascii: "Arial" },
      })],
    }));
  }

  for (const line of config.metaLines || []) {
    children.push(new Paragraph({
      indent: { left: padL + 200 }, spacing: { after: 80 },
      border: { left: accentLeft },
      children: [new TextRun({
        text: line, size: 24, color: C.metaColor,
        font: { eastAsia: "Microsoft YaHei", ascii: "Arial" },
      })],
    }));
  }

  children.push(new Paragraph({ spacing: { before: spacing.bottomSpacing } }));

  children.push(new Paragraph({
    indent: { left: padL, right: padR },
    border: { top: { style: BorderStyle.SINGLE, size: 2, color: C.accent, space: 8 } },
    spacing: { before: 200 },
    children: [
      new TextRun({ text: config.footerLeft || "", size: 16, color: C.footerColor, font: { ascii: "Arial" } }),
      new TextRun({ text: "                                        " }),
      new TextRun({ text: config.footerRight || "", size: 16, color: C.footerColor, font: { ascii: "Arial" } }),
    ],
  }));

  return [new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
    borders: allNoBorders,
    rows: [new TableRow({
      height: { value: 16838, rule: "exact" },
      children: [new TableCell({
        shading: { type: ShadingType.CLEAR, fill: C.bg },
        borders: noBorders,
        verticalAlign: "top",
        children,
      })],
    })],
  })];
}

/* ---------- PNG size parser (IHDR) ---------- */
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { width: b.readUInt32BE(16), height: b.readUInt32BE(20) };
}

/* ---------- Block renderers ---------- */
function segRuns(seg, base) {
  if (typeof seg === "string") return [new TextRun({ text: seg, ...base })];
  return seg.map(([text, b]) => new TextRun({ text, bold: !!b, ...base }));
}

function renderBlock(bk) {
  const out = [];
  switch (bk.t) {
    case "h1":
      out.push(new Paragraph({
        heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER,
        spacing: { before: 360, after: 160, line: 312 },
        children: [new TextRun({ text: bk.x, bold: true, size: 32, color: P.heading, font: F_HEAD })],
      }));
      break;
    case "h2":
      out.push(new Paragraph({
        heading: HeadingLevel.HEADING_2,
        spacing: { before: 240, after: 120, line: 312 },
        children: [new TextRun({ text: bk.x, bold: true, size: 30, color: P.heading, font: F_HEAD })],
      }));
      break;
    case "h3":
      out.push(new Paragraph({
        heading: HeadingLevel.HEADING_3,
        spacing: { before: 200, after: 100, line: 312 },
        children: [new TextRun({ text: bk.x, bold: true, size: 28, color: P.heading, font: F_HEAD })],
      }));
      break;
    case "p":
      out.push(new Paragraph({
        alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 480 },
        spacing: { line: 312, after: 60 },
        children: segRuns(bk.x, { size: 24, color: P.body, font: F_BODY }),
      }));
      break;
    case "code": {
      const n = bk.lines.length;
      bk.lines.forEach((ln, i) => {
        out.push(new Paragraph({
          alignment: AlignmentType.LEFT,
          shading: { type: ShadingType.CLEAR, fill: "F6F8FA" },
          spacing: { line: 312, before: i === 0 ? 120 : 0, after: i === n - 1 ? 120 : 0 },
          indent: { left: 200, right: 200 },
          children: [new TextRun({ text: ln.length ? ln : " ", size: 17, color: "24292E", font: F_CODE })],
        }));
      });
      break;
    }
    case "table": {
      out.push(new Paragraph({
        keepNext: true, alignment: AlignmentType.CENTER,
        spacing: { before: 200, after: 100, line: 312 },
        children: [new TextRun({ text: bk.cap, bold: true, size: 21, color: P.heading, font: F_HEAD })],
      }));
      const rows = [];
      rows.push(new TableRow({
        tableHeader: true, cantSplit: true,
        children: bk.head.map((h, i) => new TableCell({
          shading: { type: ShadingType.CLEAR, fill: P.table.headerBg },
          margins: { top: 60, bottom: 60, left: 120, right: 120 },
          width: { size: bk.w[i], type: WidthType.PERCENTAGE },
          children: [new Paragraph({
            spacing: { line: 312 },
            children: [new TextRun({ text: h, bold: true, size: 21, color: P.table.headerText, font: F_HEAD })],
          })],
        })),
      }));
      bk.rows.forEach((r, ri) => {
        rows.push(new TableRow({
          cantSplit: true,
          children: r.map((c, i) => new TableCell({
            shading: { type: ShadingType.CLEAR, fill: ri % 2 === 1 ? P.table.surface : "FFFFFF" },
            margins: { top: 60, bottom: 60, left: 120, right: 120 },
            width: { size: bk.w[i], type: WidthType.PERCENTAGE },
            children: [new Paragraph({
              spacing: { line: 312 },
              children: segRuns(c, { size: 21, color: P.body, font: F_BODY }),
            })],
          })),
        }));
      });
      out.push(new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        layout: TableLayoutType.FIXED,
        borders: {
          top: { style: BorderStyle.SINGLE, size: 4, color: P.table.accentLine },
          bottom: { style: BorderStyle.SINGLE, size: 4, color: P.table.accentLine },
          left: NB, right: NB,
          insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: P.table.innerLine },
          insideVertical: NB,
        },
        rows,
      }));
      if (bk.note) {
        out.push(new Paragraph({
          alignment: AlignmentType.JUSTIFIED, spacing: { before: 80, after: 120, line: 312 },
          children: [new TextRun({ text: bk.note, size: 18, color: P.secondary, font: F_BODY })],
        }));
      } else {
        out.push(new Paragraph({ spacing: { after: 120, line: 312 }, children: [] }));
      }
      break;
    }
    case "img": {
      const dims = pngSize(bk.src);
      const w = bk.w || 560;
      const h = Math.round((w * dims.height) / dims.width);
      out.push(new Paragraph({
        keepNext: true, alignment: AlignmentType.CENTER,
        spacing: { before: 200, after: 80, line: 312 },
        children: [new ImageRun({
          data: fs.readFileSync(bk.src),
          transformation: { width: w, height: h },
          type: "png",
        })],
      }));
      out.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { after: 200, line: 312 },
        children: [new TextRun({ text: bk.cap, size: 21, color: P.secondary, font: F_BODY })],
      }));
      break;
    }
    case "bullets":
      bk.items.forEach((it) => {
        out.push(new Paragraph({
          bullet: { level: 0 },
          alignment: AlignmentType.LEFT,
          spacing: { line: 312, after: 60 },
          children: segRuns(it, { size: 24, color: P.body, font: F_BODY }),
        }));
      });
      break;
    case "olist":
      bk.items.forEach((it) => {
        out.push(new Paragraph({
          numbering: { reference: "olist-1", level: 0 },
          alignment: AlignmentType.LEFT,
          spacing: { line: 312, after: 60 },
          children: segRuns(it, { size: 24, color: P.body, font: F_BODY }),
        }));
      });
      break;
    default:
      throw new Error("unknown block type: " + bk.t);
  }
  return out;
}

/* ---------- Header / Footer ---------- */
function pageHeader() {
  return new Header({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "D0D5DD", space: 4 } },
      children: [new TextRun({ text: META.headerText, size: 18, color: "808080", font: F_BODY })],
    })],
  });
}
function pageFooter() {
  return new Footer({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080", font: F_BODY })],
    })],
  });
}

/* ---------- Assemble document ---------- */
const pgSize = { width: 11906, height: 16838 };
const pgMargin = { top: 1440, bottom: 1440, left: 1701, right: 1417 };

const bodyChildren = [];
for (const bk of BLOCKS) bodyChildren.push(...renderBlock(bk));

const doc = new Document({
  features: { updateFields: true },
  styles: {
    default: {
      document: {
        run: { font: F_BODY, size: 24, color: P.body },
        paragraph: { spacing: { line: 312 } },
      },
      heading1: {
        run: { font: F_HEAD, size: 32, bold: true, color: P.heading },
        paragraph: { spacing: { before: 360, after: 160, line: 312 }, outlineLevel: 0 },
      },
      heading2: {
        run: { font: F_HEAD, size: 30, bold: true, color: P.heading },
        paragraph: { spacing: { before: 240, after: 120, line: 312 }, outlineLevel: 1 },
      },
      heading3: {
        run: { font: F_HEAD, size: 28, bold: true, color: P.heading },
        paragraph: { spacing: { before: 200, after: 100, line: 312 }, outlineLevel: 2 },
      },
    },
  },
  numbering: {
    config: [{
      reference: "olist-1",
      levels: [{
        level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    }],
  },
  sections: [
    {
      properties: { page: { size: pgSize, margin: { top: 0, bottom: 0, left: 0, right: 0 } } },
      children: buildCoverR1({
        title: META.coverTitle,
        subtitle: META.coverSubtitle,
        englishLabel: META.coverLabel,
        metaLines: META.coverMeta,
        footerLeft: META.coverFooterLeft,
        footerRight: META.coverFooterRight,
        palette: { bg: P.bg, accent: P.accent, ...P.cover },
      }),
    },
    {
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN } },
      },
      headers: { default: pageHeader() },
      footers: { default: pageFooter() },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER, spacing: { before: 480, after: 360 },
          children: [new TextRun({ text: "目  录", bold: true, size: 32, font: F_HEAD, color: P.heading })],
        }),
        new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
        new Paragraph({
          spacing: { before: 200 },
          children: [new TextRun({
            text: "注：本目录由域代码自动生成。文档编辑后如需刷新页码，请在目录上点击右键并选择\u201c更新域\u201d。",
            italics: true, size: 18, color: "888888", font: F_BODY,
          })],
        }),
      ],
    },
    {
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } },
      },
      headers: { default: pageHeader() },
      footers: { default: pageFooter() },
      children: bodyChildren,
    },
  ],
});

const OUT = process.argv[2] || "/home/z/my-project/download/百度Unlimited-OCR深度拆解报告.docx";
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("docx written:", OUT, buf.length, "bytes");
});
