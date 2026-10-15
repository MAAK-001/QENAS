"""Builds the report 'QENAS Study v1' (PDF) from the finished study v1 in QENAS/results_v1.

    python reports/build_study_v1_pdf.py . reports/QENAS_Study_v1.pdf cache/report_figures

Needs ``reportlab`` (pip install reportlab). Every number in the tables is read from results_v1/analysis/T*.md (the
output of `study.py analyse` on the v1 results, branch kaggle-study) or from the v1 experiment folders; the text
only interprets them.
"""
import glob
import json
import os
import statistics as st
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(sys.argv[1])                       # QENAS/
OUT = Path(sys.argv[2])                        # output PDF
TMP = Path(sys.argv[3])                        # scratch folder for downscaled figures
RES = ROOT / "results_v1"
AN = RES / "analysis"
FIG = AN / "figures"
TMP.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- fonts and styles
import matplotlib
TTF = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(TTF / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(TTF / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(TTF / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", str(TTF / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK = colors.HexColor("#1b2430")
ACCENT = colors.HexColor("#1f5f8b")
MUTED = colors.HexColor("#5b6675")
RULE = colors.HexColor("#c9d3de")
HEAD_BG = colors.HexColor("#e6eef5")
ZEBRA = colors.HexColor("#f6f8fa")
HILITE = colors.HexColor("#fff6dc")

S = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=30, leading=36, textColor=INK, alignment=TA_CENTER),
    "subtitle": ParagraphStyle("subtitle", fontName="DV", fontSize=13, leading=18, textColor=MUTED,
                               alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=16, leading=20, textColor=ACCENT, spaceBefore=6,
                         spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=12, leading=15, textColor=INK, spaceBefore=10,
                         spaceAfter=5),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.2, leading=13.2, textColor=INK, alignment=TA_JUSTIFY,
                           spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9.2, leading=13, textColor=INK, leftIndent=12,
                             bulletIndent=3, spaceAfter=2.5),
    "caption": ParagraphStyle("caption", fontName="DV-I", fontSize=8, leading=10.5, textColor=MUTED, spaceBefore=3,
                              spaceAfter=9),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7, leading=8.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=7, leading=8.6, textColor=INK),
    "head": ParagraphStyle("head", fontName="DV-B", fontSize=7, leading=8.6, textColor=INK),
    "toc1": ParagraphStyle("toc1", fontName="DV-B", fontSize=10, leading=15, leftIndent=0),
    "toc2": ParagraphStyle("toc2", fontName="DV", fontSize=9, leading=12.5, leftIndent=14),
    "box": ParagraphStyle("box", fontName="DV", fontSize=9, leading=13, textColor=INK),
    "mono": ParagraphStyle("mono", fontName="DV", fontSize=7.6, leading=10.2, textColor=INK),
}

PAGE_W, PAGE_H = A4
MARGIN = 17 * mm
FRAME_W = PAGE_W - 2 * MARGIN


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def H1(text):
    p = Paragraph(text, S["h1"])
    p.toc_level = 0
    return p


def H2(text):
    p = Paragraph(text, S["h2"])
    p.toc_level = 1
    return p


# ---------------------------------------------------------------- tables
def md_rows(name):
    lines = [l.strip() for l in open(AN / f"{name}.md", encoding="utf-8") if l.strip().startswith("|")]
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    return rows[0], rows[2:]


def table(header, body, widths=None, bold_rows=(), hilite_rows=(), font=7.0):
    cell = ParagraphStyle("c", parent=S["cell"], fontSize=font, leading=font * 1.23)
    cellb = ParagraphStyle("cb", parent=S["cellb"], fontSize=font, leading=font * 1.23)
    head = ParagraphStyle("h", parent=S["head"], fontSize=font, leading=font * 1.23)
    data = [[Paragraph(escape(str(h)), head) for h in header]]
    for i, r in enumerate(body):
        st_ = cellb if i in bold_rows else cell
        data.append([Paragraph(escape(str(c)), st_) for c in r])
    if widths is None:
        n = len(header)
        w = []
        for j in range(n):
            longest = max([len(str(header[j])) * 0.75] + [len(str(r[j])) for r in body])
            w.append(min(max(longest, 5), 30) + 3)
        tot = sum(w)
        widths = [FRAME_W * x / tot for x in w]
    else:
        tot = sum(widths)
        widths = [FRAME_W * x / tot for x in widths]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(data)):
        if (i - 1) in hilite_rows:
            style.append(("BACKGROUND", (0, i), (-1, i), HILITE))
        elif i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


def md_table(name, cols=None, bold=lambda r: False, hilite=lambda r: False, widths=None, font=7.0, rename=None):
    header, body = md_rows(name)
    if cols is not None:
        header = [header[c] for c in cols]
        body = [[r[c] for c in cols] for r in body]
    if rename:
        header = [rename.get(h, h) for h in header]
    return table(header, body, widths=widths, bold_rows={i for i, r in enumerate(body) if bold(r)},
                 hilite_rows={i for i, r in enumerate(body) if hilite(r)}, font=font)


def caption(text):
    return P(text, "caption")


def tblock(t, cap):
    return KeepTogether([t, caption(cap)])


# ---------------------------------------------------------------- figures
def figure(name, cap, width=1.0, max_h=None, jpeg=False):
    src = FIG / f"{name}.png"
    im = PILImage.open(src)
    w, h = im.size
    scale = min(1.0, 2600 / w)
    im = im.convert("RGB").resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    dst = TMP / (name + (".jpg" if jpeg else ".png"))
    im.save(dst, quality=90) if jpeg else im.save(dst, optimize=True)
    dw = FRAME_W * width
    dh = dw * h / w
    if max_h and dh > max_h:
        dh = max_h
        dw = dh * w / h
    return KeepTogether([Image(str(dst), width=dw, height=dh), caption(cap)])


# ---------------------------------------------------------------- data read from the experiment folders
DS = ["BUSI", "CVC", "IDRID", "KVASIR"]
DSN = {"BUSI": "BUSI", "CVC": "CVC-ClinicDB", "IDRID": "IDRiD (optic disc)", "KVASIR": "Kvasir-SEG"}
SEEDS = [42, 43, 44]
LETTER = {0: "R", 1: "D", 2: "I", 3: "C", 4: "N"}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


arch_rows = []
alpha_rows = []
front_sizes = {}
gates_relaxed = 0
qenas_alpha = []
qenas_gain = {d: [] for d in DS}
for d in DS:
    for s in SEEDS:
        e = RES / d / "qenas" / f"seed_{s}"
        fm = load(e / "final_metrics.json")
        sel = load(e / "selection.json")
        lw = load(e / "loss_weights.json")
        ch = fm["architecture"]["chromosome"]
        gates_relaxed += bool(sel["relaxed_gates"])
        crit = {r["alpha"]: r["criterion"] for r in lw["runs"]}
        qenas_alpha.append(lw["alpha"])
        qenas_gain[d].append(100 * (crit[lw["alpha"]] - crit[0.5]))
        arch_rows.append([DSN[d], s, " ".join(LETTER[g] for g in ch[:4]), " ".join(LETTER[g] for g in ch[4:]),
                          f"{fm['params_m']:.2f}", f"{fm['gflops_per_image']:.2f}", f"{lw['alpha']:.1f}",
                          "yes" if fm["test_tta"] else "no", fm["best_epoch"], fm["epochs_run"],
                          f"{100 * fm['test']['dice']:.2f}"])
        alpha_rows.append((d, s, lw["alpha"], crit))

all_lw = [load(f) for f in glob.glob(str(RES / "*" / "*" / "seed_*" / "loss_weights.json"))]
n_lw = len(all_lw)
n_half = sum(1 for x in all_lw if x["alpha"] == 0.5)
alpha_count = {a: qenas_alpha.count(a) for a in (0.2, 0.4, 0.5, 0.6, 0.8)}

for f in glob.glob(str(RES / "*" / "*" / "seed_*" / "search_summary.json")):
    p = Path(f)
    method, dset = p.parent.parent.name, p.parent.parent.parent.name
    front_sizes.setdefault((dset, method), []).append(load(f)["final_front_size"])


def fronts(method, dsets):
    v = [x for d in dsets for x in front_sizes.get((d, method), [])]
    return f"{min(v)}–{max(v)} (mean {st.mean(v):.1f})" if min(v) != max(v) else f"{v[0]} in all {len(v)} runs"


n_exp = len(glob.glob(str(RES / "*" / "*" / "seed_*" / "state.json")))
n_ext = len(glob.glob(str(RES / "KVASIR" / "*" / "seed_*" / "external_CVC" / "external_metrics.json")))

# ---------------------------------------------------------------- document
class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=18 * mm,
                         bottomMargin=16 * mm, title="QENAS Study v1",
                         subject="Evaluation study of QENAS (training-free multi-objective NAS for medical image "
                                 "segmentation)", author="QENAS project", creator="QENAS study report builder")
        frame = Frame(MARGIN, 16 * mm, FRAME_W, PAGE_H - 34 * mm, id="f", leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="cover", frames=[frame], onPage=self.cover),
                               PageTemplate(id="page", frames=[frame], onPage=self.decorate)])

    @staticmethod
    def cover(c, doc):
        c.saveState()
        c.setFillColor(ACCENT)
        c.rect(0, PAGE_H - 9 * mm, PAGE_W, 9 * mm, stroke=0, fill=1)
        c.rect(0, 0, PAGE_W, 5 * mm, stroke=0, fill=1)
        c.restoreState()

    @staticmethod
    def decorate(c, doc):
        c.saveState()
        c.setStrokeColor(RULE)
        c.setLineWidth(0.6)
        c.line(MARGIN, PAGE_H - 13 * mm, PAGE_W - MARGIN, PAGE_H - 13 * mm)
        c.line(MARGIN, 11 * mm, PAGE_W - MARGIN, 11 * mm)
        c.setFont("DV", 7.5)
        c.setFillColor(MUTED)
        c.drawString(MARGIN, PAGE_H - 11.5 * mm, "QENAS Study v1")
        c.drawRightString(PAGE_W - MARGIN, PAGE_H - 11.5 * mm, "Evaluation study — results of 78 experiments")
        c.drawString(MARGIN, 7.5 * mm, "QENAS 2.1.0 · commit 8772825 · results branch kaggle-study")
        c.drawRightString(PAGE_W - MARGIN, 7.5 * mm, f"Page {doc.page}")
        c.restoreState()

    def afterFlowable(self, f):
        if hasattr(f, "toc_level"):
            self.notify("TOCEntry", (f.toc_level, f.getPlainText(), self.page))


story = []
from reportlab.platypus import NextPageTemplate

# ---------------- cover
story += [Spacer(1, 38 * mm), P("QENAS Study v1", "title"), Spacer(1, 6 * mm),
          P("Evaluation study of QENAS — training-free multi-objective neural architecture search "
            "for medical image segmentation", "subtitle"), Spacer(1, 16 * mm)]
facts = [
    ["Method tested", "QENAS 2.1.0 (NSGA-II on parameters, SynFlow and ZiCo; learning-capability selection)"],
    ["Code", "github.com/MAAK-001/QENAS, branch main, commit 8772825 (tag study-v1)"],
    ["Results", "GitHub branch kaggle-study; local copy QENAS/results_v1/"],
    ["Experiments", f"{n_exp} finished experiments + {n_ext} external evaluations (all planned jobs complete)"],
    ["Datasets", "BUSI, CVC-ClinicDB, IDRiD (optic disc), Kvasir-SEG"],
    ["Conditions", "QENAS, U-Net, Attention U-Net, 4 ablations, random search, CVC frame-split study"],
    ["Seeds", "42, 43, 44 (the data split is identical for all seeds and conditions)"],
    ["Hardware", "Kaggle, 2 × NVIDIA Tesla T4 (one job per GPU)"],
    ["Report date", "15 October 2026"],
]
ft = Table([[P(f"<b>{a}</b>", "box"), P(escape(b), "box")] for a, b in facts], colWidths=[34 * mm, FRAME_W - 34 * mm])
ft.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("LINEABOVE", (0, 0), (-1, 0), 1, ACCENT), ("LINEBELOW", (0, -1), (-1, -1), 1, ACCENT)]))
story += [ft, NextPageTemplate("page"), PageBreak()]

# ---------------- contents
toc = TableOfContents()
toc.levelStyles = [S["toc1"], S["toc2"]]
story += [P("Contents", "h1"), toc, PageBreak()]

# ---------------- 1 executive summary
story.append(H1("1  Executive summary"))
story.append(P(
    "This report documents the first complete evaluation study of QENAS (v1). QENAS searches U-shaped "
    "segmentation networks without training: NSGA-II minimises the parameter count and maximises two zero-cost "
    "proxies (SynFlow and ZiCo). Afterwards, a small part of the final Pareto front is trained briefly, one "
    "architecture is chosen by its learning behaviour, and that architecture is trained normally. The study ran "
    f"{n_exp} experiments on four public datasets with three seeds each. It compares QENAS with U-Net and Attention "
    "U-Net trained under the same protocol on the same splits, and it adds ablations, a random-search control, "
    "external validation and a data-leakage study. All tests are paired on the same test images and "
    "Holm-corrected."))
story.append(H2("Main findings"))
story += bullets([
    "<b>Accuracy.</b> QENAS is significantly better than both baselines on BUSI (+6.84 / +5.40 pp Dice vs U-Net / "
    "Attention U-Net), CVC-ClinicDB (+21.48 / +12.34 pp) and IDRiD (+5.08 / +2.65 pp); all Holm-corrected "
    "p &lt; 0.001. On Kvasir-SEG it is <b>not</b> better: +0.90 pp vs U-Net (p = 0.84) and −1.60 pp vs Attention "
    "U-Net (Holm p = 0.0076).",
    "<b>Size and cost.</b> The QENAS networks have 1.63–2.22 M parameters on average: 14–19× fewer than U-Net "
    "(31.04 M) and 3.9–5.4× fewer than Attention U-Net (8.73 M). They need 3.4–13.6 GFLOPs per image, "
    "against 54.7–136.8 for U-Net.",
    "<b>Generalisation.</b> When trained on Kvasir-SEG and tested on all 612 CVC-ClinicDB frames, QENAS reaches "
    "75.92 % Dice, against 68.52 % for U-Net and 70.94 % for Attention U-Net. Both differences are significant.",
    "<b>Search.</b> NSGA-II beats random search with the same number of evaluations in hypervolume in 12 of 12 "
    "runs.",
    "<b>Speed.</b> One complete QENAS run, from data to the test result, takes 33–106 minutes on a single Tesla "
    "T4. The training-free search alone takes 5–10 minutes. Mixed-GGNAS reports a 26 h search on IDRiD.",
    "<b>Weak components.</b> None of the three objectives is significantly correlated with the 20-epoch "
    "validation Dice of the trained candidates. The ablations do not show that ZiCo, SynFlow or the "
    "learning-capability selection help. On Kvasir-SEG, removing SynFlow (+3.10 pp), removing ZiCo (+1.79 pp) and "
    "the zero-shot max-ZiCo pick (+2.75 pp) are all significantly <b>better</b> than full QENAS.",
    "<b>Selection.</b> The ΔDice rule chose the candidate with the best 20-epoch validation Dice in only 3 of 12 "
    "runs. With three objectives, the final population was entirely non-dominated (20 of 20) in all 12 runs.",
    "<b>Leakage.</b> A random frame split inflates the CVC-ClinicDB test Dice by +20.29 pp (QENAS) and +28.41 pp "
    "(U-Net) compared with the leakage-free sequence split.",
])
story.append(H2("Verdict"))
story.append(P(
    "QENAS v1 is a fast and very compact method. It generalises well and clearly beats the hand-designed "
    "baselines on three of four datasets. However, the evidence does not yet show that its individual components "
    "(the third objective, SynFlow, the ΔDice selection, the loss-weight grid and TTA) are needed. These findings "
    "lead directly to the changes for QENAS v2 (Section 6): remove SynFlow, remove TTA, select by the "
    "end-of-run validation Dice, train for 120 epochs, use successive halving for the candidates, add nnU-Net "
    "as a baseline, and test whether optimising the loss coefficients helps at all."))
story.append(PageBreak())

# ---------------- 2 method
story.append(H1("2  The method tested: QENAS v1"))
story.append(P(
    "QENAS (Quick Evolutionary NAS) searches a U-shaped network with 8 genes (4 encoder and 4 decoder positions). "
    "Each gene is one of 5 blocks, so there are 5<super>8</super> = 390,625 architectures. The blocks are "
    "Residual, Dense, Inception, ConvNeXt and a DARTS block (a NAS-Unet cell with the genotype that Mixed-GGNAS "
    "published for the dataset). The four manual blocks and their transitions are the official Mixed-GGNAS "
    "implementations. The base width is 32 and the convolution scale is fixed. The pipeline is a single fixed "
    "sequence with no options:"))
steps = [
    ["1", "Split", "Fixed, leakage-free train / validation / test split (data.split_seed = 42) for every seed."],
    ["2", "Search (training-free)", "pymoo NSGA-II, population 20, generation 0 + 10 generations; UniformCrossover "
     "0.9, ChoiceRandomMutation 1/8, duplicate elimination. Objectives (all minimised): f1 = parameters, "
     "f2 = 1/(1+ln(1+SynFlow)), f3 = 1/(1+ZiCo). ZiCo uses the official SLDGroup code on 2 augmented training "
     "batches with the segmentation loss."],
    ["3", "Pareto candidates", "Final non-dominated front, reduced to k = 8 points: the extreme point of each "
     "objective, plus greedy farthest-point samples."],
    ["4", "Candidate training", "Each candidate is trained for 20 epochs: AdamW 1e-3, constant LR, α = 0.5, the "
     "same seed and the same data order."],
    ["5", "Learning-capability selection", "ΔDice = val Dice(20) − val Dice(1), with the gates G1 finite losses, "
     "G2 decreasing val loss, G3 no overfitting, G4 significant upward Dice trend and G5 better than the "
     "all-foreground predictor. Ties within the noise are broken by final Dice, then ΔLoss, then size."],
    ["6", "Loss weights", "L = α·BCE + (1−α)·soft-mIoU with α ∈ {0.2, 0.4, 0.5, 0.6, 0.8}. Each value gets "
     "20 epochs; the criterion is the mean val Dice of the last 5 epochs. The α = 0.5 run reuses the candidate "
     "run."],
    ["7", "Final training", "AdamW 1e-3, wd 5e-5, 1 warm-up epoch then poly decay, 100 epochs, best-validation "
     "checkpoint. Early stopping only on overfitting."],
    ["8", "TTA decision", "Horizontal-flip TTA is used only if it increases the validation Dice."],
    ["9", "Test", "A single evaluation on the untouched test split: Dice, mIoU, IoU, HD95, Sensitivity, Params, "
     "FLOPs and latency."],
]
story.append(tblock(table(["#", "Stage", "What it does (v1 settings)"], steps, widths=[3, 16, 81], font=7.6),
                    "Table 2.1. The QENAS v1 pipeline (METHODOLOGY.md; DESIGN_DECISIONS D1–D45)."))
story.append(P(
    "<b>Metrics.</b> Dice (= DSC), mIoU (mean of the foreground and background IoU, as in Mixed-GGNAS), "
    "foreground IoU, HD95 in pixels (the maximum of the two directed 95th percentiles; an empty prediction counts "
    "as the image diagonal), sensitivity, parameters, FLOPs and inference time per image. FLOPs follow the thop "
    "convention for one training batch, which reproduces the Mixed-GGNAS U-Net values of 437.94 G and 547.42 G "
    "exactly. GFLOPs per image are reported as well. Inference time is the median of 50 forward passes at "
    "batch 1 on the run's GPU."))

# ---------------- 3 study design
story.append(H1("3  Study design"))
story.append(H2("3.1  Research questions"))
rq = [["RQ1", "How does QENAS compare with hand-designed networks trained on the same splits?", "T1, T2"],
      ["RQ2", "What does each QENAS component contribute (ZiCo, SynFlow, learning-capability selection)?", "T3, T4"],
      ["RQ3", "Does NSGA-II search better than random sampling with the same budget?", "T5"],
      ["RQ4", "Do the models generalise to another hospital and device (Kvasir-SEG → CVC-ClinicDB)?", "T14, T15"],
      ["RQ5", "How much does a random frame split inflate CVC-ClinicDB scores?", "T16"],
      ["—", "Evidence for the design decisions: objective relations, proxies, selection, TTA, training, time",
       "T6–T13"]]
story.append(tblock(table(["RQ", "Question", "Tables"], rq, widths=[6, 80, 14], font=7.6),
                    "Table 3.1. Research questions of the v1 study (STUDY.md)."))
story.append(H2("3.2  Conditions"))
cond = [["qenas", "the method (Section 2)", "all four"],
        ["unet", "U-Net, 31.04 M parameters (the Mixed-GGNAS baseline variant), no search", "all four"],
        ["attention_unet", "Attention U-Net, 8.73 M parameters (the Mixed-GGNAS baseline variant), no search",
         "all four"],
        ["qenas_no_zico", "NSGA-II on (params, SynFlow) only", "BUSI, Kvasir-SEG"],
        ["qenas_no_synflow", "NSGA-II on (params, ZiCo) only", "BUSI, Kvasir-SEG"],
        ["qenas_random_pick", "final architecture = a random Pareto candidate (reuses the QENAS search)",
         "BUSI, Kvasir-SEG"],
        ["qenas_zico_pick", "final architecture = the candidate with the highest ZiCo, chosen without training "
         "(the zero-shot rule)", "BUSI, Kvasir-SEG"],
        ["random_search", "uniform random sampling instead of NSGA-II, same number of evaluations (220), no "
         "training", "all four"],
        ["CVC frame split", "data.split_unit = image: frames instead of colonoscopy sequences are split",
         "CVC (QENAS, U-Net)"]]
story.append(tblock(table(["Condition", "Difference from QENAS", "Datasets"], cond, widths=[18, 62, 20], font=7.4),
                    "Table 3.2. Conditions (qenas/methods.py). The baselines get the full QENAS training protocol: "
                    "the α grid on validation, 100-epoch final training, the best-validation checkpoint, the TTA "
                    "decision on validation and a single test."))
story.append(H2("3.3  Datasets and splits"))
dsr = [["BUSI (benign + malignant)", "645", "413 / 103 / 129, stratified by class",
        "image (exact duplicates handled)"],
       ["CVC-ClinicDB", "612", "395 / 96 / 121", "colonoscopy sequence (29 sequences)"],
       ["IDRiD, optic disc", "81", "43 / 11 / 27 (official test split)", "image"],
       ["Kvasir-SEG", "1000", "704 / 176 / 120 (official val.txt = test)", "image"]]
story.append(tblock(table(["Dataset", "Images", "Train / validation / test", "Split unit"], dsr,
                          widths=[24, 9, 37, 30], font=7.6),
                    "Table 3.3. The split is identical for every condition and seed (data.split_seed = 42). The "
                    "run seed changes only the search, the initialisation, the data order and the augmentation."))
story.append(H2("3.4  Statistics and execution"))
story += bullets([
    "Results are given as mean ± SD over the three seeds (42, 43, 44).",
    "Paired comparisons use the same test images. The per-image Dice is averaged over the seeds, then tested "
    "with a two-sided Wilcoxon signed-rank test. The mean difference has a 95 % bootstrap CI (10,000 resamples) "
    "and Holm–Bonferroni correction is applied within each table and dataset.",
    "Search quality is measured by the hypervolume of the non-dominated set of all evaluated architectures "
    "(union-normalised, reference point 1.1), as the best-so-far value along the archive.",
    "Proxy validity is measured by the Spearman ρ between each objective and the candidates' 20-epoch "
    "validation Dice.",
    "Execution: Kaggle with 2 × Tesla T4. A deadline-aware scheduler runs one job per GPU, resumes from "
    "checkpoints and pushes the results to the GitHub branch kaggle-study every 30 minutes. All "
    f"{n_exp} experiments and {n_ext} external evaluations finished.",
])
story.append(PageBreak())

# ---------------- 4 results
story.append(H1("4  Results"))
story.append(H2("4.1  RQ1 — QENAS vs hand-designed networks"))
is_q = lambda r: r[1] == "QENAS"
is_pub = lambda r: "published" in r[1]
story.append(tblock(md_table("T1_main_results", cols=[0, 1, 3, 4, 5, 6, 7], bold=is_q, hilite=is_pub,
                             widths=[70, 129, 60, 60, 60, 60, 60], font=6.8),
                    "Table T1a. Test accuracy on identical splits (mean ± SD over 3 seeds). QENAS rows are bold; "
                    "the published Mixed-GGNAS results (highlighted) used their own splits, and for CVC-ClinicDB "
                    "a random frame split, so they are not directly comparable."))
story.append(tblock(md_table("T1_main_results", cols=[0, 1, 8, 9, 10, 11], bold=is_q, hilite=is_pub,
                             widths=[70, 189, 60, 60, 60, 60], font=6.8),
                    "Table T1b. Model cost. FLOPs per training batch (8 images; IDRiD 4) follow the Mixed-GGNAS "
                    "convention; ms/image is the median of 50 passes at batch 1 on a Tesla T4."))
story.append(tblock(md_table("T2_significance_main", widths=[52, 92, 28, 34, 34, 32, 70, 46, 46, 42], font=6.6),
                    "Table T2. Paired comparison on the same test images (per-image Dice averaged over seeds, "
                    "two-sided Wilcoxon, Holm-corrected)."))
story.append(figure("per_image_dice_boxplots", "Figure 4.1. Per-image test Dice (mean over seeds). Triangles are "
                    "means, lines are medians."))
story.append(figure("accuracy_vs_parameters", "Figure 4.2. Test Dice against parameter count; QENAS reaches higher "
                    "Dice with an order of magnitude fewer parameters on BUSI, CVC-ClinicDB and IDRiD."))
story.append(P(
    "<b>Reading.</b> QENAS wins clearly on BUSI, CVC-ClinicDB and IDRiD. It wins on Dice and also on boundary "
    "quality: on CVC-ClinicDB its HD95 is 56.1 px, against 103.1 px for U-Net and 91.0 px for Attention U-Net. "
    "It does this with 1.6–2.2 M parameters. On Kvasir-SEG all three methods are close, and Attention U-Net is "
    "significantly better than QENAS; QENAS also has its largest seed variance there (SD 2.38 pp). Compared with "
    "the published Mixed-GGNAS numbers on their own splits, QENAS is 0.77 pp lower on BUSI with 4.3× fewer "
    "parameters, and 0.72 pp lower on IDRiD with 15.9× fewer parameters. These are not paired comparisons."))

story.append(H2("4.2  Architectures selected by QENAS"))
story.append(tblock(table(["Dataset", "Seed", "Encoder (g1–g4)", "Decoder (g5–g8)", "Params (M)", "GFLOPs / img",
                           "α", "TTA", "Best epoch", "Epochs run", "Test Dice (%)"], arch_rows,
                          widths=[17, 6, 13, 13, 9, 10, 5, 5, 8, 8, 10], font=7.0),
                    "Table 4.1. Final architectures (R = Residual, D = Dense, I = Inception, C = ConvNeXt, N = DARTS / "
                    "NAS-Unet cell). Read from final_metrics.json of each run."))
story.append(P(
    "The search does not converge to a single architecture: each dataset and seed gives a different network, "
    "and every block type appears. This matches the low agreement between the proxies and the trained "
    f"quality (Section 4.6). The learning-capability gates never had to be relaxed ({gates_relaxed} of 12 runs)."))

story.append(H2("4.3  RQ2 — Ablations"))
story.append(tblock(md_table("T3_ablations", cols=[0, 1, 3, 4, 5, 6, 8], bold=is_q,
                             widths=[60, 139, 60, 60, 60, 60, 60], font=6.8),
                    "Table T3. Ablations on BUSI and Kvasir-SEG (test, mean ± SD over 3 seeds)."))
story.append(tblock(md_table("T4_significance_ablations", widths=[48, 110, 28, 34, 34, 32, 70, 46, 46, 42], font=6.4),
                    "Table T4. QENAS vs each ablation on the same test images (Holm-corrected)."))
story.append(figure("ablations", "Figure 4.3. Test Dice of QENAS and its ablations.", width=0.82))
story.append(P(
    "<b>Reading.</b> On BUSI, no ablation is significantly worse than QENAS except the zero-shot max-ZiCo pick "
    "(−0.93 pp, Holm p = 0.032). On Kvasir-SEG three ablations are significantly <b>better</b> than QENAS: without "
    "SynFlow +3.10 pp (Holm p = 2.3e-4), without ZiCo +1.79 pp (p = 0.034) and the max-ZiCo pick +2.75 pp "
    "(p = 0.013). So the v1 data do not support the claim that all three objectives and the ΔDice selection are "
    "needed. The two-objective search without SynFlow is never significantly worse than QENAS and is clearly better "
    "on Kvasir-SEG."))
story.append(P(
    "<b>Known defect.</b> On Kvasir-SEG, qenas_random_pick picked exactly the same candidate as QENAS in all three "
    "seeds, so its result equals QENAS (Δ = 0). The random index came from derive_seed(seed, "
    "\"random-pareto-candidate\"), which does not depend on the dataset, so all datasets of a seed draw the same "
    "index. On Kvasir-SEG that index coincided with the QENAS choice. The random-pick result on Kvasir-SEG is "
    "therefore not informative. The seed must include the dataset (fixed in v2)."))

story.append(H2("4.4  RQ3 — NSGA-II vs random search"))
story.append(tblock(md_table("T5_search_vs_random", font=7.4),
                    "Table T5. Hypervolume of the non-dominated archive with the same number of training-free "
                    "evaluations."))
story.append(figure("search_vs_random", "Figure 4.4. Best-so-far hypervolume over evaluations, NSGA-II vs random "
                    "search."))
story.append(P(
    "NSGA-II finds a better proxy front than random sampling in every run. The size of the final non-dominated "
    f"set shows how little selection pressure is left with three objectives: QENAS ended with {fronts('qenas', DS)} "
    "non-dominated individuals out of 20. The two-objective searches ended with fewer: "
    f"params + ZiCo {fronts('qenas_no_synflow', ['BUSI', 'KVASIR'])}, params + SynFlow "
    f"{fronts('qenas_no_zico', ['BUSI', 'KVASIR'])}. Note that RQ3 shows a better <i>proxy</i> front; it does not "
    "show a better segmentation result, because the proxies hardly predict the trained Dice (Section 4.6)."))

story.append(H2("4.5  Objective relations"))
story.append(tblock(md_table("T6_objective_agreement", font=7.2),
                    "Table T6. Kendall τ between the minimised objectives on random architectures (> 0: they improve "
                    "together; < 0: trade-off)."))
story.append(tblock(md_table("T7_block_effects", font=7.0),
                    "Table T7. Effect of one gene of each block type relative to Residual (least squares on random "
                    "architectures)."))
story.append(tblock(md_table("T8_nondominated_share", font=7.2),
                    "Table T8. Share of non-dominated individuals in random populations, 2 (params, SynFlow) vs 3 "
                    "objectives."))
story.append(P(
    "SynFlow and ZiCo agree positively (τ = +0.17 to +0.39, still +0.13 to +0.41 after removing model size). "
    "This means SynFlow adds little information that ZiCo does not already carry. SynFlow mainly trades off "
    "against the parameter count (τ = −0.41 to −0.51), i.e. it largely rewards bigger networks. ZiCo is almost "
    "independent of size (τ = −0.09 to −0.18). A third objective raises the non-dominated share of a "
    "20-individual population from 29–34 % to 40–51 %, which weakens the rank-based selection of NSGA-II."))

story.append(H2("4.6  Do the zero-cost proxies predict trained quality?"))
story.append(tblock(md_table("T9_proxies_vs_quality", widths=[58, 30, 24, 55, 55, 55, 55, 55, 55], font=6.6),
                    "Table T9. Spearman ρ between each objective and the 20-epoch validation Dice of the 24 trained "
                    "candidates per dataset (8 per seed)."))
story.append(figure("proxies_vs_val_dice", "Figure 4.5. Zero-cost proxies against the 20-epoch validation Dice of "
                    "the Pareto candidates.", max_h=105 * mm))
story.append(P(
    "No correlation is significant. ZiCo on BUSI is the strongest (ρ = +0.37, p = 0.078). SynFlow is close to zero "
    "on BUSI and negative on CVC-ClinicDB and IDRiD. Inside a Pareto front the candidates are already good by the "
    "proxies, so the remaining differences in trained quality are not explained by them. This is why a supervised "
    "selection step after the search is needed, and why that step has to be reliable."))

story.append(H2("4.7  Learning-capability selection"))
story.append(tblock(md_table("T10_selection_quality", font=7.4),
                    "Table T10. Rank of the candidate chosen by ΔDice among the 8 trained candidates, by 20-epoch "
                    "validation Dice (1 = best)."))
story.append(P(
    "ΔDice chose the candidate with the best 20-epoch validation Dice in only 3 of 12 runs; the mean rank is 3.3 "
    "of 8. ΔDice rewards a low starting point (Dice after epoch 1) as much as a high end point, so a candidate that "
    "starts badly can win even though it ends below the others. The end-of-run validation Dice, averaged over the "
    "last epochs to reduce noise, measures the quantity that matters directly. That change is made in v2."))

story.append(H2("4.8  Loss-coefficient (α) selection"))
qa = [[DSN[d], ", ".join(f"{a:.1f}" for (dd, s, a, c) in alpha_rows if dd == d),
       ", ".join(f"{g:+.2f}" for g in qenas_gain[d]), f"{st.mean(qenas_gain[d]):+.2f}"] for d in DS]
allg = [g for d in DS for g in qenas_gain[d]]
story.append(tblock(table(["Dataset", "α chosen (seeds 42, 43, 44)", "Val-criterion gain over α = 0.5 (pp)", "Mean (pp)"],
                          qa, widths=[22, 26, 36, 12], font=7.4),
                    "Table 4.2. The α grid in the QENAS runs. Gain = mean validation Dice of the last 5 of 20 epochs, "
                    "chosen α minus α = 0.5."))
story.append(P(
    f"In the 12 QENAS runs α = 0.5 was never chosen: 0.2 was chosen {alpha_count[0.2]}×, 0.4 {alpha_count[0.4]}×, "
    f"0.6 {alpha_count[0.6]}× and 0.8 {alpha_count[0.8]}×. Over all {n_lw} runs of the study (all conditions), "
    f"α = 0.5 was chosen {n_half}×. The validation gain of the chosen α is on average {st.mean(allg):+.2f} pp "
    f"(median {st.median(allg):+.2f} pp). However, this gain is measured on the same validation runs that choose "
    "α, which is an optimistic estimate (the best of five noisy values). The grid costs 5–24 minutes per QENAS "
    "run (T13), 17–24 % of the total time. <b>v1 cannot answer whether the grid improves the final test "
    "result</b>, because no run trained the final model with a fixed α. This is the new loss-coefficient ablation "
    "of v2."))

story.append(H2("4.9  Test-time augmentation"))
story.append(tblock(md_table("T11_tta", font=7.0),
                    "Table T11. Flip-TTA decided on validation."))
story.append(P(
    "TTA was chosen in 6 of 12 QENAS runs (43 of 66 runs over all conditions). The mean validation difference "
    "with and without TTA is within ±0.5 pp for QENAS (BUSI −0.24, CVC-ClinicDB +0.32, IDRiD −0.02, Kvasir-SEG "
    "+0.43 pp) and within ±0.65 pp for every condition. TTA doubles "
    "the inference cost and adds a decision step without a measurable benefit. It is removed in v2."))

story.append(H2("4.10  Final training behaviour"))
story.append(tblock(md_table("T12_training", widths=[62, 118, 24, 52, 52, 45, 60, 60], font=6.8),
                    "Table T12. Best epoch, epochs run, early stopping on overfitting, and the train/val Dice of the "
                    "best checkpoint."))
story.append(P(
    "The QENAS networks reach their best validation Dice at epochs 65–75 on average. Several other runs peak at "
    "the very end of the 100 epochs: on Kvasir-SEG, U-Net (93.0), Attention U-Net (91.7), QENAS without SynFlow "
    "(98.3) and QENAS without ZiCo (88.7), and on IDRiD, Attention U-Net (98.0). These runs were still improving, "
    "so 100 epochs is too short to compare fairly. v2 trains for 120 epochs and keeps early stopping on "
    "overfitting."))

story.append(H2("4.11  Compute time"))
story.append(tblock(md_table("T13_time", cols=[0, 1, 3, 4, 5, 6, 7], bold=is_q, widths=[62, 130, 50, 55, 55, 55, 60],
                             font=6.8),
                    "Table T13. Time per experiment in minutes on one Tesla T4 (start-to-end timer of each run)."))
t8 = [["NASUnet", "3 h", "supervised (DARTS)"], ["HCT-net", "3 h", "supervised"], ["UNAS-Net", "19 h", "supervised"],
      ["Genetic U-Net", "4 h", "supervised (GA)"], ["MedUNAS", "11 h", "supervised"], ["BiX-NAS", "4 h", "supervised"],
      ["Mixed-GGNAS (with ViT / without)", "26 h / 15 h", "supervised GA + DARTS (RTX 4070 Ti Super)"],
      ["QENAS — search only", "10.2 min", "training-free (Tesla T4)"],
      ["QENAS — search + candidate training", "19.6 min", "architecture fixed (Tesla T4)"],
      ["QENAS — complete run incl. α grid, final training, test", "32.7 min", "everything (Tesla T4)"]]
story.append(tblock(table(["Method", "Search time on IDRiD", "Notes"], t8, widths=[42, 18, 40], font=7.4,
                          bold_rows={7, 8, 9}),
                    "Table 4.3. Search cost on IDRiD. The other methods' times are as reported in Mixed-GGNAS (ESWA "
                    "2025) Table 8; hardware differs, and the Tesla T4 is slower than the RTX 4070 Ti Super."))
story.append(P(
    "The QENAS search alone is 18–153× faster than the supervised NAS searches in Table 4.3 (3–26 h). Even the "
    "complete QENAS run, which also includes candidate training, the α grid, final training and the test, is "
    "5.5–48× faster. For reference, the training-free medical NAS methods AEG-cTFNAS and "
    "MedNAS report 370 s and about 3,184 s of search time on ACDC (RTX 4090, AEG-cTFNAS paper). These are "
    "different datasets, so the numbers only show the order of magnitude. In a QENAS run most of the time is "
    "spent on supervised training (candidate training 29–46 %, α grid 17–24 %), not on the search."))

story.append(H2("4.12  RQ4 — External validation (Kvasir-SEG → CVC-ClinicDB)"))
story.append(tblock(md_table("T14_external_validation", widths=[95, 55, 24, 58, 58, 58, 58, 58], font=6.8),
                    "Table T14. Kvasir-SEG best-validation checkpoints applied unchanged to all 612 CVC-ClinicDB "
                    "frames."))
story.append(tblock(md_table("T15_significance_external", widths=[62, 92, 28, 34, 34, 32, 70, 46, 46, 42], font=6.4),
                    "Table T15. Paired comparison on the 612 external frames."))
story.append(P(
    "Although QENAS is not better than the baselines inside Kvasir-SEG, it transfers better to a different "
    "hospital and device: +7.41 pp Dice vs U-Net and +4.99 pp vs Attention U-Net, both significant. Against "
    "Attention U-Net, however, QENAS is better on fewer frames than it is worse on (274 / 332), so the mean gain "
    "comes from fewer large failures, not from a uniform improvement. The seed variance is large (SD 5.66 pp)."))

story.append(H2("4.13  RQ5 — CVC-ClinicDB data leakage"))
story.append(tblock(md_table("T16_cvc_leakage", widths=[50, 60, 70, 70, 50, 80], font=7.0),
                    "Table T16. Sequence-level (leakage-free) vs random frame split on CVC-ClinicDB."))
story.append(P(
    "Consecutive frames of the same colonoscopy sequence are almost identical. A random frame split therefore "
    "puts near-copies of the test frames into training, and the test Dice rises by 20–28 pp. The published "
    "CVC-ClinicDB scores obtained with frame splits, including Mixed-GGNAS's 93.51 %, measure mostly this "
    "leakage. QENAS reaches 90.05 % under a random frame split. The sequence split is the honest protocol and "
    "stays the default."))

story.append(H2("4.14  Qualitative results"))
for d, name in [("BUSI", "BUSI"), ("CVC", "CVC-ClinicDB"), ("IDRID", "IDRiD (optic disc)"),
                ("KVASIR", "Kvasir-SEG")]:
    story.append(figure(f"qualitative_{d}", f"Figure 4.{6 + DS.index(d)}. {name}: test images at the 10th, 40th, "
                        f"70th and 95th percentile of the QENAS Dice (seed 42), with U-Net and Attention U-Net.",
                        max_h=112 * mm, jpeg=True))

# ---------------- 5 discussion
story.append(H1("5  Discussion"))
story.append(H2("5.1  What works"))
story += bullets([
    "Very small, accurate networks. QENAS has 14–19× fewer parameters than U-Net and still wins by 2.7–21.5 pp Dice "
    "on three datasets, with better HD95 and sensitivity.",
    "Training-free search in minutes. The search takes 5–10 min and a complete run 33–106 min on a T4, against "
    "15–26 h for Mixed-GGNAS.",
    "Better transfer to an unseen hospital and device (RQ4).",
    "A rigorous protocol: leakage-free splits, a single test, paired statistics, three seeds, external "
    "validation, and a measured leakage effect (RQ5).",
])
story.append(H2("5.2  What does not work (yet)"))
story += bullets([
    "The proxies do not predict trained quality within the front (T9), so the search finds a better proxy front "
    "(T5) but not necessarily better networks.",
    "SynFlow is redundant with ZiCo and biased towards larger networks (T6). Removing it was never significantly worse "
    "and was significantly better on Kvasir-SEG (T4).",
    "The third objective makes the whole final population non-dominated (20/20), so NSGA-II ends with weak "
    "selection pressure (T8, Section 4.4).",
    "ΔDice selection is unreliable: it chose the best-ending candidate in only 3 of 12 runs (T10).",
    "TTA has no measurable benefit (T11). The loss-coefficient grid is untested against a fixed α (Section 4.8).",
    "100 epochs under-trains several Kvasir-SEG and IDRiD runs (T12).",
    "On Kvasir-SEG QENAS does not beat the baselines (T2).",
])
story.append(H2("5.3  Threats to validity"))
story += bullets([
    "Three seeds on one fixed split. The seed variance of QENAS on Kvasir-SEG (2.38 pp) and in external "
    "validation (5.66 pp) is large.",
    "The baselines are U-Net and Attention U-Net in the Mixed-GGNAS variants. The strongest standard "
    "medical-segmentation baseline, nnU-Net, is missing.",
    "Mixed-GGNAS could not be re-run on these splits (no released architectures or weights; ≈ 26 h search), so "
    "its published numbers are only reference values.",
    "The ablation datasets (BUSI and Kvasir-SEG) were also used for development. The v2 changes are motivated "
    "by these results, so v2 must be validated with new seeds.",
    "The qenas_random_pick result on Kvasir-SEG is invalid because of the seed defect (Section 4.3).",
])

# ---------------- 6 conclusions and v2
story.append(H1("6  Conclusions and consequences for QENAS v2"))
story.append(P(
    "QENAS v1 is fast, compact and accurate on three of four datasets, and it transfers well. The study also shows "
    "that several components are unsupported or harmful. Each change for v2 follows from a measured v1 result:"))
v2 = [["1", "Remove SynFlow; search on params + ZiCo", "T4 (Kvasir +3.10 pp without SynFlow), T6 (redundant with "
       "ZiCo, size-biased), T8 and the final front of 20/20"],
      ["2", "Remove flip-TTA", "T11: QENAS differences within ±0.5 pp (all conditions ±0.65 pp); TTA doubles "
       "the inference cost"],
      ["3", "Select by the mean validation Dice of the last 5 epochs (gates G1–G5 kept)", "T10: ΔDice picked the "
       "best-ending candidate in 3/12 runs"],
      ["4", "Final training 120 epochs (early stopping on overfitting unchanged)", "T12: runs peaking at epochs "
       "88–98 of 100"],
      ["5", "Successive halving: 16 candidates × 5 epochs → best 8 get 5 more → best 4 trained to 20 epochs",
       "T9/T10: the selection needs more candidates and more reliable evidence. The budget is the same 160 "
       "candidate-epochs"],
      ["6", "Add nnU-Net as a baseline (study only)", "Threat to validity 5.3: the standard strong baseline is "
       "missing"],
      ["—", "Random-pick seed includes the dataset", "Defect of Section 4.3"],
      ["—", "New ablation: α grid vs fixed α = 0.5", "Section 4.8: never tested in v1; the grid costs 17–24 % "
       "of a run"]]
story.append(tblock(table(["#", "Change in v2", "v1 evidence"], v2, widths=[4, 46, 50], font=7.4),
                    "Table 6.1. From v1 findings to the v2 design."))
story.append(P(
    "v2 does not repeat what v1 already answered. These results stay valid and are reported from v1: the leakage "
    "study (RQ5), the SynFlow and ZiCo objective analyses (T6–T8), and the proxy validity (T9). v2 tests only "
    "what changed or was never tested, with new seeds so that the changes are not judged on the runs that "
    "motivated them."))

# ---------------- appendix
story.append(H1("Appendix A  Reproduction"))
story.append(P("All numbers in this report are read from the study outputs. To regenerate the tables and "
               "figures from the results folder:"))
cmds = [["Code of study v1", "git checkout study-v1"],
        ["Pull the results", "git fetch origin kaggle-study, then: mkdir results_v1 && git archive origin/kaggle-study results | tar -x -C results_v1 --strip-components=1"],
        ["Tables T1–T16 (v1 code)", "python study.py analyse --results results_v1 --data-root ../Datasets"],
        ["Per-experiment figures", "python study.py figures --results results_v1"],
        ["This report", "python reports/build_study_v1_pdf.py . reports/QENAS_Study_v1.pdf cache/report_figures"]]
story.append(tblock(table(["Purpose", "Command (run in QENAS/)"], cmds, widths=[24, 76], font=7.4),
                    "Table A.1. Commands. The v1 results are analysed with the v1 code (tag study-v1)."))
locs = [["results_v1/<DATASET>/<method>/seed_<s>/", "one experiment: all stage outputs, logs, final weights"],
        ["results_v1/CVC_frame_split/...", "the leakage study"],
        ["results_v1/KVASIR/<method>/seed_<s>/external_CVC/", "external validation"],
        ["results_v1/analysis/", "tables T1–T16 (Markdown + JSON) and the paper figures"],
        ["results_v1/study_status.json, results_v1/study_logs/", "scheduler state and one log per job"]]
story.append(tblock(table(["Location", "Content"], locs, widths=[45, 55], font=7.4),
                    "Table A.2. Where the v1 results are."))

def glue(flow):
    """Keeps every heading on the same page as the block that follows it."""
    out, i = [], 0
    while i < len(flow):
        if not hasattr(flow[i], "toc_level"):
            out.append(flow[i])
            i += 1
            continue
        group = []
        while i < len(flow) and hasattr(flow[i], "toc_level"):
            group.append(flow[i])
            i += 1
        if i < len(flow) and not isinstance(flow[i], PageBreak):
            nxt = flow[i]
            group += nxt._content if isinstance(nxt, KeepTogether) else [nxt]
            i += 1
        out.append(KeepTogether(group))
    return out


story = glue(story)
doc = Doc(OUT)
OUT.parent.mkdir(parents=True, exist_ok=True)
doc.multiBuild(story)
print("written", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")
