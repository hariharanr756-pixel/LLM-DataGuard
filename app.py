import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback, no_update, ALL, ctx
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# LLM DATAGUARD
# Professional Data Analyst / Data Engineer Research Console
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "results" / "results.json"
POISON_INDEX_PATH = BASE_DIR / "data" / "poisoned" / "poison_indices.csv"
CLEAN_TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
POISONED_TRAIN_PATH = BASE_DIR / "data" / "poisoned" / "train_poisoned.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"

APP_TITLE = "LLM DataGuard"

# ---------- Professional dark analytical theme ----------
BG = "#080B10"
PANEL = "#0F141B"
PANEL_2 = "#121922"
BORDER = "#202A36"
TEXT = "#F2F5F8"
MUTED = "#8A96A5"
ACCENT = "#66E3C4"
ACCENT_2 = "#6EA8FF"
WARNING = "#F0B35B"
DANGER = "#FF6B7A"
PURPLE = "#A78BFA"

FONT = (
    "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "
    "'Segoe UI', sans-serif"
)


# ============================================================
# DATA ACCESS
# ============================================================

def load_results():
    """Load the real experiment output. Never invent dashboard metrics."""
    if not RESULTS_PATH.exists():
        return None

    try:
        with RESULTS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(value):
    """Return percentage points from either 0-1 or 0-100 input."""
    value = num(value)
    return value * 100 if abs(value) <= 1 else value


def fmt_num(value, decimals=2):
    value = num(value)
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def fmt_pct(value, decimals=1):
    return f"{pct(value):.{decimals}f}%"


def summary_value(summary, key, field="mean", default=0):
    item = summary.get(key, {}) if isinstance(summary, dict) else {}

    if isinstance(item, dict):
        return num(item.get(field, default), default)

    return num(item, default)


def file_info(path):
    if not path.exists():
        return {"status": "MISSING", "size": 0, "modified": "—"}

    try:
        stat = path.stat()
        return {
            "status": "READY",
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime(
                "%Y-%m-%d %H:%M"
            ),
        }
    except OSError:
        return {"status": "READY", "size": 0, "modified": "—"}


def build_model_data(results):
    """Normalize the real experiment output for the dashboard.

    Supports the actual results.json structure produced by
    pipeline/run_experiment.py, including top-level clean_model,
    poisoned_model, impact and detection objects, while retaining
    compatibility with per-run/summary formats.
    """
    if not results:
        return {}

    cfg = results.get("config", {}) or {}
    summary = results.get("summary", {}) or {}
    runs = results.get("runs", []) or []
    detection = results.get("detection", {}) or {}

    clean_model = results.get("clean_model", {}) or {}
    poisoned_model = results.get("poisoned_model", {}) or {}
    impact = results.get("impact", {}) or {}

    # ------------------------------------------------------------
    # REAL MODEL METRICS
    # ------------------------------------------------------------
    clean_acc = num(
        clean_model.get("accuracy", summary_value(summary, "clean_accuracy"))
    )
    clean_f1 = num(
        clean_model.get("f1", summary_value(summary, "clean_f1"))
    )
    poison_acc = num(
        poisoned_model.get(
            "accuracy",
            summary_value(summary, "poisoned_accuracy"),
        )
    )
    poison_f1 = num(
        poisoned_model.get(
            "f1",
            summary_value(summary, "poisoned_f1"),
        )
    )

    acc_drop = num(
        impact.get(
            "accuracy_drop",
            summary_value(summary, "accuracy_drop"),
        )
    )
    f1_drop = num(
        impact.get(
            "f1_drop",
            summary_value(summary, "f1_drop"),
        )
    )

    # ------------------------------------------------------------
    # DETECTION METRICS
    # ------------------------------------------------------------
    det_precision = num(
        detection.get(
            "precision",
            summary_value(summary, "detection_precision"),
        )
    )
    det_recall = num(
        detection.get(
            "recall",
            summary_value(summary, "detection_recall"),
        )
    )
    det_f1 = num(
        detection.get(
            "f1",
            summary_value(summary, "detection_f1"),
        )
    )

    # ------------------------------------------------------------
    # DATASET INFORMATION
    # ------------------------------------------------------------
    train_samples = int(
        num(cfg.get("train_samples", detection.get("total_samples", 0)))
    )
    test_samples = int(num(cfg.get("test_samples", 0)))

    actual_poisoned = num(detection.get("actual_poisoned", 0))
    poison_rate = num(cfg.get("poison_rate", 0))

    if poison_rate == 0 and train_samples:
        poison_rate = actual_poisoned / train_samples

    poisoned_samples = int(
        round(
            actual_poisoned
            if actual_poisoned
            else train_samples * poison_rate
        )
    )

    # ------------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------------
    cm = detection.get("confusion_matrix", {}) or {}

    tp = num(detection.get("true_positive", cm.get("true_positive", 0)))
    fp = num(detection.get("false_positive", cm.get("false_positive", 0)))
    fn = num(detection.get("false_negative", cm.get("false_negative", 0)))
    tn = num(detection.get("true_negative", cm.get("true_negative", 0)))

    # ------------------------------------------------------------
    # FLAGGED SAMPLES
    # ------------------------------------------------------------
    avg_flagged = num(
        detection.get(
            "flagged_samples",
            detection.get("avg_n_flagged", detection.get("n_flagged", 0)),
        )
    )

    flag_rate = num(
        detection.get(
            "flag_rate",
            detection.get("avg_flag_rate", 0),
        )
    )

    if flag_rate == 0 and train_samples:
        flag_rate = avg_flagged / train_samples

    # ------------------------------------------------------------
    # PER-RUN DATA
    # ------------------------------------------------------------
    if not isinstance(runs, list):
        runs = []

    # The current experiment stores its aggregate model/detection results
    # at the top level. Create one auditable run row when runs[] is absent.
    if not runs and (clean_model or poisoned_model or detection):
        runs = [
            {
                "seed": cfg.get("random_state", 42),
                "clean": {
                    "accuracy": clean_acc,
                    "f1": clean_f1,
                },
                "poisoned": {
                    "accuracy": poison_acc,
                    "f1": poison_f1,
                },
                "detection": {
                    "n_flagged": avg_flagged,
                    "precision": det_precision,
                    "recall": det_recall,
                    "f1": det_f1,
                    "confusion_matrix": {
                        "true_positive": tp,
                        "false_positive": fp,
                        "false_negative": fn,
                        "true_negative": tn,
                    },
                },
            }
        ]

    # ------------------------------------------------------------
    # NORMALIZED SUMMARY
    # ------------------------------------------------------------
    normalized_summary = dict(summary) if isinstance(summary, dict) else {}
    normalized_summary.update(
        {
            "clean_accuracy": clean_acc,
            "poisoned_accuracy": poison_acc,
            "clean_f1": clean_f1,
            "poisoned_f1": poison_f1,
            "accuracy_drop": acc_drop,
            "f1_drop": f1_drop,
            "detection_precision": det_precision,
            "detection_recall": det_recall,
            "detection_f1": det_f1,
            "avg_n_flagged": avg_flagged,
            "avg_flag_rate": flag_rate,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        }
    )

    # ------------------------------------------------------------
    # STANDARD DEVIATIONS FROM PER-RUN DATA, WHEN AVAILABLE
    # ------------------------------------------------------------
    clean_acc_values = []
    poison_acc_values = []
    clean_f1_values = []
    poison_f1_values = []
    flagged_values = []

    for run in runs:
        if not isinstance(run, dict):
            continue

        clean = run.get("clean", {}) or {}
        poisoned = run.get("poisoned", {}) or {}
        run_detection = run.get("detection", {}) or {}

        if "accuracy" in clean:
            clean_acc_values.append(num(clean.get("accuracy")))
        if "accuracy" in poisoned:
            poison_acc_values.append(num(poisoned.get("accuracy")))
        if "f1" in clean:
            clean_f1_values.append(num(clean.get("f1")))
        if "f1" in poisoned:
            poison_f1_values.append(num(poisoned.get("f1")))

        if "n_flagged" in run_detection:
            flagged_values.append(num(run_detection.get("n_flagged")))
        elif "flagged_samples" in run_detection:
            flagged_values.append(num(run_detection.get("flagged_samples")))

    def sample_std(values):
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    clean_acc_std = sample_std(clean_acc_values)
    poison_acc_std = sample_std(poison_acc_values)
    clean_f1_std = sample_std(clean_f1_values)
    poison_f1_std = sample_std(poison_f1_values)
    flagged_std = sample_std(flagged_values)

    # ------------------------------------------------------------
    # FINAL DASHBOARD DATA
    # ------------------------------------------------------------
    return {
        "cfg": cfg,
        "summary": normalized_summary,
        "runs": runs,
        "n_seeds": len(runs),
        "train_samples": train_samples,
        "test_samples": test_samples,
        "poison_rate": poison_rate,
        "poisoned_samples": poisoned_samples,
        "clean_acc": clean_acc,
        "clean_acc_std": clean_acc_std,
        "poison_acc": poison_acc,
        "poison_acc_std": poison_acc_std,
        "clean_f1": clean_f1,
        "clean_f1_std": clean_f1_std,
        "poison_f1": poison_f1,
        "poison_f1_std": poison_f1_std,
        "acc_drop": acc_drop,
        "acc_drop_std": sample_std(
            [
                num(r.get("clean", {}).get("accuracy", 0))
                - num(r.get("poisoned", {}).get("accuracy", 0))
                for r in runs
                if isinstance(r, dict)
                and isinstance(r.get("clean", {}), dict)
                and isinstance(r.get("poisoned", {}), dict)
                and "accuracy" in r.get("clean", {})
                and "accuracy" in r.get("poisoned", {})
            ]
        ),
        "f1_drop": f1_drop,
        "f1_drop_std": sample_std(
            [
                num(r.get("clean", {}).get("f1", 0))
                - num(r.get("poisoned", {}).get("f1", 0))
                for r in runs
                if isinstance(r, dict)
                and isinstance(r.get("clean", {}), dict)
                and isinstance(r.get("poisoned", {}), dict)
                and "f1" in r.get("clean", {})
                and "f1" in r.get("poisoned", {})
            ]
        ),
        "det_precision": det_precision,
        "det_recall": det_recall,
        "det_f1": det_f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "avg_flagged": avg_flagged,
        "flagged_std": flagged_std,
        "flag_rate": flag_rate,
    }


# ============================================================
# UI HELPERS
# ============================================================

def card(label, value, note="", accent=ACCENT):
    return html.Div(
        [
            html.Div(label.upper(), className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(note, className="kpi-note"),
            html.Div(
                className="kpi-line",
                style={"background": accent},
            ),
        ],
        className="kpi-card",
    )


def section(title, subtitle=""):
    children = [
        html.H2(title, className="section-title")
    ]

    if subtitle:
        children.append(
            html.P(
                subtitle,
                className="section-subtitle"
            )
        )

    return html.Div(
        children,
        className="section-head"
    )


def panel(children, class_name="panel"):
    return html.Div(
        children,
        className=class_name
    )


def pill(text, kind="neutral"):
    return html.Span(
        text,
        className=f"pill {kind}"
    )


def empty_state(message):
    return panel(
        [
            html.Div(
                "NO EXPERIMENT DATA",
                className="empty-eyebrow"
            ),
            html.H2(
                "results/results.json not available",
                className="empty-title"
            ),
            html.P(
                message,
                className="empty-text"
            ),
        ],
        "panel empty-state",
    )


def fig_base(fig, height=360):
    fig.update_layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={
            "family": FONT,
            "color": TEXT
        },
        height=height,
        margin={
            "l": 45,
            "r": 25,
            "t": 55,
            "b": 45
        },
        hoverlabel={
            "bgcolor": "#111923",
            "font_color": TEXT
        },
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": MUTED}
        },
    )

    fig.update_xaxes(
        gridcolor=BORDER,
        zerolinecolor=BORDER,
        linecolor=BORDER,
    )

    fig.update_yaxes(
        gridcolor=BORDER,
        zerolinecolor=BORDER,
        linecolor=BORDER,
    )

    return fig


def metric_bar(title, labels, values, colors=None):
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=[pct(v) for v in values],
            text=[
                f"{pct(v):.1f}%"
                for v in values
            ],
            textposition="auto",
            marker_color=colors or [
                ACCENT,
                ACCENT_2
            ],
            hovertemplate=(
                "%{x}<br>"
                "%{y:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        yaxis_title="Score (%)",
        yaxis_range=[0, 100],
    )

    return fig_base(fig, 350)


def html_table(headers, rows):
    head = html.Thead(
        html.Tr(
            [html.Th(h) for h in headers]
        )
    )

    body = html.Tbody(
        [
            html.Tr(
                [
                    html.Td(
                        row.get(h, "—")
                    )
                    for h in headers
                ]
            )
            for row in rows
        ]
    )

    return html.Div(
        html.Table(
            [head, body],
            className="data-table"
        ),
        className="table-wrap",
    )


def download_buttons():
    return html.Div(
        [
            html.Button(
                "↓ Export run CSV",
                id="download-runs-btn",
                className="button secondary",
            ),
            html.Button(
                "↓ Export results JSON",
                id="download-json-btn",
                className="button secondary",
            ),
            dcc.Download(id="download-runs"),
            dcc.Download(id="download-json"),
        ],
        className="button-row",
    )


# ============================================================
# OVERVIEW
# ============================================================

def overview_page(results):
    if not results:
        return empty_state(
            "Run pipeline/run_experiment.py first. "
            "The dashboard never fabricates experimental results."
        )

    d = build_model_data(results)
    cfg = d["cfg"]
    runs = d["runs"]

    demo = results.get("demo") is True

    status = (
        pill("SYNTHETIC DEMO DATA", "warn")
        if demo
        else pill("EXPERIMENTAL RUN", "success")
    )

    impact_fig = metric_bar(
        "Clean vs. Poisoned Performance",
        [
            "Accuracy · Clean",
            "Accuracy · Poisoned",
            "F1 · Clean",
            "F1 · Poisoned",
        ],
        [
            d["clean_acc"],
            d["poison_acc"],
            d["clean_f1"],
            d["poison_f1"],
        ],
        [
            ACCENT,
            DANGER,
            ACCENT_2,
            WARNING,
        ],
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            status,
                            html.H1(
                                "LLM DataGuard",
                                className="hero-title",
                            ),
                            html.P(
                                "Data poisoning detection, impact analysis "
                                "and mitigation research console",
                                className="hero-subtitle",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                "RUN STATUS",
                                className="eyebrow"
                            ),
                            html.Div(
                                "READY",
                                className="status-ready"
                            ),
                            html.Div(
                                results.get(
                                    "generated_at",
                                    "Timestamp unavailable"
                                )[:19],
                                className="status-meta",
                            ),
                        ],
                        className="status-box",
                    ),
                ],
                className="hero",
            ),

            html.Div(
                [
                    card(
                        "Dataset",
                        str(
                            cfg.get(
                                "dataset_name",
                                "—"
                            )
                        ),
                        "Training / test experiment",
                    ),
                    card(
                        "Model",
                        str(
                            cfg.get(
                                "model_name",
                                "—"
                            )
                        ),
                        "Fine-tuned classifier",
                        ACCENT_2,
                    ),
                    card(
                        "Poison Rate",
                        fmt_pct(
                            d["poison_rate"],
                            1
                        ),
                        str(
                            cfg.get(
                                "poison_strategy",
                                "—"
                            )
                        ),
                        WARNING,
                    ),
                    card(
                        "Train Samples",
                        f"{d['train_samples']:,}",
                        (
                            f"{d['poisoned_samples']:,} "
                            "expected poisoned"
                        ),
                        PURPLE,
                    ),
                    card(
                        "Accuracy Drop",
                        (
                            f"−{pct(d['acc_drop']):.2f} pp"
                        ),
                        (
                            f"± {pct(d['acc_drop_std']):.2f} pp"
                        ),
                        DANGER,
                    ),
                    card(
                        "Detection F1",
                        fmt_pct(
                            d["det_f1"],
                            1
                        ),
                        (
                            f"P {fmt_pct(d['det_precision'], 1)} · "
                            f"R {fmt_pct(d['det_recall'], 1)}"
                        ),
                        ACCENT,
                    ),
                ],
                className="kpi-grid",
            ),

            section(
                "Executive Readout",
                "The high-level view a reviewer can understand in under one minute.",
            ),

            html.Div(
                [
                    panel(
                        [
                            html.Div(
                                "MODEL IMPACT",
                                className="panel-label"
                            ),
                            dcc.Graph(
                                figure=impact_fig,
                                config={
                                    "displayModeBar": False
                                },
                            ),
                        ]
                    ),
                    panel(
                        [
                            html.Div(
                                "RESEARCH SIGNAL",
                                className="panel-label"
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "Accuracy",
                                                className="signal-name"
                                            ),
                                            html.Strong(
                                                f"−{pct(d['acc_drop']):.2f} pp"
                                            ),
                                        ],
                                        className="signal-row",
                                    ),
                                    html.Div(
                                        [
                                            html.Span(
                                                "F1 score",
                                                className="signal-name"
                                            ),
                                            html.Strong(
                                                f"−{pct(d['f1_drop']):.2f} pp"
                                            ),
                                        ],
                                        className="signal-row",
                                    ),
                                    html.Div(
                                        [
                                            html.Span(
                                                "Detector precision",
                                                className="signal-name"
                                            ),
                                            html.Strong(
                                                fmt_pct(
                                                    d["det_precision"],
                                                    1
                                                )
                                            ),
                                        ],
                                        className="signal-row",
                                    ),
                                    html.Div(
                                        [
                                            html.Span(
                                                "Detector recall",
                                                className="signal-name"
                                            ),
                                            html.Strong(
                                                fmt_pct(
                                                    d["det_recall"],
                                                    1
                                                )
                                            ),
                                        ],
                                        className="signal-row",
                                    ),
                                    html.Div(
                                        [
                                            html.Span(
                                                "Avg. flagged",
                                                className="signal-name"
                                            ),
                                            html.Strong(
                                                f"{d['avg_flagged']:.0f}"
                                            ),
                                        ],
                                        className="signal-row",
                                    ),
                                ]
                            ),
                            html.Div(
                                "All values are derived from results.json; "
                                "no dashboard metric is hardcoded.",
                                className="callout",
                            ),
                        ]
                    ),
                ],
                className="two-col",
            ),

            section(
                "Run-Level Evidence",
                "Every seed is visible so the aggregate statistics can be audited.",
            ),

            panel(
                [
                    html.Div(
                        [
                            html.Div(
                                "PER-SEED RESULTS",
                                className="panel-label"
                            ),
                            download_buttons(),
                        ],
                        className="panel-toolbar",
                    ),
                    html_table(
                        [
                            "Seed",
                            "Clean Accuracy",
                            "Poisoned Accuracy",
                            "Accuracy Drop",
                            "Clean F1",
                            "Poisoned F1",
                            "F1 Drop",
                            "Flagged",
                        ],
                        run_table(runs),
                    ),
                ]
            ),

            section(
                "Analyst Notes",
                "Interpret the experiment within its actual scope.",
            ),

            html.Div(
                [
                    panel(
                        [
                            html.Div(
                                "WHAT THIS RUN SHOWS",
                                className="panel-label"
                            ),
                            html.P(
                                (
                                    f"A controlled "
                                    f"{pct(d['poison_rate']):.1f}% "
                                    "label-flipping condition was compared "
                                    f"with clean training on "
                                    f"{d['train_samples']:,} training samples "
                                    f"over {d['n_seeds']} seed(s). "
                                    f"Mean accuracy changed from "
                                    f"{fmt_pct(d['clean_acc'], 2)} to "
                                    f"{fmt_pct(d['poison_acc'], 2)}."
                                ),
                                className="body-text",
                            ),
                        ]
                    ),
                    panel(
                        [
                            html.Div(
                                "WHAT IT DOES NOT PROVE",
                                className="panel-label"
                            ),
                            html.P(
                                "The result is specific to this dataset subset, "
                                "model, hyperparameters and poisoning strategy. "
                                "It should not be generalized to all LLMs or all "
                                "poisoning attacks without additional experiments.",
                                className="body-text",
                            ),
                        ]
                    ),
                ],
                className="two-col",
            ),
        ]
    )


# ============================================================
# IMPACT ANALYSIS
# ============================================================

def impact_page(results):
    if not results:
        return empty_state(
            "No experiment results are available for impact analysis."
        )

    d = build_model_data(results)
    runs = d["runs"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Clean",
            x=["Accuracy", "F1"],
            y=[
                pct(d["clean_acc"]),
                pct(d["clean_f1"])
            ],
            error_y={
                "type": "data",
                "array": [
                    pct(d["clean_acc_std"]),
                    pct(d["clean_f1_std"]),
                ],
                "visible": True,
            },
            marker_color=ACCENT,
        )
    )

    fig.add_trace(
        go.Bar(
            name="Poisoned",
            x=["Accuracy", "F1"],
            y=[
                pct(d["poison_acc"]),
                pct(d["poison_f1"])
            ],
            error_y={
                "type": "data",
                "array": [
                    pct(d["poison_acc_std"]),
                    pct(d["poison_f1_std"]),
                ],
                "visible": True,
            },
            marker_color=DANGER,
        )
    )

    fig.update_layout(
        title="Performance Impact · Mean ± Std",
        barmode="group",
        yaxis_title="Score (%)",
        yaxis_range=[0, 100],
    )

    fig_base(fig, 400)

    seed_rows = []

    for r in runs:
        clean = r.get("clean", {})
        poison = r.get("poisoned", {})

        seed_rows.append(
            {
                "Seed": r.get("seed", "—"),
                "Clean Accuracy": fmt_pct(
                    clean.get("accuracy", 0),
                    2
                ),
                "Poisoned Accuracy": fmt_pct(
                    poison.get("accuracy", 0),
                    2
                ),
                "Accuracy Δ": fmt_pct(
                    num(poison.get("accuracy", 0))
                    - num(clean.get("accuracy", 0)),
                    2
                ),
                "Clean F1": fmt_pct(
                    clean.get("f1", 0),
                    2
                ),
                "Poisoned F1": fmt_pct(
                    poison.get("f1", 0),
                    2
                ),
                "F1 Δ": fmt_pct(
                    num(poison.get("f1", 0))
                    - num(clean.get("f1", 0)),
                    2
                ),
            }
        )

    noisy = (
        d["n_seeds"] > 1
        and abs(d["acc_drop_std"])
        > abs(d["acc_drop"]) * 0.5
    )

    return html.Div(
        [
            section(
                "Impact Analysis",
                "Quantify degradation, variability and per-seed consistency.",
            ),

            html.Div(
                [
                    card(
                        "Clean Accuracy",
                        fmt_pct(d["clean_acc"], 2),
                        f"± {fmt_pct(d['clean_acc_std'], 2)}",
                        ACCENT,
                    ),
                    card(
                        "Poisoned Accuracy",
                        fmt_pct(d["poison_acc"], 2),
                        f"± {fmt_pct(d['poison_acc_std'], 2)}",
                        DANGER,
                    ),
                    card(
                        "Clean F1",
                        fmt_pct(d["clean_f1"], 2),
                        f"± {fmt_pct(d['clean_f1_std'], 2)}",
                        ACCENT_2,
                    ),
                    card(
                        "Poisoned F1",
                        fmt_pct(d["poison_f1"], 2),
                        f"± {fmt_pct(d['poison_f1_std'], 2)}",
                        WARNING,
                    ),
                ],
                className="kpi-grid compact",
            ),

            panel(
                dcc.Graph(
                    figure=fig,
                    config={
                        "displayModeBar": False
                    },
                )
            ),

            html.Div(
                [
                    panel(
                        [
                            html.Div(
                                "ACCURACY EFFECT",
                                className="panel-label"
                            ),
                            html.Div(
                                f"−{pct(d['acc_drop']):.2f} percentage points",
                                className="big-number danger",
                            ),
                            html.P(
                                f"Std across seeds: "
                                f"{pct(d['acc_drop_std']):.2f} pp",
                                className="body-text",
                            ),
                        ]
                    ),
                    panel(
                        [
                            html.Div(
                                "F1 EFFECT",
                                className="panel-label"
                            ),
                            html.Div(
                                f"−{pct(d['f1_drop']):.2f} percentage points",
                                className="big-number warn",
                            ),
                            html.P(
                                f"Std across seeds: "
                                f"{pct(d['f1_drop_std']):.2f} pp",
                                className="body-text",
                            ),
                        ]
                    ),
                ],
                className="two-col",
            ),

            panel(
                [
                    html.Div(
                        "AUDITABLE PER-SEED DATA",
                        className="panel-label"
                    ),
                    html_table(
                        [
                            "Seed",
                            "Clean Accuracy",
                            "Poisoned Accuracy",
                            "Accuracy Δ",
                            "Clean F1",
                            "Poisoned F1",
                            "F1 Δ",
                        ],
                        seed_rows,
                    ),
                ]
            ),

            panel(
                [
                    html.Div(
                        "STATISTICAL CAUTION",
                        className="panel-label"
                    ),
                    html.P(
                        (
                            "Variance is material relative to the observed "
                            "accuracy effect."
                            if noisy
                            else
                            "The observed accuracy effect is not dominated "
                            "by the simple variance heuristic used by "
                            "this dashboard."
                        ),
                        className="body-text",
                    ),
                ],
                "panel warning-panel" if noisy else "panel",
            ),
        ]
    )


# ============================================================
# DETECTION
# ============================================================

def detection_page(results):
    if not results:
        return empty_state(
            "No experiment results are available for detection analysis."
        )

    d = build_model_data(results)

    cm = np.array(
        [
            [d["tn"], d["fp"]],
            [d["fn"], d["tp"]],
        ]
    )

    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=[
                "Predicted Clean",
                "Predicted Poisoned"
            ],
            y=[
                "Actual Clean",
                "Actual Poisoned"
            ],
            text=cm.astype(int),
            texttemplate="%{text}",
            colorscale=[
                [0, "#111821"],
                [0.5, "#30445D"],
                [1, ACCENT],
            ],
            showscale=False,
            hovertemplate=(
                "Count: %{z}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Aggregated Confusion Matrix · All Seeds"
    )

    fig_base(fig, 390)

    metrics = go.Figure(
        go.Bar(
            x=[
                "Precision",
                "Recall",
                "F1"
            ],
            y=[
                pct(d["det_precision"]),
                pct(d["det_recall"]),
                pct(d["det_f1"]),
            ],
            text=[
                fmt_pct(
                    d["det_precision"],
                    1
                ),
                fmt_pct(
                    d["det_recall"],
                    1
                ),
                fmt_pct(
                    d["det_f1"],
                    1
                ),
            ],
            textposition="auto",
            marker_color=[
                ACCENT_2,
                ACCENT,
                PURPLE,
            ],
        )
    )

    metrics.update_layout(
        title="Detector Performance vs Ground Truth",
        yaxis_title="Score (%)",
        yaxis_range=[0, 100],
    )

    fig_base(metrics, 390)

    detection_rows = []

    for r in d["runs"]:
        det = r.get("detection", {})
        cmr = det.get(
            "confusion_matrix",
            {}
        )

        detection_rows.append(
            {
                "Seed": r.get("seed", "—"),
                "Flagged": fmt_num(
                    det.get("n_flagged", 0),
                    0
                ),
                "True Positive": fmt_num(
                    cmr.get("true_positive", 0),
                    0
                ),
                "False Positive": fmt_num(
                    cmr.get("false_positive", 0),
                    0
                ),
                "False Negative": fmt_num(
                    cmr.get("false_negative", 0),
                    0
                ),
                "True Negative": fmt_num(
                    cmr.get("true_negative", 0),
                    0
                ),
            }
        )

    return html.Div(
        [
            section(
                "Detection & Data Quality",
                "Isolation Forest anomaly screening scored against known poisoned indices.",
            ),

            html.Div(
                [
                    card(
                        "Precision",
                        fmt_pct(
                            d["det_precision"],
                            1
                        ),
                        "Correct positive flags",
                        ACCENT_2,
                    ),
                    card(
                        "Recall",
                        fmt_pct(
                            d["det_recall"],
                            1
                        ),
                        "Known poison detected",
                        ACCENT,
                    ),
                    card(
                        "Detector F1",
                        fmt_pct(
                            d["det_f1"],
                            1
                        ),
                        "Balanced detector score",
                        PURPLE,
                    ),
                    card(
                        "Avg Flagged",
                        f"{d['avg_flagged']:.0f}",
                        (
                            f"{pct(d['flag_rate']):.1f}% "
                            "of training data"
                        ),
                        WARNING,
                    ),
                ],
                className="kpi-grid compact",
            ),

            html.Div(
                [
                    panel(
                        dcc.Graph(
                            figure=fig,
                            config={
                                "displayModeBar": False
                            },
                        )
                    ),
                    panel(
                        dcc.Graph(
                            figure=metrics,
                            config={
                                "displayModeBar": False
                            },
                        )
                    ),
                ],
                className="two-col",
            ),

            panel(
                [
                    html.Div(
                        "DETECTION AUDIT TRAIL",
                        className="panel-label"
                    ),
                    html_table(
                        [
                            "Seed",
                            "Flagged",
                            "True Positive",
                            "False Positive",
                            "False Negative",
                            "True Negative",
                        ],
                        detection_rows,
                    ),
                ]
            ),

            panel(
                [
                    html.Div(
                        "INTERPRETATION & LIMITATION",
                        className="panel-label"
                    ),
                    html.P(
                        "The implemented attack is label flipping. "
                        "The text itself remains unchanged, so embedding-space "
                        "anomaly detection is not expected to perfectly identify "
                        "every poisoned record. For a production data-security "
                        "system, provenance, label consistency, loss-based signals, "
                        "source reliability and quarantine/review workflows should "
                        "complement anomaly screening.",
                        className="body-text",
                    ),
                ]
            ),
        ]
    )


# ============================================================
# PIPELINE & DATA LINEAGE
# ============================================================

def pipeline_page(results):
    if not results:
        return empty_state(
            "No experiment configuration is available."
        )

    d = build_model_data(results)
    cfg = d["cfg"]

    files = [
        (
            "results/results.json",
            RESULTS_PATH
        ),
        (
            "data/processed/train.csv",
            CLEAN_TRAIN_PATH
        ),
        (
            "data/processed/test.csv",
            TEST_PATH
        ),
        (
            "data/poisoned/train_poisoned.csv",
            POISONED_TRAIN_PATH
        ),
        (
            "data/poisoned/poison_indices.csv",
            POISON_INDEX_PATH
        ),
    ]

    file_rows = []

    for label, path in files:
        info = file_info(path)

        file_rows.append(
            {
                "Artifact": label,
                "Status": info["status"],
                "Size": (
                    f"{info['size'] / (1024 * 1024):.2f} MB"
                    if info["size"]
                    else "—"
                ),
                "Modified": info["modified"],
            }
        )

    cfg_rows = [
        (
            "Dataset",
            cfg.get("dataset_name", "—")
        ),
        (
            "Model",
            cfg.get("model_name", "—")
        ),
        (
            "Training samples",
            f"{int(num(cfg.get('train_samples', 0))):,}"
        ),
        (
            "Test samples",
            f"{int(num(cfg.get('test_samples', 0))):,}"
        ),
        (
            "Poison rate",
            fmt_pct(
                cfg.get("poison_rate", 0),
                1
            )
        ),
        (
            "Poison strategy",
            cfg.get(
                "poison_strategy",
                "—"
            )
        ),
        (
            "Max sequence length",
            cfg.get(
                "max_seq_len",
                "—"
            )
        ),
        (
            "Epochs",
            cfg.get(
                "epochs",
                "—"
            )
        ),
        (
            "Learning rate",
            cfg.get(
                "learning_rate",
                "—"
            )
        ),
        (
            "Batch size",
            cfg.get(
                "batch_size",
                "—"
            )
        ),
        (
            "Seeds",
            ", ".join(
                map(
                    str,
                    cfg.get(
                        "seeds",
                        []
                    )
                )
            )
        ),
        (
            "Detector",
            cfg.get(
                "detector",
                "—"
            )
        ),
        (
            "Detector contamination",
            fmt_pct(
                cfg.get(
                    "detector_contamination",
                    0
                ),
                1
            )
        ),
    ]

    stages = [
        (
            "01",
            "Dataset",
            "Prepare controlled train/test subsets"
        ),
        (
            "02",
            "Clean training",
            "Establish baseline classifier"
        ),
        (
            "03",
            "Poisoning",
            f"Apply {cfg.get('poison_strategy', 'configured strategy')}"
        ),
        (
            "04",
            "Poisoned training",
            "Train under poisoned condition"
        ),
        (
            "05",
            "Impact analysis",
            "Compare accuracy / F1 / loss"
        ),
        (
            "06",
            "Anomaly screening",
            f"Run {cfg.get('detector', 'detector')}"
        ),
        (
            "07",
            "Ground-truth scoring",
            "Evaluate precision / recall / F1"
        ),
    ]

    return html.Div(
        [
            section(
                "Pipeline & Data Lineage",
                "A data-engineering view of how experiment artifacts become research evidence.",
            ),

            panel(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    n,
                                    className="pipeline-num"
                                ),
                                html.Div(
                                    name,
                                    className="pipeline-name"
                                ),
                                html.Div(
                                    desc,
                                    className="pipeline-desc"
                                ),
                                pill(
                                    "READY",
                                    "success"
                                ),
                            ],
                            className="pipeline-node",
                        )
                        for n, name, desc in stages
                    ],
                    className="pipeline-flow",
                )
            ),

            section(
                "Experiment Configuration",
                "Exact parameters captured by the experiment runner.",
            ),

            panel(
                html_table(
                    ["Parameter", "Value"],
                    [
                        {
                            "Parameter": k,
                            "Value": v
                        }
                        for k, v in cfg_rows
                    ],
                )
            ),

            section(
                "Artifact & Data Availability",
                "Local project artifacts used or produced by the workflow.",
            ),

            panel(
                html_table(
                    [
                        "Artifact",
                        "Status",
                        "Size",
                        "Modified",
                    ],
                    file_rows,
                )
            ),

            section(
                "Engineering Readout",
                "Operational properties that matter when moving from prototype to repeatable experiments.",
            ),

            html.Div(
                [
                    panel(
                        [
                            html.Div(
                                "REPRODUCIBILITY",
                                className="panel-label"
                            ),
                            html.P(
                                (
                                    "Seeds: "
                                    + ", ".join(
                                        map(
                                            str,
                                            cfg.get(
                                                "seeds",
                                                []
                                            )
                                        )
                                    )
                                ),
                                className="body-text",
                            ),
                            html.P(
                                (
                                    "Generated: "
                                    + str(
                                        results.get(
                                            "generated_at",
                                            "—"
                                        )
                                    )
                                ),
                                className="body-text",
                            ),
                            html.P(
                                (
                                    "Git commit: "
                                    + str(
                                        results.get(
                                            "git_commit",
                                            "not recorded"
                                        )
                                    )
                                ),
                                className="body-text",
                            ),
                        ]
                    ),
                    panel(
                        [
                            html.Div(
                                "DATA CONTRACT",
                                className="panel-label"
                            ),
                            html.P(
                                "results.json is the dashboard's metric contract. "
                                "The UI derives metrics from config, summary and "
                                "per-run detection records.",
                                className="body-text",
                            ),
                            html.P(
                                "Missing artifacts are surfaced as MISSING rather "
                                "than silently replaced.",
                                className="body-text",
                            ),
                        ]
                    ),
                ],
                className="two-col",
            ),
        ]
    )


# ============================================================
# METHODOLOGY & MITIGATION
# ============================================================

def methodology_page(results):
    cfg = (
        results or {}
    ).get(
        "config",
        {}
    )

    return html.Div(
        [
            section(
                "Methodology & Mitigation",
                "Research framing, limitations and the production-oriented mitigation path.",
            ),

            html.Div(
                [
                    panel(
                        [
                            html.Div(
                                "01 · DATASET PREPARATION",
                                className="panel-label"
                            ),
                            html.P(
                                (
                                    f"Use a controlled "
                                    f"{cfg.get('dataset_name', 'dataset')} "
                                    "train/test split with fixed seeds so clean "
                                    "and poisoned conditions remain comparable."
                                ),
                                className="body-text",
                            ),
                        ]
                    ),
                    panel(
                        [
                            html.Div(
                                "02 · BASELINE",
                                className="panel-label"
                            ),
                            html.P(
                                (
                                    f"Fine-tune "
                                    f"{cfg.get('model_name', 'the configured model')} "
                                    "on clean data and record baseline "
                                    "accuracy, F1 and loss."
                                ),
                                className="body-text",
                            ),
                        ]
                    ),
                    panel(
                        [
                            html.Div(
                                "03 · POISONING",
                                className="panel-label"
                            ),
                            html.P(
                                (
                                    f"Apply the configured "
                                    f"{cfg.get('poison_strategy', 'poisoning strategy')} "
                                    f"at {fmt_pct(cfg.get('poison_rate', 0), 1)} "
                                    "and persist exact poisoned indices as ground truth."
                                ),
                                className="body-text",
                            ),
                        ]
                    ),
                    panel(
                        [
                            html.Div(
                                "04 · DETECTION",
                                className="panel-label"
                            ),
                            html.P(
                                (
                                    f"Screen poisoned-run embeddings with "
                                    f"{cfg.get('detector', 'the configured detector')} "
                                    "and score flags against ground truth."
                                ),
                                className="body-text",
                            ),
                        ]
                    ),
                ],
                className="four-col",
            ),

            section(
                "Production Mitigation Workflow",
                "The dashboard documents mitigation as a workflow, not as a claim that the current experiment implements every production control.",
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "01",
                                className="mit-num"
                            ),
                            html.Div(
                                "Detect",
                                className="mit-title"
                            ),
                            html.Div(
                                "Identify suspicious records using multiple independent signals.",
                                className="mit-text",
                            ),
                        ],
                        className="mit-card",
                    ),
                    html.Div(
                        [
                            html.Div(
                                "02",
                                className="mit-num"
                            ),
                            html.Div(
                                "Quarantine",
                                className="mit-title"
                            ),
                            html.Div(
                                "Prevent high-risk records from entering the training set until reviewed.",
                                className="mit-text",
                            ),
                        ],
                        className="mit-card",
                    ),
                    html.Div(
                        [
                            html.Div(
                                "03",
                                className="mit-num"
                            ),
                            html.Div(
                                "Validate",
                                className="mit-title"
                            ),
                            html.Div(
                                "Check labels, provenance, source reliability and consistency.",
                                className="mit-text",
                            ),
                        ],
                        className="mit-card",
                    ),
                    html.Div(
                        [
                            html.Div(
                                "04",
                                className="mit-num"
                            ),
                            html.Div(
                                "Remediate",
                                className="mit-title"
                            ),
                            html.Div(
                                "Remove, re-label or re-weight confirmed suspicious samples.",
                                className="mit-text",
                            ),
                        ],
                        className="mit-card",
                    ),
                    html.Div(
                        [
                            html.Div(
                                "05",
                                className="mit-num"
                            ),
                            html.Div(
                                "Retrain",
                                className="mit-title"
                            ),
                            html.Div(
                                "Retrain and compare clean, poisoned and mitigated conditions.",
                                className="mit-text",
                            ),
                        ],
                        className="mit-card",
                    ),
                    html.Div(
                        [
                            html.Div(
                                "06",
                                className="mit-num"
                            ),
                            html.Div(
                                "Audit",
                                className="mit-title"
                            ),
                            html.Div(
                                "Persist data lineage, configuration, metrics and model versions.",
                                className="mit-text",
                            ),
                        ],
                        className="mit-card",
                    ),
                ],
                className="mit-grid",
            ),

            section(
                "Scope & Limitations",
                "What a reviewer should understand before interpreting the results.",
            ),

            panel(
                [
                    html.Div(
                        "SCOPE",
                        className="panel-label"
                    ),
                    html.P(
                        "This dashboard presents a controlled research experiment. "
                        "The measured outcomes are not a universal benchmark for all "
                        "LLM architectures, datasets or attack types.",
                        className="body-text",
                    ),
                    html.P(
                        "The current poisoning mechanism is label flipping and the "
                        "current detector is embedding-space anomaly screening. "
                        "A production security layer would require broader validation.",
                        className="body-text",
                    ),
                ]
            ),
        ]
    )


# ============================================================
# RUN EXPLORER
# ============================================================

def run_table(runs):
    rows = []

    for r in runs:
        clean = r.get("clean", {})
        poison = r.get("poisoned", {})
        det = r.get("detection", {})

        rows.append(
            {
                "Seed": r.get("seed", "—"),
                "Clean Accuracy": fmt_pct(
                    clean.get("accuracy", 0),
                    2
                ),
                "Poisoned Accuracy": fmt_pct(
                    poison.get("accuracy", 0),
                    2
                ),
                "Accuracy Drop": fmt_pct(
                    num(clean.get("accuracy", 0))
                    - num(poison.get("accuracy", 0)),
                    2
                ),
                "Clean F1": fmt_pct(
                    clean.get("f1", 0),
                    2
                ),
                "Poisoned F1": fmt_pct(
                    poison.get("f1", 0),
                    2
                ),
                "F1 Drop": fmt_pct(
                    num(clean.get("f1", 0))
                    - num(poison.get("f1", 0)),
                    2
                ),
                "Flagged": fmt_num(
                    det.get("n_flagged", 0),
                    0
                ),
            }
        )

    return rows


def explorer_page(results):
    if not results:
        return empty_state(
            "No per-run data is available."
        )

    d = build_model_data(results)
    rows = []

    for r in d["runs"]:
        clean = r.get("clean", {})
        poison = r.get("poisoned", {})
        det = r.get("detection", {})
        cm = det.get(
            "confusion_matrix",
            {}
        )

        rows.append(
            {
                "Seed": r.get("seed", "—"),
                "Clean Accuracy": fmt_pct(
                    clean.get("accuracy", 0),
                    3
                ),
                "Poisoned Accuracy": fmt_pct(
                    poison.get("accuracy", 0),
                    3
                ),
                "Clean F1": fmt_pct(
                    clean.get("f1", 0),
                    3
                ),
                "Poisoned F1": fmt_pct(
                    poison.get("f1", 0),
                    3
                ),
                "Flagged": fmt_num(
                    det.get("n_flagged", 0),
                    0
                ),
                "TP": fmt_num(
                    cm.get("true_positive", 0),
                    0
                ),
                "FP": fmt_num(
                    cm.get("false_positive", 0),
                    0
                ),
                "FN": fmt_num(
                    cm.get("false_negative", 0),
                    0
                ),
                "TN": fmt_num(
                    cm.get("true_negative", 0),
                    0
                ),
            }
        )

    return html.Div(
        [
            section(
                "Run Explorer",
                "Auditable per-seed view for analysts and experiment maintainers.",
            ),

            panel(
                [
                    html.Div(
                        [
                            html.Div(
                                "RAW RUN MATRIX",
                                className="panel-label"
                            ),
                            download_buttons(),
                        ],
                        className="panel-toolbar",
                    ),
                    html_table(
                        [
                            "Seed",
                            "Clean Accuracy",
                            "Poisoned Accuracy",
                            "Clean F1",
                            "Poisoned F1",
                            "Flagged",
                            "TP",
                            "FP",
                            "FN",
                            "TN",
                        ],
                        rows,
                    ),
                ]
            ),

            section(
                "Raw Experiment Metadata",
                "The dashboard does not alter the experiment record.",
            ),

            panel(
                [
                    html.Pre(
                        json.dumps(
                            results,
                            indent=2
                        ),
                        className="json-view",
                    )
                ]
            ),
        ]
    )


# ============================================================
# NAVIGATION
# ============================================================

# ============================================================
# UPLOAD & SCAN PAGE
# ============================================================

def upload_scan_page(results=None):
    return html.Div(
        [
            html.Div(
                [
                    html.Div("UPLOAD & SCAN", className="eyebrow"),
                    html.H1("Scan Your Own Dataset", className="page-title"),
                    html.P(
                        "Upload your own text dataset and DataGuard will validate it, "
                        "run TF-IDF feature analysis, and screen records for potential anomalies.",
                        className="page-subtitle",
                    ),
                ],
                className="page-header",
            ),
            panel(
                [
                    html.Div("DATASET UPLOAD", className="section-kicker"),
                    dcc.Upload(
                        id="upload-dataset",
                        children=html.Div(
                            [
                                html.Div("↑", style={"fontSize": "36px", "marginBottom": "10px"}),
                                html.Div(
                                    "Drag & Drop your dataset here",
                                    style={"fontSize": "18px", "fontWeight": "700"},
                                ),
                                html.Div(
                                    "or click to browse",
                                    style={"color": MUTED, "marginTop": "6px"},
                                ),
                                html.Div(
                                    "Supported: CSV • TSV • JSON • TXT • extensionless text files",
                                    style={
                                        "color": ACCENT,
                                        "fontSize": "12px",
                                        "marginTop": "14px",
                                    },
                                ),
                            ]
                        ),
                        # Intentionally broad so extensionless UCI-style text files can be selected.
                        accept="",
                        multiple=False,
                        style={
                            "width": "100%",
                            "minHeight": "220px",
                            "border": f"1px dashed {ACCENT}",
                            "borderRadius": "14px",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "textAlign": "center",
                            "cursor": "pointer",
                            "background": PANEL_2,
                            "boxSizing": "border-box",
                            "padding": "30px",
                        },
                    ),
                    html.Div(
                        id="upload-status",
                        children="No dataset uploaded yet.",
                        style={"marginTop": "18px", "color": MUTED},
                    ),
                ],
                class_name="panel",
            ),
            html.Div(id="upload-kpis", style={"marginTop": "18px"}),
            panel(
                [
                    html.Div("SCAN RESULT", className="section-kicker"),
                    html.Div(id="upload-summary", children="Upload a dataset to begin scanning."),
                    dcc.Graph(
                        id="upload-anomaly-chart",
                        figure=go.Figure(),
                        style={"marginTop": "12px"},
                        config={"displayModeBar": False},
                    ),
                ],
                class_name="panel",
            ),
            panel(
                [
                    html.Div(
                        [
                            html.Div("FLAGGED RECORDS", className="section-kicker"),
                            html.Button(
                                "Download Flagged Records",
                                id="download-upload-flagged-btn",
                                className="secondary-btn",
                                n_clicks=0,
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                            "gap": "12px",
                            "flexWrap": "wrap",
                        },
                    ),
                    html.Div(
                        id="upload-table",
                        children="No scan results yet.",
                        style={"marginTop": "14px"},
                    ),
                ],
                class_name="panel",
            ),
            dcc.Store(id="upload-flagged-store", data=[]),
            dcc.Download(id="download-upload-flagged"),
            html.Div(
                "Detection note: without verified poisoning ground truth, flagged records are reported as potentially suspicious/anomalous, not confirmed poisoned records.",
                style={
                    "color": MUTED,
                    "fontSize": "12px",
                    "marginTop": "12px",
                },
            ),
        ]
    )

# ============================================================
# NAVIGATION
# ============================================================

NAV_ITEMS = [
    ("Overview", "01"),
    ("Impact Analysis", "02"),
    ("Detection & QA", "03"),
    ("Upload & Scan", "04"),
    ("Run Explorer", "05"),
    ("Pipeline & Lineage", "06"),
    ("Methodology & Mitigation", "07"),
]


def nav():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        "🛡",
                        className="brand-icon"
                    ),
                    html.Div(
                        [
                            html.Div(
                                "LLM",
                                className="brand-small"
                            ),
                            html.Div(
                                "DATAGUARD",
                                className="brand-name"
                            ),
                        ]
                    ),
                ],
                className="brand",
            ),

            html.Div(
                "RESEARCH CONSOLE",
                className="sidebar-label"
            ),

            html.Div(
                [
                    html.Button(
                        [
                            html.Span(
                                number,
                                className="nav-number"
                            ),
                            html.Span(label),
                        ],
                        id={
                            "type": "nav",
                            "page": label
                        },
                        className="nav-button",
                    )
                    for label, number in NAV_ITEMS
                ]
            ),

            html.Div(
                className="sidebar-spacer"
            ),

            html.Div(
                [
                    html.Div(
                        "PROJECT STATUS",
                        className="sidebar-label"
                    ),
                    html.Div(
                        [
                            html.Span(
                                "●",
                                className="status-dot"
                            ),
                            html.Span(
                                "Experiment complete"
                            ),
                        ],
                        className="side-status",
                    ),
                    html.Div(
                        "Metrics sourced from results/results.json",
                        className="side-note",
                    ),
                ],
                className="sidebar-footer",
            ),
        ],
        className="sidebar",
    )


def shell():
    return html.Div(
        [
            nav(),

            html.Main(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        "LLM DataGuard / Research Dashboard",
                                        className="topline",
                                    ),
                                    html.Div(
                                        id="page-breadcrumb",
                                        className="breadcrumb",
                                    ),
                                ]
                            ),

                            html.Div(
                                [
                                    html.Button(
                                        "↻ Refresh",
                                        id="refresh-btn",
                                        className="button secondary",
                                    ),
                                    html.Div(
                                        id="data-status",
                                        className="data-status",
                                    ),
                                ],
                                className="top-actions",
                            ),
                        ],
                        className="topbar",
                    ),

                    dcc.Store(
                        id="current-page",
                        data="Overview"
                    ),

                    dcc.Loading(
                        id="page-loading",
                        type="dot",
                        color=ACCENT,
                        children=html.Div(
                            id="page-content"
                        ),
                    ),

                    html.Footer(
                        "LLM DataGuard · Data Poisoning Detection & Mitigation Research · Analytical prototype",
                        className="footer",
                    ),
                ],
                className="main",
            ),
        ],
        className="app-shell",
    )


# ============================================================
# APP
# ============================================================

app = Dash(
    __name__,
    title="LLM DataGuard | Research Console",
    suppress_callback_exceptions=True,
)

app.layout = shell()


@callback(
    Output(
        "current-page",
        "data"
    ),
    Input(
        {
            "type": "nav",
            "page": ALL
        },
        "n_clicks",
    ),
    State(
        "current-page",
        "data"
    ),
    prevent_initial_call=True,
)
def change_page(_clicks, current):
    triggered = ctx.triggered_id

    if (
        isinstance(triggered, dict)
        and triggered.get("type") == "nav"
    ):
        return triggered.get(
            "page",
            current
        )

    return current


@callback(
    Output(
        "page-content",
        "children"
    ),
    Output(
        "page-breadcrumb",
        "children"
    ),
    Output(
        "data-status",
        "children"
    ),
    Input(
        "current-page",
        "data"
    ),
    Input(
        "refresh-btn",
        "n_clicks"
    ),
)
def render_page(page, _refresh):
    results = load_results()

    mapping = {
        "Overview": overview_page,
        "Impact Analysis": impact_page,
        "Detection & QA": detection_page,
        "Upload & Scan": upload_scan_page,
        "Run Explorer": explorer_page,
        "Pipeline & Lineage": pipeline_page,
        "Methodology & Mitigation": methodology_page,
    }

    renderer = mapping.get(
        page,
        overview_page
    )

    content = renderer(results)

    status = (
        "● LIVE RESULTS"
        if results
        else "● WAITING FOR RESULTS"
    )

    return (
        content,
        page.upper(),
        status,
    )


@callback(
    Output(
        "download-runs",
        "data"
    ),
    Input(
        "download-runs-btn",
        "n_clicks"
    ),
    prevent_initial_call=True,
)
def download_runs(_n):
    results = load_results()

    if not results:
        return no_update

    rows = []

    for r in results.get(
        "runs",
        []
    ):
        clean = r.get(
            "clean",
            {}
        )
        poison = r.get(
            "poisoned",
            {}
        )
        det = r.get(
            "detection",
            {}
        )
        cm = det.get(
            "confusion_matrix",
            {}
        )

        rows.append(
            {
                "seed": r.get(
                    "seed"
                ),
                "clean_accuracy": clean.get(
                    "accuracy"
                ),
                "poisoned_accuracy": poison.get(
                    "accuracy"
                ),
                "clean_f1": clean.get(
                    "f1"
                ),
                "poisoned_f1": poison.get(
                    "f1"
                ),
                "n_flagged": det.get(
                    "n_flagged"
                ),
                "true_positive": cm.get(
                    "true_positive"
                ),
                "false_positive": cm.get(
                    "false_positive"
                ),
                "false_negative": cm.get(
                    "false_negative"
                ),
                "true_negative": cm.get(
                    "true_negative"
                ),
            }
        )

    df = pd.DataFrame(rows)

    return dcc.send_data_frame(
        df.to_csv,
        "llm_dataguard_run_results.csv",
        index=False,
    )


@callback(
    Output(
        "download-json",
        "data"
    ),
    Input(
        "download-json-btn",
        "n_clicks"
    ),
    prevent_initial_call=True,
)
def download_json(_n):
    results = load_results()

    if not results:
        return no_update

    return {
        "content": json.dumps(
            results,
            indent=2
        ),
        "filename": (
            "llm_dataguard_results.json"
        ),
        "type": "application/json",
    }



app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            * { box-sizing: border-box; }

            html, body {
                margin: 0;
                background: #080B10;
                color: #F2F5F8;
                font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                    BlinkMacSystemFont, "Segoe UI", sans-serif;
            }

            body {
                min-height: 100vh;
            }

            button, input, textarea, select {
                font-family: inherit;
            }

            .app-shell {
                min-height: 100vh;
                display: flex;
                background:
                    radial-gradient(circle at 75% -10%, rgba(102,227,196,.08), transparent 30%),
                    radial-gradient(circle at 100% 60%, rgba(110,168,255,.05), transparent 28%),
                    #080B10;
            }

            .sidebar {
                width: 260px;
                min-height: 100vh;
                position: fixed;
                left: 0;
                top: 0;
                bottom: 0;
                z-index: 10;
                display: flex;
                flex-direction: column;
                padding: 24px 16px;
                background: rgba(10,14,19,.96);
                border-right: 1px solid #202A36;
                backdrop-filter: blur(16px);
            }

            .brand {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 2px 8px 28px;
            }

            .brand-icon {
                width: 40px;
                height: 40px;
                display: grid;
                place-items: center;
                border: 1px solid #2C5A50;
                border-radius: 11px;
                background: #0E211D;
                font-size: 20px;
            }

            .brand-small {
                color: #758291;
                font-size: 9px;
                font-weight: 800;
                letter-spacing: 2px;
            }

            .brand-name {
                color: #F3F6F8;
                font-size: 15px;
                font-weight: 800;
                letter-spacing: 1.3px;
            }

            .sidebar-label {
                margin: 0 8px 10px;
                color: #5F6C7B;
                font-size: 9px;
                font-weight: 800;
                letter-spacing: 1.5px;
            }

            .nav-button {
                width: 100%;
                display: flex;
                align-items: center;
                gap: 11px;
                margin: 3px 0;
                padding: 11px 12px;
                border: 1px solid transparent;
                border-radius: 9px;
                background: transparent;
                color: #8D99A7;
                font-size: 12px;
                font-weight: 600;
                text-align: left;
                cursor: pointer;
                transition: .18s ease;
            }

            .nav-button:hover {
                color: #EAF0F4;
                background: #111821;
                border-color: #202A36;
            }

            .nav-number {
                width: 23px;
                color: #4D5B69;
                font-size: 9px;
                font-weight: 800;
                letter-spacing: .5px;
            }

            .sidebar-spacer { flex: 1; }

            .sidebar-footer {
                padding: 15px 8px 4px;
                border-top: 1px solid #202A36;
            }

            .side-status {
                display: flex;
                align-items: center;
                gap: 7px;
                color: #A9B5C1;
                font-size: 11px;
                font-weight: 600;
            }

            .status-dot {
                color: #66E3C4;
                font-size: 11px;
            }

            .side-note {
                margin-top: 8px;
                color: #586575;
                font-size: 9px;
                line-height: 1.5;
            }

            .main {
                width: calc(100% - 260px);
                margin-left: 260px;
                min-height: 100vh;
                padding: 0 34px 30px;
            }

            .topbar {
                min-height: 78px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #18212B;
                margin-bottom: 8px;
            }

            .topline {
                color: #5F6C7B;
                font-size: 9px;
                font-weight: 700;
                letter-spacing: 1.2px;
                text-transform: uppercase;
            }

            .breadcrumb {
                margin-top: 4px;
                color: #C7D0D9;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: .7px;
            }

            .top-actions {
                display: flex;
                align-items: center;
                gap: 14px;
            }

            .data-status {
                color: #718090;
                font-size: 9px;
                font-weight: 800;
                letter-spacing: .8px;
            }

            .button {
                border: 1px solid #273443;
                border-radius: 8px;
                padding: 8px 11px;
                color: #D7E0E8;
                background: #101720;
                font-size: 10px;
                font-weight: 700;
                cursor: pointer;
            }

            .button:hover {
                border-color: #3A4B5E;
                background: #141D27;
            }

            .button-row {
                display: flex;
                gap: 7px;
                flex-wrap: wrap;
            }

            .hero {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 20px;
                padding: 25px 0 26px;
                border-bottom: 1px solid #202A36;
            }

            .hero-title {
                margin: 12px 0 5px;
                color: #F5F8FA;
                font-size: clamp(29px, 4vw, 44px);
                font-weight: 800;
                letter-spacing: -1.8px;
                line-height: 1.05;
            }

            .hero-subtitle {
                max-width: 700px;
                margin: 0;
                color: #8290A0;
                font-size: 13px;
                line-height: 1.65;
            }

            .eyebrow {
                color: #667585;
                font-size: 8px;
                font-weight: 800;
                letter-spacing: 1.4px;
            }

            .status-box {
                min-width: 160px;
                padding: 13px 15px;
                border: 1px solid #1E3D37;
                border-radius: 10px;
                background: #0C1715;
                text-align: right;
            }

            .status-ready {
                margin-top: 4px;
                color: #66E3C4;
                font-size: 13px;
                font-weight: 800;
                letter-spacing: .6px;
            }

            .status-meta {
                margin-top: 4px;
                color: #566575;
                font-size: 8px;
            }

            .pill {
                display: inline-flex;
                align-items: center;
                width: fit-content;
                padding: 5px 8px;
                border: 1px solid #2A3440;
                border-radius: 999px;
                color: #8E9AA8;
                background: #10161D;
                font-size: 8px;
                font-weight: 800;
                letter-spacing: .8px;
            }

            .pill.success {
                color: #66E3C4;
                border-color: #245548;
                background: #0D211C;
            }

            .pill.warn {
                color: #F0B35B;
                border-color: #5B4425;
                background: #21170B;
            }

            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(6, minmax(0, 1fr));
                gap: 10px;
                margin-top: 16px;
            }

            .kpi-grid.compact {
                grid-template-columns: repeat(4, minmax(0, 1fr));
            }

            .kpi-card {
                position: relative;
                overflow: hidden;
                min-height: 122px;
                padding: 16px;
                border: 1px solid #202A36;
                border-radius: 11px;
                background: linear-gradient(145deg, #111821, #0D131A);
                box-shadow: 0 10px 30px rgba(0,0,0,.13);
            }

            .kpi-label {
                color: #6F7D8C;
                font-size: 8px;
                font-weight: 800;
                letter-spacing: 1px;
            }

            .kpi-value {
                margin-top: 10px;
                color: #F0F4F7;
                font-size: 22px;
                font-weight: 800;
                letter-spacing: -.5px;
            }

            .kpi-note {
                margin-top: 6px;
                min-height: 15px;
                color: #657282;
                font-size: 9px;
                line-height: 1.45;
            }

            .kpi-line {
                position: absolute;
                left: 0;
                right: 0;
                bottom: 0;
                height: 2px;
                opacity: .8;
            }

            .section-head {
                margin: 34px 0 13px;
            }

            .section-title {
                margin: 0;
                color: #E8EDF2;
                font-size: 18px;
                font-weight: 800;
                letter-spacing: -.3px;
            }

            .section-subtitle {
                margin: 5px 0 0;
                color: #6F7D8C;
                font-size: 10px;
                line-height: 1.55;
            }

            .two-col {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 12px;
            }

            .four-col {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 10px;
            }

            .panel {
                min-width: 0;
                padding: 17px;
                border: 1px solid #202A36;
                border-radius: 11px;
                background: #0F141B;
                box-shadow: 0 10px 35px rgba(0,0,0,.10);
            }

            .panel-label {
                margin-bottom: 11px;
                color: #657383;
                font-size: 8px;
                font-weight: 800;
                letter-spacing: 1.2px;
            }

            .panel-toolbar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
                margin-bottom: 10px;
            }

            .signal-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 13px 0;
                border-bottom: 1px solid #1B2530;
                color: #DDE4EA;
                font-size: 11px;
            }

            .signal-row:last-child { border-bottom: 0; }

            .signal-row strong {
                color: #EAF1F5;
                font-size: 12px;
            }

            .signal-name { color: #778493; }

            .callout {
                margin-top: 17px;
                padding: 11px 12px;
                border-left: 2px solid #2C6A5B;
                border-radius: 0 7px 7px 0;
                background: #0C1715;
                color: #718090;
                font-size: 9px;
                line-height: 1.55;
            }

            .body-text {
                margin: 6px 0;
                color: #8895A4;
                font-size: 10px;
                line-height: 1.7;
            }

            .big-number {
                margin: 8px 0 3px;
                color: #F0F4F7;
                font-size: 26px;
                font-weight: 800;
            }

            .big-number.danger { color: #FF7D8A; }
            .big-number.warn { color: #F0B35B; }

            .data-table {
                width: 100%;
                border-collapse: collapse;
                min-width: 720px;
                font-size: 9px;
            }

            .data-table th {
                padding: 10px 11px;
                color: #657383;
                border-bottom: 1px solid #27323E;
                background: #111821;
                font-size: 8px;
                font-weight: 800;
                letter-spacing: .6px;
                text-align: left;
                text-transform: uppercase;
                white-space: nowrap;
            }

            .data-table td {
                padding: 10px 11px;
                color: #AAB5C0;
                border-bottom: 1px solid #1B2530;
                white-space: nowrap;
            }

            .data-table tr:hover td {
                background: #111821;
                color: #E0E7EC;
            }

            .table-wrap {
                width: 100%;
                overflow-x: auto;
                border: 1px solid #1A2430;
                border-radius: 8px;
            }

            .pipeline-flow {
                display: grid;
                grid-template-columns: repeat(7, minmax(110px, 1fr));
                gap: 7px;
                overflow-x: auto;
            }

            .pipeline-node {
                min-height: 125px;
                padding: 13px;
                border: 1px solid #202A36;
                border-radius: 9px;
                background: #111821;
            }

            .pipeline-num {
                color: #4E5D6D;
                font-size: 8px;
                font-weight: 800;
            }

            .pipeline-name {
                margin-top: 12px;
                color: #DCE4EA;
                font-size: 10px;
                font-weight: 800;
            }

            .pipeline-desc {
                min-height: 39px;
                margin: 6px 0 10px;
                color: #718090;
                font-size: 8px;
                line-height: 1.45;
            }

            .mit-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 10px;
            }

            .mit-card {
                padding: 15px;
                border: 1px solid #202A36;
                border-radius: 10px;
                background: #111821;
            }

            .mit-num {
                color: #66E3C4;
                font-size: 9px;
                font-weight: 800;
            }

            .mit-title {
                margin-top: 7px;
                color: #E2E9EE;
                font-size: 11px;
                font-weight: 800;
            }

            .mit-text {
                margin-top: 5px;
                color: #778594;
                font-size: 9px;
                line-height: 1.55;
            }

            .warning-panel {
                border-color: #57421F;
                background: #15120D;
            }

            .empty-state {
                margin-top: 35px;
                padding: 45px;
                text-align: center;
            }

            .empty-eyebrow {
                color: #F0B35B;
                font-size: 9px;
                font-weight: 800;
                letter-spacing: 1.4px;
            }

            .empty-title {
                margin: 11px 0 7px;
                color: #E9EEF2;
                font-size: 23px;
            }

            .empty-text {
                max-width: 620px;
                margin: 0 auto;
                color: #758291;
                font-size: 11px;
                line-height: 1.7;
            }

            .json-view {
                max-height: 600px;
                overflow: auto;
                margin: 0;
                padding: 14px;
                border: 1px solid #1A2430;
                border-radius: 8px;
                background: #0A0F15;
                color: #91A0AF;
                font-size: 9px;
                line-height: 1.55;
                white-space: pre-wrap;
            }

            .footer {
                margin-top: 38px;
                padding: 20px 0 5px;
                border-top: 1px solid #18212B;
                color: #4F5C6B;
                font-size: 8px;
                text-align: center;
                letter-spacing: .5px;
            }

            .dash-loading {
                color: #66E3C4 !important;
            }

            @media (max-width: 1250px) {
                .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
                .kpi-grid.compact { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                .four-col { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                .pipeline-flow { grid-template-columns: repeat(4, minmax(150px, 1fr)); }
            }

            @media (max-width: 900px) {
                .sidebar {
                    width: 210px;
                }

                .main {
                    width: calc(100% - 210px);
                    margin-left: 210px;
                    padding: 0 18px 25px;
                }

                .two-col {
                    grid-template-columns: 1fr;
                }

                .mit-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }

            @media (max-width: 650px) {
                .sidebar {
                    position: relative;
                    width: 100%;
                    min-height: auto;
                }

                .app-shell {
                    display: block;
                }

                .main {
                    width: 100%;
                    margin-left: 0;
                }

                .hero {
                    display: block;
                }

                .status-box {
                    margin-top: 15px;
                    text-align: left;
                }

                .kpi-grid,
                .kpi-grid.compact,
                .four-col,
                .mit-grid {
                    grid-template-columns: 1fr;
                }

                .topbar {
                    gap: 10px;
                    flex-wrap: wrap;
                    padding: 12px 0;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# ============================================================
# UPLOAD DATASET / SCAN CALLBACKS
# ============================================================

def _parse_uploaded_dataset(decoded, filename):
    """Parse common text dataset formats, including extensionless TSV files."""
    import io

    filename_lower = (filename or "").lower()
    text_data = decoded.decode("utf-8", errors="replace")

    # JSON is unambiguous and should be handled first.
    if filename_lower.endswith(".json"):
        try:
            return pd.read_json(io.StringIO(text_data))
        except ValueError:
            payload = json.loads(text_data)
            if isinstance(payload, dict):
                return pd.DataFrame(payload)
            return pd.json_normalize(payload)

    lines = [line for line in text_data.splitlines() if line.strip()]
    sample = lines[:50]

    # UCI SMS Spam Collection and similar label<TAB>text files.
    if any("\t" in line for line in sample):
        rows = []
        for line in lines:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                rows.append({"label": parts[0].strip(), "text": parts[1].strip()})
            else:
                rows.append({"text": line.strip()})
        return pd.DataFrame(rows)

    # Standard CSV. If it is not parseable as a table, treat each line as text.
    if filename_lower.endswith((".csv", ".tsv")):
        try:
            sep = "\t" if filename_lower.endswith(".tsv") else ","
            return pd.read_csv(io.StringIO(text_data), sep=sep)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
            pass

    # Try comma-separated content even if the file has no extension.
    try:
        df = pd.read_csv(io.StringIO(text_data))
        if len(df.columns) > 1:
            return df
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        pass

    # Plain text: one non-empty line = one record.
    return pd.DataFrame({"text": [line.strip() for line in lines]})


def _find_text_column(df):
    preferred = [
        "text", "content", "review", "message", "sms", "sentence",
        "body", "comment", "description", "document",
    ]
    normalized = {str(col).strip().lower(): col for col in df.columns}

    for name in preferred:
        if name in normalized:
            return normalized[name]

    object_cols = [
        col for col in df.columns
        if pd.api.types.is_object_dtype(df[col])
        or pd.api.types.is_string_dtype(df[col])
    ]
    if object_cols:
        return object_cols[0]

    return None


def _ground_truth(df):
    """Return (truth_array, source) when the upload contains usable ground truth."""
    if "is_poisoned" in df.columns:
        raw = df["is_poisoned"]
        truth = raw.astype(str).str.strip().str.lower().isin(
            {"1", "true", "yes", "y", "poisoned", "poison"}
        ).astype(int).to_numpy()
        return truth, "is_poisoned column"

    if "original_label" in df.columns and "label" in df.columns:
        original = df["original_label"].astype(str).str.strip()
        current = df["label"].astype(str).str.strip()
        truth = (original != current).astype(int).to_numpy()
        return truth, "original_label vs label"

    return None, "No verified poisoning ground truth provided"


def _empty_upload_figure():
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"family": FONT, "color": TEXT},
        height=300,
        margin={"l": 45, "r": 25, "t": 45, "b": 45},
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "Upload a dataset to view anomaly scores",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"color": MUTED, "size": 15},
            }
        ],
    )
    return fig


@app.callback(
    Output("upload-status", "children"),
    Output("upload-kpis", "children"),
    Output("upload-summary", "children"),
    Output("upload-anomaly-chart", "figure"),
    Output("upload-table", "children"),
    Output("upload-flagged-store", "data"),
    Input("upload-dataset", "contents"),
    State("upload-dataset", "filename"),
    prevent_initial_call=True,
)
def process_uploaded_dataset(contents, filename):
    if not contents:
        return (
            "No dataset uploaded.",
            [],
            "Upload a dataset to begin scanning.",
            _empty_upload_figure(),
            "No scan results yet.",
            [],
        )

    try:
        import base64

        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        df_upload = _parse_uploaded_dataset(decoded, filename)

        if df_upload.empty:
            raise ValueError("The uploaded dataset is empty.")

        df_upload = df_upload.reset_index(drop=True)
        text_col = _find_text_column(df_upload)

        if text_col is None:
            raise ValueError(
                "No text column was found. Include a column such as text, message, review, or content."
            )

        texts = df_upload[text_col].fillna("").astype(str).str.strip()
        valid_mask = texts.ne("")
        if valid_mask.sum() < 5:
            raise ValueError("At least 5 non-empty text records are required for anomaly screening.")

        # Keep the original row index so flagged records can be traced back to the upload.
        working = df_upload.loc[valid_mask].copy().reset_index()
        working.rename(columns={"index": "source_row"}, inplace=True)
        texts = working[text_col].fillna("").astype(str)

        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        X = vectorizer.fit_transform(texts)

        n_rows = len(working)
        contamination = min(0.05, max(1.0 / n_rows, 0.001))

        detector = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        predictions = detector.fit_predict(X)
        anomaly_scores = -detector.decision_function(X)
        flagged = predictions == -1

        working["anomaly_score"] = anomaly_scores
        working["potentially_suspicious"] = flagged

        truth, truth_source = _ground_truth(working)

        precision = recall = f1 = None
        if truth is not None and len(np.unique(truth)) > 1:
            precision = precision_score(truth, flagged.astype(int), zero_division=0)
            recall = recall_score(truth, flagged.astype(int), zero_division=0)
            f1 = f1_score(truth, flagged.astype(int), zero_division=0)

        flagged_df = working[flagged].sort_values(
            "anomaly_score", ascending=False
        ).copy()

        # Store only JSON-safe values needed for download.
        download_df = flagged_df.copy()
        for col in download_df.columns:
            download_df[col] = download_df[col].map(
                lambda value: value.item() if hasattr(value, "item") else value
            )
        flagged_store = download_df.to_dict("records")

        total = len(working)
        flagged_count = int(flagged.sum())
        suspicious_rate = flagged_count / total if total else 0

        if truth is not None and len(np.unique(truth)) > 1:
            metric_text = (
                f"Ground truth available ({truth_source}). "
                f"Precision {precision:.1%}, Recall {recall:.1%}, F1 {f1:.1%}."
            )
        else:
            metric_text = (
                "No verified poisoning ground truth was supplied. "
                "Results are anomaly screening signals, not proof of poisoning."
            )

        kpis = html.Div(
            [
                card("Records scanned", f"{total:,}", f"Text column: {text_col}"),
                card("Potentially suspicious", f"{flagged_count:,}", "Isolation Forest flags", WARNING),
                card("Suspicious rate", f"{suspicious_rate:.1%}", "Flagged / scanned", ACCENT_2),
                card(
                    "Ground truth F1",
                    f"{f1:.1%}" if f1 is not None else "N/A",
                    "Only when verified ground truth exists",
                    PURPLE,
                ),
            ],
            className="kpi-grid compact",
        )

        summary = html.Div(
            [
                html.Div(
                    f"File: {filename or 'uploaded dataset'} · {len(df_upload):,} uploaded rows · {len(df_upload.columns)} columns",
                    style={"fontWeight": "700", "marginBottom": "8px"},
                ),
                html.Div(metric_text, style={"color": MUTED}),
            ]
        )

        fig = go.Figure(
            go.Histogram(
                x=anomaly_scores,
                nbinsx=30,
                marker_color=ACCENT_2,
                opacity=0.85,
                hovertemplate="Anomaly score: %{x}<br>Records: %{y}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Anomaly Score Distribution",
            xaxis_title="Higher score = more anomalous",
            yaxis_title="Records",
        )
        fig_base(fig, 330)

        display_cols = ["source_row", text_col, "anomaly_score"]
        if "label" in flagged_df.columns:
            display_cols.append("label")
        if "is_poisoned" in flagged_df.columns:
            display_cols.append("is_poisoned")

        rows_for_table = flagged_df[display_cols].head(100).copy()
        headers = [
            "Source row" if c == "source_row" else
            "Anomaly score" if c == "anomaly_score" else
            str(c).replace("_", " ").title()
            for c in rows_for_table.columns
        ]

        table_children = []
        if rows_for_table.empty:
            table_children = html.Div(
                "No records were flagged by the anomaly detector.",
                style={"color": MUTED, "padding": "16px 0"},
            )
        else:
            table_header = html.Thead(
                html.Tr(
                    [html.Th(h, style={"textAlign": "left", "padding": "10px"}) for h in headers]
                )
            )
            body_rows = []
            for _, row in rows_for_table.iterrows():
                cells = []
                for col in rows_for_table.columns:
                    value = row[col]
                    if col == text_col:
                        value = str(value)
                        if len(value) > 180:
                            value = value[:177] + "..."
                    elif col == "anomaly_score":
                        value = f"{float(value):.4f}"
                    cells.append(
                        html.Td(
                            str(value),
                            style={
                                "padding": "10px",
                                "verticalAlign": "top",
                                "borderTop": f"1px solid {BORDER}",
                            },
                        )
                    )
                body_rows.append(html.Tr(cells))

            table_children = html.Div(
                [
                    html.Div(
                        f"Showing {min(len(flagged_df), 100):,} of {len(flagged_df):,} flagged records.",
                        style={"color": MUTED, "marginBottom": "10px"},
                    ),
                    html.Div(
                        html.Table(
                            [table_header, html.Tbody(body_rows)],
                            style={
                                "width": "100%",
                                "borderCollapse": "collapse",
                                "fontSize": "13px",
                            },
                        ),
                        style={"overflowX": "auto"},
                    ),
                ]
            )

        return (
            f"✅ Dataset loaded and scanned: {filename or 'uploaded dataset'}",
            kpis,
            summary,
            fig,
            table_children,
            flagged_store,
        )

    except Exception as exc:
        return (
            f"❌ Could not scan the uploaded dataset: {exc}",
            [],
            "The file could not be analysed. Check that it contains readable text records.",
            _empty_upload_figure(),
            "No scan results available.",
            [],
        )


@app.callback(
    Output("download-upload-flagged", "data"),
    Input("download-upload-flagged-btn", "n_clicks"),
    State("upload-flagged-store", "data"),
    prevent_initial_call=True,
)
def download_uploaded_flagged(_n_clicks, data):
    if not data:
        return no_update

    df = pd.DataFrame(data)
    return dcc.send_data_frame(
        df.to_csv,
        "dataguard_flagged_records.csv",
        index=False,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
