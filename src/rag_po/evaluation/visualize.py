from __future__ import annotations

import csv
import html
import json
from pathlib import Path


def flatten_results(results_path: str | Path, csv_path: str | Path | None = None) -> Path:
    """Write a flat CSV summary that can be plotted in notebooks or spreadsheets."""
    results_path = Path(results_path)
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    if csv_path is None:
        csv_path = results_path.with_suffix(".csv")
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "event", "attack", "ratio", "defense", "retrieval_purity", "poison_hit",
        "avg_poison_rank", "asr", "filter_precision", "filter_recall",
        "factual_accuracy", "misinfo_propagation",
        "uncertainty_expression", "overall_trustworthiness",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row.get("retrieval_metrics", {})
            judge = row.get("judge", {})
            generation = row.get("generation_metrics", {})
            defense = row.get("defense_metrics", {})
            writer.writerow({
                "event": row.get("event"),
                "attack": row.get("attack"),
                "ratio": row.get("ratio"),
                "defense": row.get("defense"),
                "retrieval_purity": metrics.get("retrieval_purity"),
                "poison_hit": metrics.get("poison_hit"),
                "avg_poison_rank": metrics.get("avg_poison_rank"),
                "asr": generation.get("asr"),
                "filter_precision": defense.get("filter_precision"),
                "filter_recall": defense.get("filter_recall"),
                "factual_accuracy": judge.get("factual_accuracy"),
                "misinfo_propagation": judge.get("misinfo_propagation"),
                "uncertainty_expression": judge.get("uncertainty_expression"),
                "overall_trustworthiness": judge.get("overall_trustworthiness"),
            })
    return csv_path


def _num(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric(row: dict, group: str, key: str, default: float = 0.0) -> float:
    return _num((row.get(group) or {}).get(key), default)


def _bar_svg(
    rows: list[dict],
    title: str,
    group: str,
    key: str,
    max_value: float,
    suffix: str = "",
) -> str:
    width = 920
    left = 210
    row_h = 38
    height = 58 + max(1, len(rows)) * row_h
    chart_w = width - left - 40
    bars: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="0" y="22" class="chart-title">{html.escape(title)}</text>',
    ]
    for idx, row in enumerate(rows):
        y = 46 + idx * row_h
        label = f"{row.get('event')} | {row.get('attack')} | {row.get('defense')}"
        value = _metric(row, group, key)
        bar_w = 0 if max_value <= 0 else int(chart_w * max(0.0, min(value, max_value)) / max_value)
        color = "#11856f" if str(row.get("defense")) == "D_all" else "#c4542f"
        bars.append(f'<text x="0" y="{y + 16}" class="axis-label">{html.escape(label)}</text>')
        bars.append(f'<rect x="{left}" y="{y}" width="{chart_w}" height="22" rx="7" class="track"/>')
        bars.append(f'<rect x="{left}" y="{y}" width="{bar_w}" height="22" rx="7" fill="{color}"/>')
        bars.append(
            f'<text x="{left + bar_w + 8}" y="{y + 16}" class="value">'
            f'{html.escape(f"{value:g}{suffix}")}</text>'
        )
    bars.append("</svg>")
    return "\n".join(bars)


def _judge_radar_svg(row: dict) -> str:
    metrics = [
        ("factual_accuracy", _metric(row, "judge", "factual_accuracy")),
        ("misinfo_propagation", _metric(row, "judge", "misinfo_propagation")),
        ("uncertainty_expression", _metric(row, "judge", "uncertainty_expression")),
        ("overall_trustworthiness", _metric(row, "judge", "overall_trustworthiness")),
    ]
    width = 420
    cx = cy = 180
    max_r = 112
    points: list[tuple[float, float]] = []
    import math

    for idx, (_, value) in enumerate(metrics):
        angle = -math.pi / 2 + idx * 2 * math.pi / len(metrics)
        radius = max_r * max(0.0, min(value, 5.0)) / 5.0
        points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    axes = []
    for idx, (name, _) in enumerate(metrics):
        angle = -math.pi / 2 + idx * 2 * math.pi / len(metrics)
        x = cx + math.cos(angle) * max_r
        y = cy + math.sin(angle) * max_r
        tx = cx + math.cos(angle) * (max_r + 42)
        ty = cy + math.sin(angle) * (max_r + 28)
        axes.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" class="radar-axis"/>')
        axes.append(f'<text x="{tx:.1f}" y="{ty:.1f}" class="radar-label">{html.escape(name)}</text>')
    return "\n".join(
        [
            '<svg viewBox="0 0 420 360" role="img" aria-label="judge score radar">',
            '<circle cx="180" cy="180" r="112" class="radar-ring"/>',
            '<circle cx="180" cy="180" r="67" class="radar-ring"/>',
            '<circle cx="180" cy="180" r="22" class="radar-ring"/>',
            *axes,
            f'<polygon points="{poly}" class="radar-poly"/>',
            "</svg>",
        ]
    )


def _kpi_cards(rows: list[dict]) -> str:
    count = len(rows)
    avg_purity = sum(_metric(row, "retrieval_metrics", "retrieval_purity") for row in rows) / max(1, count)
    poison_hits = sum(_metric(row, "retrieval_metrics", "poison_hit") for row in rows)
    avg_trust = sum(_metric(row, "judge", "overall_trustworthiness") for row in rows) / max(1, count)
    cards = [
        ("Experiment Cells", f"{count:g}"),
        ("Avg Retrieval Purity", f"{avg_purity:.2f}"),
        ("Poison Hits", f"{poison_hits:g}"),
        ("Avg Trustworthiness", f"{avg_trust:.2f}/5"),
    ]
    return "\n".join(
        f'<section class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>'
        for label, value in cards
    )


def _table(rows: list[dict]) -> str:
    columns = [
        ("event", "Event"),
        ("attack", "Attack"),
        ("ratio", "Ratio"),
        ("defense", "Defense"),
        ("retrieval_purity", "Retrieval Purity"),
        ("poison_hit", "Poison Hit"),
        ("filter_precision", "filter_precision"),
        ("filter_recall", "filter_recall"),
        ("overall_trustworthiness", "Trust"),
    ]
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body_rows = []
    for row in rows:
        values = {
            "event": row.get("event"),
            "attack": row.get("attack"),
            "ratio": row.get("ratio"),
            "defense": row.get("defense"),
            "retrieval_purity": f"{_metric(row, 'retrieval_metrics', 'retrieval_purity'):.3g}",
            "poison_hit": f"{_metric(row, 'retrieval_metrics', 'poison_hit'):g}",
            "filter_precision": f"{_metric(row, 'defense_metrics', 'filter_precision'):.3g}",
            "filter_recall": f"{_metric(row, 'defense_metrics', 'filter_recall'):.3g}",
            "overall_trustworthiness": f"{_metric(row, 'judge', 'overall_trustworthiness'):g}",
        }
        body_rows.append("<tr>" + "".join(f"<td>{html.escape(str(values[key]))}</td>" for key, _ in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def build_html_report(results_path: str | Path, out_path: str | Path | None = None) -> Path:
    """Render a self-contained HTML/SVG report for a results.json file."""
    results_path = Path(results_path)
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda row: (str(row.get("event")), str(row.get("attack")), _num(row.get("ratio")), str(row.get("defense"))))
    if out_path is None:
        out_path = results_path.with_name("report.html")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    radar = _judge_radar_svg(rows[-1]) if rows else ""
    style = """
    :root { --ink:#1b1d1f; --muted:#687078; --paper:#faf5ec; --panel:#fffaf2; --line:#e6d8c5; }
    body { margin:0; background:linear-gradient(135deg,#f7ead7,#eef4ef 58%,#f9f6ef); color:var(--ink);
      font-family: Georgia, 'Times New Roman', serif; }
    main { max-width:1120px; margin:0 auto; padding:42px 22px 70px; }
    h1 { font-size:42px; margin:0 0 8px; letter-spacing:-1px; }
    h2 { margin:34px 0 14px; font-size:24px; }
    .sub { color:var(--muted); margin-bottom:26px; }
    .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 12px 32px #6f55331c; }
    .card span { display:block; color:var(--muted); font-size:13px; text-transform:uppercase; letter-spacing:.08em; }
    .card strong { display:block; font-size:31px; margin-top:8px; }
    .panel { background:#fffaf2cc; border:1px solid var(--line); border-radius:22px; padding:20px; margin-top:18px; overflow-x:auto; }
    svg { width:100%; height:auto; }
    .chart-title { font:bold 22px Georgia, serif; fill:#1b1d1f; }
    .axis-label,.value { font:14px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; fill:#333; }
    .track { fill:#eadfce; }
    .radar-ring,.radar-axis { fill:none; stroke:#d4c5b2; stroke-width:1.4; }
    .radar-label { font:12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; fill:#4a4f52; text-anchor:middle; }
    .radar-poly { fill:#11856f55; stroke:#11856f; stroke-width:3; }
    table { border-collapse:collapse; width:100%; font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:13px; }
    th,td { padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; }
    th { color:#7b4b2f; background:#f4eadb; }
    """
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Poisoning Results</title>
  <style>{style}</style>
</head>
<body>
<main>
  <h1>RAG Poisoning Results</h1>
  <p class="sub">Source: {html.escape(str(results_path))}</p>
  <div class="cards">{_kpi_cards(rows)}</div>
  <section class="panel">{_bar_svg(rows, "Retrieval Purity", "retrieval_metrics", "retrieval_purity", 1.0)}</section>
  <section class="panel">{_bar_svg(rows, "Poison Hit", "retrieval_metrics", "poison_hit", 1.0)}</section>
  <section class="panel">{_bar_svg(rows, "Defense filter_recall", "defense_metrics", "filter_recall", 1.0)}</section>
  <section class="panel"><h2>Judge Score Radar (last row)</h2>{radar}</section>
  <section class="panel"><h2>Metric Table</h2>{_table(rows)}</section>
</main>
</body>
</html>
"""
    out_path.write_text(content, encoding="utf-8")
    return out_path


def find_latest_results_json(root: str | Path = "experiments") -> Path:
    candidates = list(Path(root).glob("**/results.json"))
    if not candidates:
        raise FileNotFoundError(f"No results.json found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)
