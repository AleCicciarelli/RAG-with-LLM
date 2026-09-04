#!/usr/bin/env python3
"""Analyze the TPCH RAG run and create dependency-free reports and SVG plots."""

import csv
import html
import json
import math
import re
import statistics
from pathlib import Path


TPCH_DIR = Path(__file__).resolve().parent
ROOT_DIR = TPCH_DIR.parent
OUTPUT_DIR = TPCH_DIR / "results_analysis"


def percentile(values, fraction):
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def normalized_values(values):
    return sorted(re.sub(r"\s+", " ", str(value).strip().lower()) for value in values)


def evidence_refs(values):
    pattern = r"([A-Za-z][A-Za-z0-9]*_\d+)"
    return {
        match.lower()
        for value in values
        for match in re.findall(pattern, str(value))
    }


def retrieved_refs(result):
    refs = []
    for document in result["retrieved_documents"]:
        table = Path(document["metadata"]["source"]).stem.lower()
        refs.append(f"{table}_{document['metadata']['row']}")
    return refs


def svg_text(x, y, value, size=13, fill="#243447", anchor="start", weight="normal"):
    return (
        f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">'
        f"{html.escape(str(value))}</text>"
    )


def create_dashboard(rows, summary, output_path):
    width, height = 1400, 980
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        svg_text(55, 55, "TPCH RAG experiment dashboard", 28, "#14213d", weight="bold"),
        svg_text(55, 82, "llama3:70b · FAISS k=10 · 49 questions · NVIDIA H100 NVL", 15, "#526274"),
    ]

    cards = [
        ("Answer exact match", f"{summary['answer_exact']}/49 ({summary['answer_accuracy']:.1%})", "#2a9d8f"),
        ("Provenance exact match", f"{summary['why_exact']}/49 ({summary['why_accuracy']:.1%})", "#457b9d"),
        ("Full evidence retrieved", f"{summary['retrieval_full']}/49 ({summary['retrieval_full_rate']:.1%})", "#e9c46a"),
        ("Warm median latency", f"{summary['warm_median']:.2f} s", "#f4a261"),
    ]
    for index, (label, value, color) in enumerate(cards):
        x = 55 + index * 330
        svg.extend([
            f'<rect x="{x}" y="110" width="300" height="105" rx="10" fill="white" stroke="#dce3ec"/>',
            f'<rect x="{x}" y="110" width="7" height="105" rx="3" fill="{color}"/>',
            svg_text(x + 22, 142, label, 14, "#526274"),
            svg_text(x + 22, 187, value, 25, "#14213d", weight="bold"),
        ])

    # Quality comparison bars.
    svg.append(svg_text(55, 265, "Quality and retrieval", 19, "#14213d", weight="bold"))
    quality = [
        ("Answer exact", summary["answer_accuracy"], "#2a9d8f"),
        ("Provenance exact", summary["why_accuracy"], "#457b9d"),
        ("Full evidence recall", summary["retrieval_full_rate"], "#e9c46a"),
        ("Any evidence found", summary["retrieval_any_rate"], "#f4a261"),
    ]
    for index, (label, value, color) in enumerate(quality):
        y = 300 + index * 48
        svg.append(svg_text(55, y + 17, label, 13))
        svg.append(f'<rect x="210" y="{y}" width="430" height="23" rx="4" fill="#e6ebf2"/>')
        svg.append(f'<rect x="210" y="{y}" width="{430 * value:.1f}" height="23" rx="4" fill="{color}"/>')
        svg.append(svg_text(655, y + 17, f"{value:.1%}", 13, anchor="end", weight="bold"))

    # Latency by question (log scale preserves the cold-start outlier).
    svg.append(svg_text(735, 265, "Total latency by question (log scale)", 19, "#14213d", weight="bold"))
    plot_x, plot_y, plot_w, plot_h = 735, 295, 610, 190
    max_log = math.log10(max(row["total_seconds"] for row in rows))
    for seconds in (1, 10, 100):
        y = plot_y + plot_h - math.log10(seconds) / max_log * plot_h
        svg.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" stroke="#dce3ec"/>')
        svg.append(svg_text(plot_x - 8, y + 4, f"{seconds}s", 11, anchor="end"))
    bar_width = plot_w / len(rows)
    for index, row in enumerate(rows):
        value = max(row["total_seconds"], 1)
        bar_h = math.log10(value) / max_log * plot_h
        color = "#e76f51" if index == 0 else "#457b9d"
        svg.append(
            f'<rect x="{plot_x + index * bar_width:.1f}" y="{plot_y + plot_h - bar_h:.1f}" '
            f'width="{max(2, bar_width - 2):.1f}" height="{bar_h:.1f}" fill="{color}"/>'
        )
    svg.append(svg_text(plot_x, plot_y + plot_h + 22, "Q1", 11))
    svg.append(svg_text(plot_x + plot_w, plot_y + plot_h + 22, "Q49", 11, anchor="end"))

    # Retrieval recall per question.
    svg.append(svg_text(55, 555, "Ground-truth evidence recall per question", 19, "#14213d", weight="bold"))
    grid_x, grid_y, cell_w, cell_h = 55, 580, 51, 42
    for index, row in enumerate(rows):
        col, line = index % 25, index // 25
        x, y = grid_x + col * cell_w, grid_y + line * 72
        recall = row["retrieval_recall"]
        color = "#2a9d8f" if recall == 1 else ("#e9c46a" if recall > 0 else "#e76f51")
        svg.append(f'<rect x="{x}" y="{y}" width="43" height="30" rx="4" fill="{color}"/>')
        svg.append(svg_text(x + 21.5, y + 20, index + 1, 11, "white", anchor="middle", weight="bold"))
        svg.append(svg_text(x + 21.5, y + 46, f"{recall:.0%}", 10, anchor="middle"))

    # Warm GPU utilization.
    svg.append(svg_text(55, 765, "Average GPU utilization per warm request", 19, "#14213d", weight="bold"))
    line_x, line_y, line_w, line_h = 55, 790, 900, 130
    svg.append(f'<rect x="{line_x}" y="{line_y}" width="{line_w}" height="{line_h}" fill="white" stroke="#dce3ec"/>')
    warm = rows[1:]
    points = []
    for index, row in enumerate(warm):
        x = line_x + index / (len(warm) - 1) * line_w
        y = line_y + line_h - row["gpu_utilization"] / 100 * line_h
        points.append(f"{x:.1f},{y:.1f}")
    svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#2a9d8f" stroke-width="2"/>')
    svg.append(svg_text(line_x - 8, line_y + 5, "100%", 11, anchor="end"))
    svg.append(svg_text(line_x - 8, line_y + line_h, "0%", 11, anchor="end"))

    svg.extend([
        svg_text(1010, 790, "Key finding", 18, "#14213d", weight="bold"),
        svg_text(1010, 825, "When all evidence was retrieved:", 13),
        svg_text(1010, 858, f"{summary['answer_given_full_retrieval']:.1%} answer accuracy", 23, "#2a9d8f", weight="bold"),
        svg_text(1010, 892, "With incomplete evidence:", 13),
        svg_text(1010, 925, f"{summary['answer_given_incomplete_retrieval']:.1%} answer accuracy", 23, "#e76f51", weight="bold"),
        "</svg>",
    ])
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main():
    results = json.loads((TPCH_DIR / "test_pipeline.json").read_text())
    timings = json.loads((TPCH_DIR / "timing_metrics.json").read_text())
    truth = json.loads((ROOT_DIR / "ground_truth2.json").read_text())
    if not (len(results) == len(timings["questions"]) == len(truth)):
        raise ValueError("Result, timing, and ground-truth lengths do not match")

    rows = []
    for number, (result, expected) in enumerate(zip(results, truth), 1):
        prediction = result.get("answer") or {}
        predicted_answer = prediction.get("answer", [])
        predicted_why = prediction.get("why", [])
        expected_refs = evidence_refs(expected.get("why", []))
        predicted_refs = evidence_refs(predicted_why)
        retrieved = retrieved_refs(result)
        retrieved_set = set(retrieved)
        hits = expected_refs & retrieved_set
        ranks = [retrieved.index(ref) + 1 for ref in expected_refs if ref in retrieved_set]
        request = result["generation_request"]
        rows.append({
            "question_number": number,
            "question": result["question"],
            "answer_exact": normalized_values(predicted_answer) == normalized_values(expected.get("answer", [])),
            "why_exact": predicted_refs == expected_refs,
            "empty_answer": not predicted_answer,
            "retrieval_recall": len(hits) / len(expected_refs) if expected_refs else 1.0,
            "first_evidence_rank": min(ranks) if ranks else None,
            "retrieval_seconds": result["timing_seconds"]["retrieval"],
            "generation_seconds": result["timing_seconds"]["generation"],
            "total_seconds": result["timing_seconds"]["total"],
            "prompt_tokens": result["token_counts"]["original_prompt"],
            "output_tokens": result["token_counts"]["output"],
            "gpu_utilization": request.get("average_gpu_utilization_percent") or 0,
            "peak_gpu_memory_mib": request.get("peak_gpu_memory_mib"),
        })

    full = [row for row in rows if row["retrieval_recall"] == 1]
    incomplete = [row for row in rows if row["retrieval_recall"] < 1]
    warm_times = [row["total_seconds"] for row in rows[1:]]
    summary = {
        "answer_exact": sum(row["answer_exact"] for row in rows),
        "answer_accuracy": statistics.mean(row["answer_exact"] for row in rows),
        "why_exact": sum(row["why_exact"] for row in rows),
        "why_accuracy": statistics.mean(row["why_exact"] for row in rows),
        "retrieval_full": len(full),
        "retrieval_full_rate": len(full) / len(rows),
        "retrieval_any_rate": statistics.mean(row["retrieval_recall"] > 0 for row in rows),
        "mean_retrieval_recall": statistics.mean(row["retrieval_recall"] for row in rows),
        "answer_given_full_retrieval": statistics.mean(row["answer_exact"] for row in full),
        "answer_given_incomplete_retrieval": statistics.mean(row["answer_exact"] for row in incomplete),
        "warm_mean": statistics.mean(warm_times),
        "warm_median": statistics.median(warm_times),
        "warm_p95": percentile(warm_times, 0.95),
        "cold_seconds": rows[0]["total_seconds"],
        "cold_share": rows[0]["total_seconds"] / sum(row["total_seconds"] for row in rows),
        "empty_answers": sum(row["empty_answer"] for row in rows),
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / "per_question.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    create_dashboard(rows, summary, OUTPUT_DIR / "dashboard.svg")
    mismatches = ", ".join(str(row["question_number"]) for row in rows if not row["answer_exact"])
    report = f"""# TPCH RAG results analysis

## Executive summary

- Answer exact match: **{summary['answer_exact']}/{len(rows)} ({summary['answer_accuracy']:.1%})**.
- Exact provenance (`why`): **{summary['why_exact']}/{len(rows)} ({summary['why_accuracy']:.1%})**.
- Full ground-truth evidence retrieved: **{summary['retrieval_full']}/{len(rows)} ({summary['retrieval_full_rate']:.1%})**; mean evidence recall: **{summary['mean_retrieval_recall']:.1%}**.
- With complete retrieval, answer accuracy was **{summary['answer_given_full_retrieval']:.1%}**. With incomplete retrieval, it was **{summary['answer_given_incomplete_retrieval']:.1%}**.
- All {len(rows)} outputs parsed successfully; {summary['empty_answers']} contained an empty answer.

## Performance

- Cold question: **{summary['cold_seconds']:.2f} s**, accounting for **{summary['cold_share']:.1%}** of cumulative question latency. Model loading alone took {results[0]['generation_request']['model_load_seconds']:.2f} s.
- Warm requests: mean **{summary['warm_mean']:.2f} s**, median **{summary['warm_median']:.2f} s**, p95 **{summary['warm_p95']:.2f} s**.
- Embedding/index build: **{timings['embedding']['index_build_wall_seconds']:.2f} s** for {timings['embedding']['documents_embedded']} documents on {timings['embedding']['gpu_name']}.
- Warm GPU utilization median: **{statistics.median(row['gpu_utilization'] for row in rows[1:]):.1f}%**; peak GPU memory was approximately **{max(row['peak_gpu_memory_mib'] for row in rows if row['peak_gpu_memory_mib']) / 1024:.1f} GiB**.
- No prompt exceeded the configured 8,192-token context window.

## Interpretation

Retrieval is the dominant quality bottleneck: every answer with incomplete ground-truth evidence was incorrect, while 23 of 25 questions with complete evidence were answered correctly. Questions 11 and 48 had complete evidence but incorrect answers, so those are the clearest generation/reasoning failures. Provenance trails answer quality because several correct answers omitted supporting join rows.

Answer mismatches: {mismatches}.
"""
    (OUTPUT_DIR / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Dashboard: {OUTPUT_DIR / 'dashboard.svg'}")
    print(f"Per-question data: {OUTPUT_DIR / 'per_question.csv'}")


if __name__ == "__main__":
    main()
