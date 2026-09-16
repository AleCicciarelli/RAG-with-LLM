#!/usr/bin/env python3
"""Score a TPC-H run against regenerated ground truth and write offline reports."""

import argparse
import csv
import html
import json
import math
import re
import statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

TPCH_DIR = Path(__file__).resolve().parent


def normalize(value):
    value = re.sub(r'\s+', ' ', str(value).strip().lower())
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', value):
        return str(Decimal(value).normalize())
    return value


def witness_sets(value):
    """Preserve alternative witnesses and the membership of each joined tuple set."""
    if not isinstance(value, str):
        raise ValueError('Witness must be a string')
    value = re.sub(r'\s+', '', value).lower()
    identifier = r'[a-z][a-z0-9]*_\d+'
    group = rf'\{{{identifier}(?:,{identifier})*\}}'
    if not re.fullmatch(rf'\{{{group}(?:,{group})*\}}', value):
        raise ValueError(f'Invalid witness format: {value!r}')
    return frozenset(frozenset(group.split(',')) for group in re.findall(r'\{([^{}]+)\}', value))


def provenance_pairs(answers, why):
    if len(answers) != len(why):
        raise ValueError('Every answer must have a corresponding why string')
    return {(normalize(answer), witness) for answer, value in zip(answers, why)
            for witness in witness_sets(value)}


def set_metrics(predicted, expected):
    hits = len(predicted & expected)
    precision = hits / len(predicted) if predicted else float(not expected)
    recall = hits / len(expected) if expected else float(not predicted)
    return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def percentile(values, fraction):
    return sorted(values)[math.ceil(fraction * len(values)) - 1] if values else None


def display(value, spec='.2f', suffix=''):
    return 'N/A' if value is None else format(value, spec) + suffix


def keyed(items, label):
    if not isinstance(items, list):
        raise ValueError(f'{label} must be a list')
    mapping = {}
    for item in items:
        question = item.get('question')
        if not isinstance(question, str) or not question or question in mapping:
            raise ValueError(f'{label} contains a missing or duplicate question')
        mapping[question] = item
    return mapping


def score_question(result, expected, number):
    prediction = result.get('answer')
    request = result.get('generation_request') or {}
    valid = (isinstance(prediction, dict)
             and all(isinstance(prediction.get(key), list)
                     and all(isinstance(value, str) for value in prediction[key])
                     for key in ('answer', 'why'))
             and result.get('parsing_succeeded', request.get('parsing_succeeded', True)) is True
             and not request.get('generation_error'))
    answers = prediction['answer'] if valid else []
    why = prediction['why'] if valid else []
    expected_answers = {normalize(value) for value in expected['answer']}
    predicted_answers = {normalize(value) for value in answers}
    expected_pairs = provenance_pairs(expected['answer'], expected['why'])
    provenance_valid = valid
    try:
        predicted_pairs = provenance_pairs(answers, why) if valid else set()
    except ValueError:
        predicted_pairs = set()
        provenance_valid = False
    answer_scores = set_metrics(predicted_answers, expected_answers) if valid else (0, 0, 0)
    provenance_scores = set_metrics(predicted_pairs, expected_pairs) if provenance_valid else (0, 0, 0)
    expected_refs = set().union(*(witness for _, witness in expected_pairs)) if expected_pairs else set()
    retrieved = []
    for doc in result.get('retrieved_documents', []):
        metadata = doc['metadata']
        ref = metadata.get('tuple_id')
        if ref is None:
            ref = f"{Path(metadata['source']).stem}_{metadata['row']}"
        retrieved.append(ref.lower())
    retrieved_set = set(retrieved)
    hits = expected_refs & retrieved_set
    covered_answers = {answer for answer, witness in expected_pairs if witness <= retrieved_set}
    times = result.get('timing_seconds') or {}
    tokens = result.get('token_counts') or {}
    truncation = result.get('prompt_truncation') or {}
    return {
        'question_number': number,
        'question': result['question'],
        'question_type': expected.get('question_type', 'unknown'),
        'parsing_succeeded': bool(valid),
        'provenance_format_valid': bool(provenance_valid),
        'answer_exact': bool(valid and predicted_answers == expected_answers),
        'answer_precision': answer_scores[0], 'answer_recall': answer_scores[1], 'answer_f1': answer_scores[2],
        'why_exact': bool(provenance_valid and predicted_pairs == expected_pairs),
        'why_precision': provenance_scores[0], 'why_recall': provenance_scores[1], 'why_f1': provenance_scores[2],
        'expected_answer_count': len(expected_answers),
        'predicted_answer_count': len(predicted_answers),
        'empty_answer': not answers,
        'expected_evidence_count': len(expected_refs),
        'retrieved_evidence_count': len(hits),
        'retrieval_recall': len(hits) / len(expected_refs) if expected_refs else None,
        'retrieval_full': expected_refs <= retrieved_set if expected_refs else None,
        'answer_evidence_coverage': len(covered_answers) / len(expected_answers) if expected_answers else None,
        'first_evidence_rank': min((retrieved.index(ref) + 1 for ref in hits), default=None),
        'retrieval_seconds': times.get('retrieval'),
        'generation_seconds': times.get('generation'),
        'total_seconds': times.get('total'),
        'prompt_tokens': tokens.get('original_prompt'),
        'processed_prompt_tokens': tokens.get('ollama_prompt_processed'),
        'output_tokens': tokens.get('output'),
        'prompt_exceeds_context': truncation.get('exceeds_configured_context', request.get('prompt_exceeds_configured_context')),
        'truncation_detected': truncation.get('detected', request.get('prompt_truncation_detected')),
        'gpu_utilization': request.get('average_gpu_utilization_percent'),
        'peak_gpu_memory_mib': request.get('peak_gpu_memory_mib'),
        'generation_error': request.get('generation_error'),
    }


def summarize(rows):
    warm = [row['total_seconds'] for row in rows[1:] if row['total_seconds'] is not None]
    with_evidence = [row for row in rows if row['expected_evidence_count']]
    full = [row for row in with_evidence if row['retrieval_full']]
    incomplete = [row for row in with_evidence if not row['retrieval_full']]
    summary = {
        'question_count': len(rows),
        'answer_exact': sum(row['answer_exact'] for row in rows),
        'why_exact': sum(row['why_exact'] for row in rows),
        'parse_failures': sum(not row['parsing_succeeded'] for row in rows),
        'provenance_format_failures': sum(not row['provenance_format_valid'] for row in rows),
        'generation_failures': sum(bool(row['generation_error']) for row in rows),
        'retrieval_evaluable_questions': len(with_evidence),
        'retrieval_full': len(full),
        'retrieval_full_rate': len(full) / len(with_evidence) if with_evidence else None,
        'retrieval_any_rate': mean([bool(row['retrieved_evidence_count']) for row in with_evidence]),
        'answer_given_full_retrieval': mean([row['answer_exact'] for row in full]),
        'answer_given_incomplete_retrieval': mean([row['answer_exact'] for row in incomplete]),
        'warm_mean': mean(warm),
        'warm_median': statistics.median(warm) if warm else None,
        'warm_p95': percentile(warm, 0.95),
        'first_request_seconds': rows[0]['total_seconds'],
        'truncation_detected_count': sum(row['truncation_detected'] is True for row in rows),
        'truncation_unknown_count': sum(row['truncation_detected'] is None for row in rows),
        'prompts_exceeding_context': sum(row['prompt_exceeds_context'] is True for row in rows),
        'peak_gpu_memory_mib': max((row['peak_gpu_memory_mib'] for row in rows
                                    if row['peak_gpu_memory_mib'] is not None), default=None),
    }
    for metric in ('answer_exact', 'why_exact', 'answer_precision', 'answer_recall', 'answer_f1',
                   'why_precision', 'why_recall', 'why_f1', 'retrieval_recall', 'answer_evidence_coverage',
                   'gpu_utilization'):
        summary['mean_' + metric] = mean([row[metric] for row in rows])
    return summary


def svg_text(x, y, value, size=14):
    return f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" fill="#243447">{html.escape(str(value))}</text>'


def create_dashboard(rows, summary, timings, output_path):
    height = 420 + len(rows) * 25
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">',
           '<rect width="100%" height="100%" fill="#f7f9fc"/>',
           svg_text(35, 40, 'TPC-H RAG experiment', 26)]
    model = timings.get('generation_model', {}).get('name', 'unknown model')
    k = timings.get('retrieval', {}).get('k', '?')
    svg.append(svg_text(35, 70, f"{model} | FAISS k={k} | {len(rows)} questions"))
    for index, (label, metric) in enumerate([
        ('Answer exact match', 'mean_answer_exact'), ('Provenance exact match', 'mean_why_exact'),
        ('All witness tuples retrieved', 'retrieval_full_rate'), ('Answer evidence coverage', 'mean_answer_evidence_coverage')
    ]):
        y = 108 + index * 40
        value = summary[metric]
        svg.extend([svg_text(35, y + 16, label),
                    f'<rect x="310" y="{y}" width="600" height="22" fill="#e6ebf2"/>',
                    f'<rect x="310" y="{y}" width="{600 * (value or 0):.1f}" height="22" fill="#2a9d8f"/>',
                    svg_text(930, y + 16, display(value, '.1%'))])
    svg.append(svg_text(35, 295, f"First request: {display(summary['first_request_seconds'], suffix=' s')} | Warm median: {display(summary['warm_median'], suffix=' s')} | Parse failures: {summary['parse_failures']}"))
    svg.append(svg_text(35, 330, 'Per-question evidence recall (all witness tuples); N/A = no ground-truth evidence', 17))
    svg.append(svg_text(35, 365, 'Question / answer exact / provenance exact'))
    svg.append(svg_text(820, 365, 'Total latency / GPU utilization'))
    for index, row in enumerate(rows):
        y = 380 + index * 25
        recall = row['retrieval_recall']
        svg.extend([svg_text(35, y + 15, f"Q{row['question_number']} / {'yes' if row['answer_exact'] else 'no'} / {'yes' if row['why_exact'] else 'no'}"),
                    f'<rect x="310" y="{y}" width="400" height="17" fill="#e6ebf2"/>',
                    f'<rect x="310" y="{y}" width="{400 * (recall or 0):.1f}" height="17" fill="#457b9d"/>',
                    svg_text(725, y + 15, display(recall, '.0%')),
                    svg_text(820, y + 15, f"{display(row['total_seconds'], suffix=' s')} / {display(row['gpu_utilization'], '.1f', '%')}")])
    svg.append('</svg>')
    output_path.write_text('\n'.join(svg), encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(run_dir, ground_truth, output_dir=None, allow_partial=False):
    run_dir, ground_truth = Path(run_dir), Path(ground_truth)
    output_dir = Path(output_dir) if output_dir else run_dir / 'analysis'
    results = json.loads((run_dir / 'test_pipeline.json').read_text())
    timings = json.loads((run_dir / 'timing_metrics.json').read_text())
    truth = keyed(json.loads(ground_truth.read_text()), 'Ground truth')
    by_question = keyed(results, 'Results')
    timed = keyed(timings['questions'], 'Timings')
    if not by_question:
        raise ValueError('No completed questions to analyze')
    if set(by_question) != set(timed):
        raise ValueError('Result and timing questions do not match')
    if not set(by_question) <= set(truth):
        raise ValueError('Run contains questions absent from the TPC-H ground truth')
    missing = set(truth) - set(by_question)
    if missing and not allow_partial:
        raise ValueError(f'Run is missing {len(missing)} ground-truth questions; use --allow-partial to analyze completed questions')
    if timings.get('dataset', {}).get('name', 'tpch') != 'tpch':
        raise ValueError('Expected a TPC-H run')
    rows = [score_question(result, truth[result['question']], number)
            for number, result in enumerate(results, 1)]
    summary = summarize(rows)
    summary.update({'ground_truth': str(ground_truth.resolve()), 'run_dir': str(run_dir.resolve()),
                    'ground_truth_question_count': len(truth), 'missing_question_count': len(missing),
                    'partial_run': bool(missing), 'generation_model': timings.get('generation_model', {}),
                    'embedding': timings.get('embedding', {}), 'retrieval': timings.get('retrieval', {})})
    by_type = defaultdict(list)
    for row in rows:
        by_type[row['question_type']].append(row)
    type_metrics = []
    for kind, items in sorted(by_type.items()):
        category = summarize(items)
        type_metrics.append({'question_type': kind, **{
            key: value for key, value in category.items()
            if key.startswith(('mean_', 'answer_', 'why_', 'retrieval_'))
            or key in ('question_count', 'parse_failures', 'provenance_format_failures', 'generation_failures')
        }})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'per_question.csv', rows)
    write_csv(output_dir / 'metrics_by_type.csv', type_metrics)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    create_dashboard(rows, summary, timings, output_dir / 'dashboard.svg')
    embedding = timings.get('embedding', {})
    mismatches = ', '.join(str(row['question_number']) for row in rows if not row['answer_exact']) or 'none'
    report = f"""# TPC-H RAG results analysis

Analyzed {len(rows)} of {len(truth)} questions. Partial run: {bool(missing)}.
Model: {timings.get('generation_model', {}).get('name', 'unknown')}; retrieval k: {timings.get('retrieval', {}).get('k', 'unknown')}.

## Quality

- Answer exact match: {summary['answer_exact']}/{len(rows)} ({display(summary['mean_answer_exact'], '.1%')}).
- Provenance exact match: {summary['why_exact']}/{len(rows)} ({display(summary['mean_why_exact'], '.1%')}).
- Answer macro precision / recall / F1: {display(summary['mean_answer_precision'], '.3f')} / {display(summary['mean_answer_recall'], '.3f')} / {display(summary['mean_answer_f1'], '.3f')}.
- Provenance macro precision / recall / F1: {display(summary['mean_why_precision'], '.3f')} / {display(summary['mean_why_recall'], '.3f')} / {display(summary['mean_why_f1'], '.3f')}.
- Parse/schema failures: {summary['parse_failures']}; provenance format failures: {summary['provenance_format_failures']}; generation failures: {summary['generation_failures']} (counts may overlap).
- Answer mismatches (run order): {mismatches}.

## Retrieval

- All ground-truth witness tuples retrieved: {summary['retrieval_full']}/{summary['retrieval_evaluable_questions']} ({display(summary['retrieval_full_rate'], '.1%')}).
- Mean evidence recall: {display(summary['mean_retrieval_recall'], '.1%')}.
- Mean answer evidence coverage: {display(summary['mean_answer_evidence_coverage'], '.1%')}.
- Answer accuracy with all witness tuples retrieved: {display(summary['answer_given_full_retrieval'], '.1%')}; with incomplete evidence: {display(summary['answer_given_incomplete_retrieval'], '.1%')}.

## Performance

- First request: {display(summary['first_request_seconds'], suffix=' s')}. It may include model loading.
- Subsequent requests: mean {display(summary['warm_mean'], suffix=' s')}, median {display(summary['warm_median'], suffix=' s')}, p95 {display(summary['warm_p95'], suffix=' s')}.
- Index action: {embedding.get('index_action', 'unknown')}; build time: {display(embedding.get('index_build_wall_seconds'), suffix=' s')}; documents embedded: {embedding.get('documents_embedded', 'unknown')}.
- Embedding device: {embedding.get('device', 'unknown')}; GPU: {embedding.get('gpu_name') or 'N/A'}.
- Mean generation GPU utilization: {display(summary['mean_gpu_utilization'], suffix='%')}; peak sampled generation GPU memory: {display(summary['peak_gpu_memory_mib'], suffix=' MiB')}.
- Prompts exceeding configured context: {summary['prompts_exceeding_context']}; truncation flagged: {summary['truncation_detected_count']}; truncation check unavailable: {summary['truncation_unknown_count']}.

## Metric definitions

Answers are compared as sets after case/whitespace normalization and numeric
normalization (e.g. 905.00 equals 905). Provenance compares (answer, witness-set)
pairs, retaining alternative derivations and each witness's tuple membership;
swapped explanations or merged alternatives are not exact matches. Exact
provenance requires all ground-truth alternatives. Scores are macro averages
across questions; a valid empty prediction against empty truth scores 1.
Failed parses/generations score 0, including on empty-truth questions.

Evidence recall measures the union of all ground-truth witness tuples. Answer
evidence coverage measures the fraction of expected answers with at least one
complete witness retrieved. Empty-truth questions are excluded from retrieval
metrics: lack of retrieved evidence cannot prove that an answer does not exist.
Missing measurements and empty metric groups are N/A, not zero. GPU values are
sampled observations. Truncation flags reflect the runner's available checks.
"""
    (output_dir / 'summary.md').write_text(report, encoding='utf-8')
    return summary, output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=TPCH_DIR / 'runs' / 'tpch')
    parser.add_argument('--ground-truth', type=Path, default=TPCH_DIR / 'ground_truthTpch.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    summary, output_dir = analyze(args.run_dir, args.ground_truth, args.output_dir, args.allow_partial)
    print(f"Analyzed {summary['question_count']} questions; answer exact match: {summary['mean_answer_exact']:.1%}")
    print(f'Reports: {output_dir}')


if __name__ == '__main__':
    main()
