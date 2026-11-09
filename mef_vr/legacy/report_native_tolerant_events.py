
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path


def intervals(flags):
    result = []
    start = None
    for i, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            result.append((start, i - 1))
            start = None
    return result


def match_events(targets, predictions, tolerance=3):
    
    if type(tolerance) is not int or tolerance < 0:
        raise ValueError('Tolerance must be a nonnegative integer')
    edges = []
    for a, b in targets:
        candidates = [j for j, (c, d) in enumerate(predictions)
                      if abs(a - c) <= tolerance and abs(b - d) <= tolerance]
        edges.append(sorted(candidates, key=lambda j: (
            abs(a - predictions[j][0]) + abs(b - predictions[j][1]), j)))
    left, right = {}, {}
    for root in range(len(targets)):
        queue = deque([root])
        seen_left = {root}
        via = {}
        end = None
        while queue and end is None:
            i = queue.popleft()
            for j in edges[i]:
                if j in via:
                    continue
                via[j] = i
                if j not in right:
                    end = j
                    break
                owner = right[j]
                if owner not in seen_left:
                    seen_left.add(owner)
                    queue.append(owner)
        while end is not None:
            i = via[end]
            old = left.get(i)
            left[i] = end
            right[end] = i
            end = old
    return sorted(left.items())


def evaluate(rows, tolerance=3):
    tp = fp = fn = short_total = short_tp = clean_videos = clean_fp = 0
    records = []
    seen = set()
    for row in rows:
        ident = row['sample_id']
        y, scores = row['frame_targets'], row['frame_probabilities']
        if ident in seen:
            raise ValueError('Duplicate sample ID')
        seen.add(ident)
        if (not y or len(y) != len(scores) or row['frame_count'] != len(y)
                or any(v not in (0, 1) for v in y)
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores)
                or not math.isfinite(row['fps']) or row['fps'] <= 0
                or row.get('all_source_frames_covered') is False):
            raise ValueError('Invalid complete native-frame labels/scores/FPS')
        gt = intervals(y)
        pred = intervals([v >= .5 for v in scores])
        matches = match_events(gt, pred, tolerance)
        matched_gt = {i for i, _ in matches}
        matched_pred = {j for _, j in matches}
        tp += len(matches)
        fp += len(pred) - len(matches)
        fn += len(gt) - len(matches)
        short_total += sum(b - a + 1 <= 3 for a, b in gt)
        short_tp += sum(gt[i][1] - gt[i][0] + 1 <= 3 for i in matched_gt)
        clean_videos += not bool(gt)
        clean_fp += not bool(gt) and bool(pred)
        records.append(dict(sample_id=ident, fps=row['fps'],
            matches=[dict(gt=list(gt[i]), prediction=list(pred[j]),
                          onset_error_frames=pred[j][0] - gt[i][0],
                          offset_error_frames=pred[j][1] - gt[i][1]) for i, j in matches],
            missed=[list(v) for i, v in enumerate(gt) if i not in matched_gt],
            false_events=[list(v) for j, v in enumerate(pred) if j not in matched_pred]))
    return dict(videos=len(rows), tp=tp, fp=fp, fn=fn,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        short_gt_events=short_total, short_tp=short_tp,
        short_recall=short_tp / short_total if short_total else None,
        clean_videos=clean_videos, clean_false_positive_videos=clean_fp,
        records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier reports')
    rows, hashes = [], {}
    for path in args.predictions:
        raw = path.read_bytes()
        hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        rows.extend(json.loads(line) for line in raw.splitlines() if line.strip())
    report = dict(protocol='native_endpoint_tolerance_v1_20260926',
        scope='Supplemental rescoring, not a new trained model or frozen-gate replacement',
        threshold=.5, tolerance_frames=3,
        convention='zero-based inclusive; offset is last abnormal source frame',
        matching='maximum-cardinality one-to-one, BOTH endpoints within tolerance; no IoU prerequisite',
        selection='No smoothing, gap filling, threshold fit, or new predictions',
        label_caveat='Existing synthetic frame labels, not independently verified perceptual visibility',
        source_sha256=hashes,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metrics={str(t): evaluate(rows, t) for t in (0, 3)})
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Input changed during evaluation')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({t: {k: v for k, v in m.items() if k != 'records'}
                      for t, m in report['metrics'].items()}, indent=2))


if __name__ == '__main__':
    main()
