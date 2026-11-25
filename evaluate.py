import argparse
import bisect
import csv
import json
import math
from pathlib import Path

def _as_floats(values, name):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite values and cannot be empty")
    return values

def validate_predictions(y, p1, uncertainty):
    y = _as_floats(y, "y")
    p1 = _as_floats(p1, "p1")
    uncertainty = _as_floats(uncertainty, "u")
    if len(y) != len(p1) or len(y) != len(uncertainty):
        raise ValueError("y, p1 and u must have the same length")
    if not all(value in (0.0, 1.0) for value in y):
        raise ValueError("Labels must be 0 or 1")
    if not all(0 <= value <= 1 for value in p1 + uncertainty):
        raise ValueError("p1 and u must be within [0, 1]")
    return [int(value) for value in y], p1, uncertainty

def binary_auroc(y, score):
    if len(y) != len(score):
        raise ValueError("Labels and scores must have the same length")
    positives = sum(y)
    negatives = len(y) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(score, y), key=lambda pair: pair[0])
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[start:end])
        start = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)

def average_precision(y, score):
    if len(y) != len(score):
        raise ValueError("Labels and scores must have the same length")
    positives = sum(y)
    if not positives:
        return 0.0
    ordered = sorted(zip(score, y), key=lambda pair: -pair[0])
    true_positives = 0
    result = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        group_positives = sum(label for _, label in ordered[start:end])
        true_positives += group_positives
        result += (group_positives / positives) * (true_positives / end)
        start = end
    return result

def linear_quantile(values, fraction):
    if not values or not 0 <= fraction <= 1:
        raise ValueError("Quantile requires nonempty values and a fraction in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (position - left) * (ordered[right] - ordered[left])

def probability_stratified_score(y, p1, uncertainty, threshold=0.5, bins=10, min_errors=10):
    if bins < 1 or min_errors < 0:
        raise ValueError("bins must be positive and min_errors cannot be negative")
    wrong = [int((probability >= threshold) != bool(label)) for label, probability in zip(y, p1)]
    edges = sorted(set(linear_quantile(p1, i / bins) for i in range(bins + 1)))
    result = {"score": None, "eligible": False, "errors": sum(wrong), "strata": max(0, len(edges) - 1), "valid_strata": 0, "valid_records": 0, "details": []}
    if len(edges) < 2:
        return result
    groups = [[] for _ in range(len(edges) - 1)]
    for index, probability in enumerate(p1):
        groups[bisect.bisect_left(edges[1:-1], probability)].append(index)
    numerator = 0.0
    for index, members in enumerate(groups):
        value = binary_auroc([wrong[i] for i in members], [uncertainty[i] for i in members])
        row = {"stratum": index + 1, "lower": edges[index], "upper": edges[index + 1], "n": len(members), "errors": sum(wrong[i] for i in members), "error_detection_auroc": value}
        result["details"].append(row)
        if value is not None:
            result["valid_strata"] += 1
            result["valid_records"] += len(members)
            numerator += len(members) * value
    if sum(wrong) >= min_errors and result["valid_records"]:
        result["score"] = numerator / result["valid_records"]
        result["eligible"] = True
    return result

def compute_metrics(y, p1, uncertainty, threshold=0.5, bins=10, min_errors=10):
    y, p1, uncertainty = validate_predictions(y, p1, uncertainty)
    if not 0 < threshold < 1:
        raise ValueError("Classification threshold must be within (0, 1)")
    predicted = [int(value >= threshold) for value in p1]
    wrong = [int(prediction != label) for prediction, label in zip(predicted, y)]
    count = len(y)
    order = sorted(range(count), key=lambda i: uncertainty[i])
    cumulative_errors = 0
    risks = []
    for rank, index in enumerate(order, start=1):
        cumulative_errors += wrong[index]
        risks.append(cumulative_errors / rank)
    retained = max(1, round(count / 4))
    false_negatives = [i for i in range(count) if y[i] == 1 and predicted[i] == 0]
    median = linear_quantile(uncertainty, 0.5)
    return {
        "n": count,
        "positives": sum(y),
        "errors": sum(wrong),
        "classification_threshold": threshold,
        "prediction_auroc": binary_auroc(y, p1),
        "prediction_auprc_average_precision": average_precision(y, p1),
        "accuracy": 1.0 - sum(wrong) / count,
        "error_detection_auroc": binary_auroc(wrong, uncertainty),
        "probability_stratified": probability_stratified_score(y, p1, uncertainty, threshold, bins, min_errors),
        "aurc": sum(risks) / count,
        "most_certain_quarter_n": retained,
        "accuracy_most_certain_quarter": 1.0 - sum(wrong[i] for i in order[:retained]) / retained,
        "missed_interactions": len(false_negatives),
        "missed_interaction_flag_rate": sum(uncertainty[i] > median for i in false_negatives) / len(false_negatives) if false_negatives else None,
        "uncertainty_median": median,
    }

def load_predictions(path, y_column="y", p_column="p1", u_column="u", fold=None):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        import numpy as np
        with np.load(path, allow_pickle=False) as values:
            missing = {y_column, p_column, u_column} - set(values.files)
            if missing:
                raise ValueError(f"Missing NPZ arrays: {sorted(missing)}")
            return tuple(values[key].reshape(-1).tolist() for key in (y_column, p_column, u_column))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = {y_column, p_column, u_column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        rows = [row for row in reader if fold is None or row.get("fold") == fold]
    return tuple([row[key] for row in rows] for key in (y_column, p_column, u_column))

def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate one DSE-DTI prediction set without loading a model or comparator.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--y-column", default="y")
    parser.add_argument("--p-column", default="p1")
    parser.add_argument("--u-column", default="u")
    parser.add_argument("--fold")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--min-errors", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.resolve() == Path(args.predictions).resolve():
        parser.error("Metric output cannot overwrite the prediction input")
    if output.exists() and not args.overwrite:
        parser.error("Output already exists. Use another filename or --overwrite.")
    try:
        arrays = load_predictions(args.predictions, args.y_column, args.p_column, args.u_column, args.fold)
        result = compute_metrics(*arrays, args.threshold, args.bins, args.min_errors)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    result["prediction_file"] = str(Path(args.predictions))
    result["scope"] = "One supplied prediction set. No paired model comparison, significance test, joint-valid-run filtering or dataset deduplication is performed. Undefined metrics are null."
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
