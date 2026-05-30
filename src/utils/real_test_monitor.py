"""Per-epoch real-test monitoring for training runs."""

from __future__ import annotations

import json
from pathlib import Path


def summarize_prediction_distribution(metrics):
    counts = metrics["confusion_matrix_counts"]
    labels = metrics.get("labels", list(range(len(counts[0]))))
    predicted_counts = [sum(row[col] for row in counts) for col in range(len(labels))]
    total = sum(predicted_counts)
    if total > 0:
        predicted_shares = [value / total for value in predicted_counts]
        max_share = max(predicted_shares)
    else:
        predicted_shares = [0.0 for _ in predicted_counts]
        max_share = 0.0
    zero_classes = [label for label, count in zip(labels, predicted_counts) if count == 0]
    return {
        "predicted_class_counts": {str(label): int(count) for label, count in zip(labels, predicted_counts)},
        "predicted_class_shares": {str(label): float(share) for label, share in zip(labels, predicted_shares)},
        "predicted_classes_used": sum(1 for count in predicted_counts if count > 0),
        "max_predicted_class_share": float(max_share),
        "zero_predicted_classes": zero_classes,
    }


def compute_distribution_score(distribution, class_count=10):
    used_classes = int(distribution["predicted_classes_used"])
    max_share = float(distribution["max_predicted_class_share"])
    zero_classes = len(distribution["zero_predicted_classes"])
    class_count = max(1, int(class_count))
    return used_classes / class_count - max_share - zero_classes / class_count


def should_replace_diagnostic_checkpoint(current, candidate):
    if current is None:
        return True
    current_score = float(current["distribution_score"])
    candidate_score = float(candidate["distribution_score"])
    if candidate_score > current_score:
        return True
    if candidate_score < current_score:
        return False
    return float(candidate["real_test_loss"]) < float(current["real_test_loss"])


def should_run_real_test_monitor(epoch_number, settings):
    if not bool(settings.get("enabled", False)):
        return False
    interval = max(1, int(settings.get("interval_epochs", 1)))
    return int(epoch_number) % interval == 0


def log_classification_real_test_monitor(
    run_name,
    epoch_number,
    mean_loss,
    logits,
    labels,
    output_root,
    wandb_module,
    confusion_matrix_fn,
):
    epoch_number = int(epoch_number)
    monitor_name = f"{run_name}_realtest_epoch{epoch_number:03d}"
    save_dir = Path(output_root) / "real_test_monitor" / f"epoch_{epoch_number:03d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    confusion_matrix_fn(monitor_name, logits, labels, save_dir=str(save_dir))
    metrics_path = save_dir / f"{monitor_name}_metrics.json"
    with metrics_path.open("r") as file:
        metrics = json.load(file)
    distribution = summarize_prediction_distribution(metrics)
    distribution_score = compute_distribution_score(distribution, class_count=len(metrics.get("labels", [])) or 10)
    metrics.update(distribution)
    metrics["distribution_score"] = float(distribution_score)
    metrics["real_test_loss"] = float(mean_loss)
    metrics["epoch"] = epoch_number
    distribution_path = save_dir / f"{monitor_name}_distribution.json"
    with distribution_path.open("w") as file:
        json.dump(metrics, file, indent=2)

    payload = {
        "epoch": epoch_number,
        "test_loss": float(mean_loss),
    }
    wandb_module.log(payload)
    return metrics


def log_regression_real_test_monitor(epoch_number, mean_loss, wandb_module):
    epoch_number = int(epoch_number)
    mean_loss = float(mean_loss)
    wandb_module.log({"epoch": epoch_number, "test_loss": mean_loss})
    return {"epoch": epoch_number, "real_test_loss": mean_loss}
