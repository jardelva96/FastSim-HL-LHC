from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate_checkpoint
from .model import MODEL_GRAPH_CVAE, MODEL_MLP_AE, SUPPORTED_MODELS
from .train import TrainingConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark rapido entre modelos para candidatura."
    )
    parser.add_argument("--num-events", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-events", type=int, default=1200)
    parser.add_argument("--out-dir", type=str, default="artifacts/benchmark")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def markdown_table(results: list[dict[str, object]]) -> str:
    lines = [
        "| Modelo | val_loss | mse_mean | energy_bias_mean | energy_resolution_rms | params |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['model_type']} | {item['best_val_loss']:.6f} | "
            f"{item['mse_mean']:.6f} | {item['energy_bias_mean']:.6f} | "
            f"{item['energy_resolution_rms']:.6f} | {item['params']} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_order = [MODEL_GRAPH_CVAE, MODEL_MLP_AE]
    results: list[dict[str, object]] = []

    for model_type in model_order:
        if model_type not in SUPPORTED_MODELS:
            continue
        run_dir = out_dir / model_type
        config = TrainingConfig(
            num_events=args.num_events,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            beta=args.beta,
            seed=args.seed,
            out_dir=str(run_dir),
            device=args.device,
            model_type=model_type,
        )
        training = run_training(config)
        metrics = evaluate_checkpoint(
            checkpoint_path=training["checkpoint_path"],
            num_events=args.eval_events,
            seed=args.seed + 31,
            device="cpu",
        )
        summary = dict(training["summary"])
        results.append(
            {
                "model_type": model_type,
                "best_val_loss": float(summary["best_val_loss"]),
                "best_epoch": int(summary["best_epoch"]),
                "params": int(summary["params"]),
                **metrics,
                "checkpoint_path": str(training["checkpoint_path"]),
            }
        )

    results.sort(key=lambda x: float(x["mse_mean"]))
    payload = {
        "ranking_metric": "mse_mean",
        "winner": results[0]["model_type"] if results else "",
        "results": results,
    }

    json_path = out_dir / "benchmark_results.json"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    report = [
        "# Benchmark de Modelos",
        "",
        "Comparacao automatica entre modelos para apoiar a candidatura.",
        "",
        markdown_table(results),
        "",
        f"Melhor modelo por `mse_mean`: **{payload['winner']}**",
    ]
    report_path = out_dir / "benchmark_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(f"benchmark salvo em: {json_path}")
    print(f"relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()

