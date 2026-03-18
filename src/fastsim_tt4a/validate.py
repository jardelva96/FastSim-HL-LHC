from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import evaluate_physics_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa validacao fisica detalhada de um checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-events", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--n-layers", type=int, default=0)
    parser.add_argument("--cells-per-layer", type=int, default=0)
    parser.add_argument("--out-json", type=str, default="artifacts/validation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_physics_report(
        checkpoint_path=args.checkpoint,
        num_events=args.num_events,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        n_layers_override=args.n_layers,
        cells_override=args.cells_per_layer,
    )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print(json.dumps(report["global_metrics"], indent=2))
    print(f"validacao salva em: {out_path}")


if __name__ == "__main__":
    main()

