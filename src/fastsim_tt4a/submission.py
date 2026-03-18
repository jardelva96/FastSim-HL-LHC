from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera pacote de texto para apoiar a inscricao na selecao."
    )
    parser.add_argument("--train-summary", type=str, default="artifacts/train_summary.json")
    parser.add_argument("--eval-json", type=str, default="artifacts/eval.json")
    parser.add_argument("--benchmark-json", type=str, default="")
    parser.add_argument("--candidate-name", type=str, default="JARDEL VIEIRA ALVES")
    parser.add_argument("--email-to", type=str, default="thiago.tomei@unesp.br")
    parser.add_argument("--subject", type=str, default="Bolsa TT4A - 2026.01")
    parser.add_argument("--out-md", type=str, default="artifacts/application_packet.md")
    return parser.parse_args()


def safe_load_json(path_str: str) -> dict[str, object]:
    path = Path(path_str)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, dict):
        return data
    return {}


def fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, (float, int)):
        return f"{value:.{digits}f}"
    return "-"


def build_application_packet(
    summary: dict[str, object],
    evaluation: dict[str, object],
    benchmark: dict[str, object],
    candidate_name: str,
    email_to: str,
    subject: str,
) -> str:
    winner = ""
    if benchmark:
        winner = str(benchmark.get("winner", ""))

    lines = [
        "# Pacote de Candidatura TT-IV-A",
        "",
        f"Data de geracao: {date.today().isoformat()}",
        "",
        "## Resumo Tecnico",
        "",
        "- Projeto de simulacao rapida com aprendizado de maquina para detector simplificado.",
        "- Pipeline ponta a ponta: dados sinteticos, treino, avaliacao e dashboard interativo.",
        "- Comparacao entre modelos `graph_cvae` e `mlp_ae` para mostrar criterio experimental.",
        "",
        "## Resultados",
        "",
        f"- best_val_loss: {fmt(summary.get('best_val_loss'))}",
        f"- best_epoch: {summary.get('best_epoch', '-')}",
        f"- model_type: {summary.get('model_type', '-')}",
        f"- mse_mean: {fmt(evaluation.get('mse_mean'))}",
        f"- energy_bias_mean: {fmt(evaluation.get('energy_bias_mean'))}",
        f"- energy_resolution_rms: {fmt(evaluation.get('energy_resolution_rms'))}",
    ]
    if winner:
        lines.append(f"- benchmark winner (mse): {winner}")

    lines.extend(
        [
            "",
            "## Comandos de Reproducao",
            "",
            "```bash",
            "fastsim-train --model-type graph_cvae --epochs 20 --num-events 6000",
            "fastsim-eval --checkpoint artifacts/model.pt --out-json artifacts/eval.json",
            "fastsim-benchmark --out-dir artifacts/benchmark",
            "```",
            "",
            "## Rascunho de Email",
            "",
            f"Para: {email_to}",
            f"Assunto: {subject}",
            "",
            "Prezados,",
            "",
            f"Meu nome e {candidate_name} e gostaria de me candidatar "
            "a Bolsa TT-IV-A (2026.01).",
            "Desenvolvi um projeto tecnico de simulacao rapida com "
            "aprendizado de maquina alinhado ao tema da bolsa.",
            "Implementei pipeline completo com modelo em grafo, baseline "
            "comparativo, metricas fisicas e interface interativa.",
            "Posso compartilhar repositorio e resultados de reproducao para avaliacao.",
            "",
            "Atenciosamente,",
            f"{candidate_name}",
        ]
    )
    return "\n".join(lines)


def save_application_packet(
    out_path: Path | str,
    summary: dict[str, object],
    evaluation: dict[str, object],
    benchmark: dict[str, object],
    candidate_name: str,
    email_to: str,
    subject: str,
) -> Path:
    packet = build_application_packet(
        summary=summary,
        evaluation=evaluation,
        benchmark=benchmark,
        candidate_name=candidate_name,
        email_to=email_to,
        subject=subject,
    )
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(packet, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    summary = safe_load_json(args.train_summary)
    evaluation = safe_load_json(args.eval_json)
    benchmark = safe_load_json(args.benchmark_json) if args.benchmark_json else {}

    out_path = save_application_packet(
        out_path=args.out_md,
        summary=summary,
        evaluation=evaluation,
        benchmark=benchmark,
        candidate_name=args.candidate_name,
        email_to=args.email_to,
        subject=args.subject,
    )
    print(f"pacote gerado em: {out_path}")


if __name__ == "__main__":
    main()
