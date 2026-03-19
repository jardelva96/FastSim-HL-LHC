from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from fastsim_tt4a.analysis import evaluate_physics_report, generate_conditioned_samples
    from fastsim_tt4a.data import (
        DetectorGeometry,
        SyntheticShowerDataset,
        denormalize_energy,
        denormalize_time,
    )
    from fastsim_tt4a.evaluate import build_model_from_checkpoint, load_checkpoint
    from fastsim_tt4a.model import SUPPORTED_MODELS, forward_model
    from fastsim_tt4a.submission import safe_load_json, save_application_packet
    from fastsim_tt4a.train import TrainingConfig, run_training
else:
    from .analysis import evaluate_physics_report, generate_conditioned_samples
    from .data import DetectorGeometry, SyntheticShowerDataset, denormalize_energy, denormalize_time
    from .evaluate import build_model_from_checkpoint, load_checkpoint
    from .model import SUPPORTED_MODELS, forward_model
    from .submission import safe_load_json, save_application_packet
    from .train import TrainingConfig, run_training


def _display_map(st: Any, title: str, values: list[list[float]]) -> None:
    st.markdown(title)
    rounded = [[round(cell, 4) for cell in row] for row in values]
    st.dataframe(rounded, use_container_width=True)


def _save_json(path_str: str, payload: dict[str, Any]) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return path


def _render_train_tab(st: Any) -> None:
    st.subheader("Treino rapido")
    with st.form("train_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            num_events = st.number_input("Eventos", min_value=200, value=3000, step=100)
            epochs = st.number_input("Epocas", min_value=3, value=12, step=1)
            batch_size = st.number_input("Batch size", min_value=8, value=64, step=8)
            lr = st.number_input("Learning rate", min_value=1e-5, value=1e-3, step=1e-4)
        with col2:
            hidden_dim = st.number_input("Hidden dim", min_value=16, value=96, step=16)
            latent_dim = st.number_input("Latent dim", min_value=4, value=16, step=2)
            beta = st.number_input("Beta (KL)", min_value=1e-5, value=1e-3, step=1e-4)
            patience = st.number_input("Early stopping patience", min_value=2, value=6, step=1)
        with col3:
            n_layers = st.number_input("Camadas", min_value=3, value=6, step=1)
            cells_per_layer = st.number_input(
                "Celulas por camada",
                min_value=8,
                value=16,
                step=2,
            )
            model_type = st.selectbox("Modelo", options=list(SUPPORTED_MODELS), index=0)
            seed = st.number_input("Seed", min_value=0, value=7, step=1)
            out_dir = st.text_input("Diretorio de artefatos", value="artifacts")

        run_button = st.form_submit_button("Treinar")

    if not run_button:
        return

    progress = st.progress(0)
    status = st.empty()

    def on_progress(epoch: int, total: int, metrics: dict[str, float]) -> None:
        progress.progress(int((epoch / total) * 100))
        status.text(
            f"epoca {epoch}/{total} | train={metrics['train_loss']:.5f} | "
            f"val={metrics['val_loss']:.5f}"
        )

    config = TrainingConfig(
        num_events=int(num_events),
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        beta=float(beta),
        seed=int(seed),
        out_dir=out_dir,
        patience=int(patience),
        n_layers=int(n_layers),
        cells_per_layer=int(cells_per_layer),
        model_type=str(model_type),
    )
    with st.spinner("Treinando modelo..."):
        result = run_training(config=config, progress_callback=on_progress)
    st.session_state["last_training"] = result
    st.success("Treino concluido.")
    st.json(result["summary"])

    history = result["history"]
    st.line_chart(
        {
            "train_loss": [h["train_loss"] for h in history],
            "val_loss": [h["val_loss"] for h in history],
            "lr": [h["lr"] for h in history],
        }
    )


def _render_history_tab(st: Any) -> None:
    st.subheader("Visualizar historico salvo")
    history_path = st.text_input("Arquivo history.json", value="artifacts/history.json")
    if not st.button("Carregar historico"):
        return

    path = Path(history_path)
    if not path.exists():
        st.error(f"Arquivo nao encontrado: {path}")
        return
    with path.open("r", encoding="utf-8") as fp:
        history = json.load(fp)
    if not history:
        st.warning("Historico vazio.")
        return

    st.line_chart(
        {
            "train_loss": [h["train_loss"] for h in history],
            "val_loss": [h["val_loss"] for h in history],
            "lr": [h.get("lr", 0.0) for h in history],
        }
    )
    st.dataframe(history, use_container_width=True)


def _render_event_tab(st: Any) -> None:
    st.subheader("Inspecionar evento e reconstruir com checkpoint")
    col1, col2 = st.columns(2)
    with col1:
        num_events = st.number_input(
            "Eventos sinteticos (amostra)",
            min_value=10,
            value=200,
            step=10,
        )
        event_idx = st.number_input("Indice do evento", min_value=0, value=0, step=1)
        seed = st.number_input("Seed de amostra", min_value=0, value=13, step=1)
    with col2:
        n_layers = st.number_input("Camadas (evento)", min_value=3, value=6, step=1)
        cells_per_layer = st.number_input(
            "Celulas/camada (evento)",
            min_value=8,
            value=16,
            step=2,
        )
        checkpoint_path = st.text_input("Checkpoint (opcional)", value="artifacts/model.pt")

    if not st.button("Mostrar evento"):
        return

    geometry = DetectorGeometry(
        n_layers=int(n_layers),
        cells_per_layer=int(cells_per_layer),
    )
    dataset = SyntheticShowerDataset(
        num_events=int(num_events),
        seed=int(seed),
        geometry=geometry,
    )
    idx = min(int(event_idx), len(dataset) - 1)
    sample = dataset[idx]

    cond = sample["cond"].tolist()
    st.write(
        {
            "beam_energy_norm": round(cond[0], 4),
            "pileup_norm": round(cond[1], 4),
        }
    )

    true_energy = denormalize_energy(sample["target"][..., 0]).reshape(
        geometry.n_layers,
        geometry.cells_per_layer,
    )
    true_time = denormalize_time(sample["target"][..., 1]).reshape(
        geometry.n_layers,
        geometry.cells_per_layer,
    )
    left, right = st.columns(2)
    with left:
        st.markdown("Energia verdadeira por celula")
        st.dataframe(true_energy.numpy(), use_container_width=True)
    with right:
        st.markdown("Tempo verdadeiro por celula")
        st.dataframe(true_time.numpy(), use_container_width=True)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        st.info("Checkpoint nao encontrado. Mostrando apenas o evento verdadeiro.")
        return

    checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
    model, model_config = build_model_from_checkpoint(
        checkpoint=checkpoint,
        n_nodes=geometry.n_nodes,
        device="cpu",
    )
    with torch.no_grad():
        recon, _, _ = forward_model(
            model=model,
            model_type=str(model_config["model_type"]),
            coords=sample["coords"].unsqueeze(0),
            target=sample["target"].unsqueeze(0),
            cond=sample["cond"].unsqueeze(0),
            adj=sample["adj"].unsqueeze(0),
        )
    recon = recon[0]
    pred_energy = denormalize_energy(recon[..., 0]).reshape(
        geometry.n_layers,
        geometry.cells_per_layer,
    )
    pred_time = denormalize_time(recon[..., 1]).reshape(
        geometry.n_layers,
        geometry.cells_per_layer,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("Energia reconstruida")
        st.dataframe(pred_energy.numpy(), use_container_width=True)
    with right:
        st.markdown("Tempo reconstruido")
        st.dataframe(pred_time.numpy(), use_container_width=True)


def _render_validation_tab(st: Any) -> None:
    st.subheader("Validacao fisica detalhada")
    col1, col2, col3 = st.columns(3)
    with col1:
        checkpoint_path = st.text_input("Checkpoint", value="artifacts/model.pt")
        num_events = st.number_input("Eventos validacao", min_value=100, value=1200, step=100)
    with col2:
        batch_size = st.number_input("Batch size", min_value=16, value=128, step=16)
        seed = st.number_input("Seed validacao", min_value=0, value=99, step=1)
    with col3:
        out_json = st.text_input("Salvar JSON em", value="artifacts/validation.json")
        save_flag = st.checkbox("Salvar resultado em arquivo", value=True)

    if st.button("Rodar validacao"):
        if not Path(checkpoint_path).exists():
            st.error(
                "Checkpoint nao encontrado. Treine um modelo primeiro na aba **Treino**."
            )
        else:
            with st.spinner("Executando validacao fisica..."):
                report = evaluate_physics_report(
                    checkpoint_path=checkpoint_path,
                    num_events=int(num_events),
                    batch_size=int(batch_size),
                    seed=int(seed),
                    device="cpu",
                )
            st.session_state["last_validation"] = report
            st.success("Validacao concluida.")
            if save_flag:
                path = _save_json(out_json, report)
                st.caption(f"resultado salvo em: {path}")

    report = st.session_state.get("last_validation")
    if not report:
        st.info("Rode a validacao para visualizar metricas e perfis.")
        return

    global_metrics = dict(report["global_metrics"])
    closure = dict(report["closure"])

    c1, c2, c3 = st.columns(3)
    c1.metric("mse_mean", f"{global_metrics['mse_mean']:.6f}")
    c2.metric("energy_bias_mean", f"{global_metrics['energy_bias_mean']:.6f}")
    c3.metric("energy_resolution_rms", f"{global_metrics['energy_resolution_rms']:.6f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("closure_mean", f"{closure['closure_mean']:.6f}")
    c2.metric("closure_std", f"{closure['closure_std']:.6f}")
    c3.metric("closure_p95_abs_dev", f"{closure['closure_p95_abs_dev']:.6f}")

    layer_profile = list(report["layer_profile"])
    st.markdown("Perfil longitudinal de energia")
    st.line_chart(
        {
            "true_energy_mean": [x["true_energy_mean"] for x in layer_profile],
            "pred_energy_mean": [x["pred_energy_mean"] for x in layer_profile],
            "abs_error_mean": [x["abs_error_mean"] for x in layer_profile],
        }
    )
    st.dataframe(layer_profile, use_container_width=True)

    pileup_profile = list(report["pileup_profile"])
    st.markdown("Perfil por faixa de pileup")
    st.dataframe(pileup_profile, use_container_width=True)
    st.line_chart(
        {
            "bias": [x["bias"] for x in pileup_profile],
            "resolution_rms": [x["resolution_rms"] for x in pileup_profile],
            "abs_bias_p90": [x["abs_bias_p90"] for x in pileup_profile],
        }
    )

    left, right = st.columns(2)
    with left:
        _display_map(st, "Mapa medio de energia (true)", report["true_energy_map_mean"])
        _display_map(st, "Mapa medio de energia (pred)", report["pred_energy_map_mean"])
    with right:
        _display_map(st, "Erro absoluto medio de energia", report["abs_energy_error_map_mean"])
        _display_map(st, "Mapa medio de tempo (pred)", report["pred_time_map_mean"])


def _render_generation_tab(st: Any) -> None:
    st.subheader("Geracao condicionada de eventos")
    col1, col2, col3 = st.columns(3)
    with col1:
        checkpoint_path = st.text_input("Checkpoint geracao", value="artifacts/model.pt")
        num_samples = st.number_input("N amostras", min_value=2, value=8, step=1)
    with col2:
        beam_energy = st.slider(
            "Energia do feixe (GeV)",
            min_value=30.0,
            max_value=300.0,
            value=120.0,
        )
        pileup = st.slider("Pileup", min_value=0.0, max_value=200.0, value=60.0)
    with col3:
        seed = st.number_input("Seed geracao", min_value=0, value=23, step=1)
        out_json = st.text_input("Salvar geracao em", value="artifacts/generated_samples.json")
        save_flag = st.checkbox("Salvar geracao", value=True)

    if st.button("Gerar amostras"):
        if not Path(checkpoint_path).exists():
            st.error(
                "Checkpoint nao encontrado. Treine um modelo primeiro na aba **Treino**."
            )
        else:
            with st.spinner("Gerando eventos condicionados..."):
                report = generate_conditioned_samples(
                    checkpoint_path=checkpoint_path,
                    beam_energy=float(beam_energy),
                    pileup=float(pileup),
                    num_samples=int(num_samples),
                    seed=int(seed),
                    device="cpu",
                )
            st.session_state["last_generated"] = report
            st.success("Geracao concluida.")
            if save_flag:
                path = _save_json(out_json, report)
                st.caption(f"resultado salvo em: {path}")

    generated = st.session_state.get("last_generated")
    if not generated:
        st.info("Gere amostras para visualizar mapas medio e dispersao.")
        return

    st.write(
        {
            "model_type": generated["model_type"],
            "beam_energy": generated["beam_energy"],
            "pileup": generated["pileup"],
            "n_samples": generated["n_samples"],
        }
    )
    left, right = st.columns(2)
    with left:
        _display_map(st, "Energia media gerada", generated["energy_mean_map"])
        _display_map(st, "Energia desvio padrao", generated["energy_std_map"])
    with right:
        _display_map(st, "Tempo medio gerado", generated["time_mean_map"])
        _display_map(st, "Tempo desvio padrao", generated["time_std_map"])


def _render_benchmark_tab(st: Any) -> None:
    st.subheader("Benchmark comparativo")
    benchmark_path = st.text_input(
        "Arquivo benchmark_results.json",
        value="artifacts/benchmark/benchmark_results.json",
    )
    if st.button("Carregar benchmark"):
        payload = safe_load_json(benchmark_path)
        st.session_state["last_benchmark"] = payload

    payload = st.session_state.get("last_benchmark")
    if not payload:
        st.info("Carregue um arquivo de benchmark para comparar os modelos.")
        return

    results = payload.get("results", [])
    if not isinstance(results, list) or not results:
        st.warning("Benchmark vazio ou invalido.")
        return

    winner = str(payload.get("winner", "-"))
    st.metric("Melhor modelo (mse_mean)", winner)
    st.dataframe(results, use_container_width=True)
    st.line_chart(
        {
            "mse_mean": [row["mse_mean"] for row in results],
            "energy_resolution_rms": [row["energy_resolution_rms"] for row in results],
            "energy_abs_bias_p95": [row["energy_abs_bias_p95"] for row in results],
        }
    )


def _render_selection_tab(st: Any) -> None:
    st.subheader("Checklist de selecao e pacote final")
    st.markdown(
        """
        - Mostre comparacao entre `graph_cvae` e `mlp_ae`.
        - Reporte `mse_mean`, `energy_bias_mean`, `energy_resolution_rms`.
        - Traga validacao fisica por camada e por faixa de pileup.
        - Envie comandos de reproducao + link do repositorio.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        train_summary = st.text_input("train_summary.json", value="artifacts/train_summary.json")
        eval_json = st.text_input("eval.json", value="artifacts/eval.json")
        benchmark_json = st.text_input(
            "benchmark_results.json",
            value="artifacts/benchmark/benchmark_results.json",
        )
    with col2:
        candidate_name = st.text_input("Seu nome", value="Seu Nome")
        email_to = st.text_input("Email destino", value="thiago.tomei@unesp.br")
        subject = st.text_input("Assunto", value="Bolsa TT4A - 2026.01")
        out_md = st.text_input("Salvar pacote em", value="artifacts/application_packet.md")

    if st.button("Gerar pacote da candidatura"):
        summary = safe_load_json(train_summary)
        evaluation = safe_load_json(eval_json)
        benchmark = safe_load_json(benchmark_json)
        out_path = save_application_packet(
            out_path=out_md,
            summary=summary,
            evaluation=evaluation,
            benchmark=benchmark,
            candidate_name=candidate_name,
            email_to=email_to,
            subject=subject,
        )
        st.success(f"Pacote gerado em: {out_path}")
        preview = Path(out_path).read_text(encoding="utf-8")
        st.code(preview, language="markdown")


def render_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="FastSim TT-IV-A", layout="wide")
    st.title("FastSim TT-IV-A Dashboard")
    st.caption("Treino, benchmark, validacao fisica e pacote de candidatura.")

    tabs = st.tabs(
        [
            "Treino",
            "Historico",
            "Evento",
            "Validacao Fisica",
            "Geracao",
            "Benchmark",
            "Selecao",
        ]
    )
    with tabs[0]:
        _render_train_tab(st)
    with tabs[1]:
        _render_history_tab(st)
    with tabs[2]:
        _render_event_tab(st)
    with tabs[3]:
        _render_validation_tab(st)
    with tabs[4]:
        _render_generation_tab(st)
    with tabs[5]:
        _render_benchmark_tab(st)
    with tabs[6]:
        _render_selection_tab(st)


def main() -> None:
    try:
        from streamlit.web import cli as st_cli
    except ImportError as exc:
        raise RuntimeError("Streamlit nao instalado. Use: pip install -e .[ui]") from exc

    script_path = str(Path(__file__).resolve())
    sys.argv = ["streamlit", "run", script_path]
    raise SystemExit(st_cli.main())


if __name__ == "__main__":
    render_app()
