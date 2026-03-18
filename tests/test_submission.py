from fastsim_tt4a.submission import build_application_packet


def test_build_application_packet_contains_key_fields() -> None:
    text = build_application_packet(
        summary={"best_val_loss": 0.0123, "best_epoch": 9, "model_type": "graph_cvae"},
        evaluation={"mse_mean": 0.01, "energy_bias_mean": 0.001, "energy_resolution_rms": 0.02},
        benchmark={"winner": "graph_cvae"},
        candidate_name="Alice",
        email_to="thiago.tomei@unesp.br",
        subject="Bolsa TT4A - 2026.01",
    )
    assert "Alice" in text
    assert "graph_cvae" in text
    assert "benchmark winner" in text
