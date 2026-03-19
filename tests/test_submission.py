"""Submission packet generation tests."""

from pathlib import Path

from fastsim_tt4a.submission import (
    build_application_packet,
    fmt,
    safe_load_json,
    save_application_packet,
)


def test_fmt_numeric() -> None:
    """fmt formats numbers with the right precision."""
    assert fmt(0.123456, 4) == "0.1235"
    assert fmt(42, 2) == "42.00"
    assert fmt(None) == "-"
    assert fmt("text") == "-"


def test_safe_load_json_missing(tmp_path: Path) -> None:
    """safe_load_json returns empty dict for non-existent files."""
    result = safe_load_json(str(tmp_path / "missing.json"))
    assert result == {}


def test_safe_load_json_valid(tmp_path: Path) -> None:
    """safe_load_json loads a valid JSON file."""
    path = tmp_path / "data.json"
    path.write_text('{"key": "value"}', encoding="utf-8")
    result = safe_load_json(str(path))
    assert result == {"key": "value"}


def test_safe_load_json_non_dict(tmp_path: Path) -> None:
    """safe_load_json returns empty dict for non-dict JSON."""
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    result = safe_load_json(str(path))
    assert result == {}


def test_build_application_packet_contains_sections() -> None:
    """The generated packet contains all expected sections."""
    summary = {"best_val_loss": 0.001, "best_epoch": 10, "model_type": "graph_cvae"}
    evaluation = {"mse_mean": 0.002, "energy_bias_mean": -0.01, "energy_resolution_rms": 0.05}
    benchmark = {"winner": "graph_cvae"}
    packet = build_application_packet(
        summary=summary,
        evaluation=evaluation,
        benchmark=benchmark,
        candidate_name="Test User",
        email_to="test@example.com",
        subject="Bolsa Test",
    )
    assert "Resumo Tecnico" in packet
    assert "Resultados" in packet
    assert "Comandos de Reproducao" in packet
    assert "Rascunho de Email" in packet
    assert "graph_cvae" in packet
    assert "Test User" in packet


def test_save_application_packet(tmp_path: Path) -> None:
    """save_application_packet writes the packet file."""
    out_path = tmp_path / "sub" / "packet.md"
    result = save_application_packet(
        out_path=out_path,
        summary={"best_val_loss": 0.01},
        evaluation={},
        benchmark={},
        candidate_name="Jardel",
        email_to="test@unesp.br",
        subject="Bolsa",
    )
    assert result.exists()
    content = result.read_text(encoding="utf-8")
    assert "Jardel" in content
