import pytest
import src.main as main_module


def test_write_output_artifacts_scoped_by_user(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = {"metric": 123, "series": [1, 2, 3]}

    created = main_module._write_output_artifacts(payload, user_id="user_42")

    # Output artifacts must be isolated per user.
    assert "analysis_results" in created
    assert "output/user_42" in created["analysis_results"].replace("\\", "/")


def test_write_output_artifacts_for_non_dict_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    created = main_module._write_output_artifacts([1, 2, 3], user_id="alpha")
    assert "analysis_results" in created
    assert created["analysis_results"].endswith("analysis_results.json")


def test_main_handles_missing_api_key(monkeypatch, capsys):
    # Force config loading to fail to simulate missing API key
    def failing_load():
        raise ValueError("CRITICAL ERROR: FRED_API_KEY not found")

    monkeypatch.setattr(main_module.ProjectConfig, "load_from_env", failing_load)

    # Run main; it should handle the error, print a message, and exit with code 1
    with pytest.raises(SystemExit) as exc_info:
        main_module.main()
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Unexpected Application Error" in captured.out or "CRITICAL ERROR" in captured.out or "Configuration error: missing FRED API key." in captured.out


def test_quick_diagnostics_handles_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    report = main_module.quick_diagnostics(user_id="u1")
    assert "quality_report.json not found" in report
    assert "dead_letter.jsonl not found" in report


def test_quick_diagnostics_reads_quality_and_dead_letter(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    quality_dir = tmp_path / "data" / "users" / "u2" / "processed" / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    (quality_dir / "quality_report.json").write_text(
        '{"files": {"f1": {"status": "failed"}}, "summary": {"missing_sources": ["fred"]}}',
        encoding="utf-8",
    )
    (quality_dir / "dead_letter.jsonl").write_text(
        '{"entity":"f1","error_type":"SchemaMismatchError","error_message":"bad"}\n',
        encoding="utf-8",
    )

    report = main_module.quick_diagnostics(user_id="u2")
    assert "Failed entities: 1" in report
    assert "Dead-letter entries: 1" in report
