import src.main as main_module


def test_main_handles_missing_api_key(monkeypatch, capsys):
    # Force config loading to fail to simulate missing API key
    def failing_load():
        raise ValueError("CRITICAL ERROR: FRED_API_KEY not found")

    monkeypatch.setattr(
        "src.Fetchers.ProjectConfig.ProjectConfig.load_from_env", failing_load
    )

    # Run main; it should handle the error and print a message
    main_module.main()
    captured = capsys.readouterr()
    assert (
        "Unexpected Application Error" in captured.out
        or "CRITICAL ERROR" in captured.out
    )
