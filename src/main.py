# Copyright (c) 2026 EvangelGK. All Rights Reserved.
import gzip
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from exceptions.FetchersExceptions import FetcherError
from exceptions.MedallionExceptions import DataPipelineError
from Fetchers.Factory import DataFactory
from Fetchers.ProjectConfig import ProjectConfig, RunMode
from logger.Catalog import catalog
from logger.Messages.DirectionsMess import (
    LIVE_ERROR_API_KEY,
    LIVE_ERROR_NETWORK,
    LIVE_STEP_0_WELCOME,
    LIVE_STEP_1_PREREQUISITES_CHECK,
    LIVE_STEP_2_CONFIG_LOADING,
    LIVE_STEP_3_DATA_FETCHING_START,
    LIVE_STEP_8_COMPLETION,
)
from logger.Messages.MainMess import (
    APPLICATION_TITLE,
    MAIN_COMPLETION,
    MAIN_CONFIG_LOADED,
    MAIN_PIPELINE_START,
    MAIN_PIPELINE_SUCCESS,
    MAIN_RESULTS_SUMMARY,
    MAIN_START,
    QUICK_START,
)
from Medallion import MedallionPipeline
from ai_agent import generate_ai_pipeline_brief

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _user_error_message(error: Exception) -> str:
    msg = str(error)
    if "FRED_API_KEY" in msg:
        return "Configuration error: missing FRED API key."
    if "Schema drift" in msg:
        return "Data contract error: source schema changed unexpectedly."
    if "hard stop" in msg or "guardrails" in msg:
        return "Data quality guardrail triggered. Review quality_report and dead-letter logs."
    if "No financial data files" in msg:
        return "No usable data reached Gold layer. Check Bronze/Silver outputs."
    return msg


def _hash_payload(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _hash_file_if_exists(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "unavailable"
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _to_serializable(value: Any) -> Any:
    """Convert analysis objects to JSON-friendly payloads."""
    try:
        import numpy as np
        import pandas as pd

        max_df_rows = 300
        max_series_rows = 600
        max_array_elements = 5000

        if isinstance(value, pd.DataFrame):
            numeric_cols = value.select_dtypes(include=["number"]).columns.tolist()
            numeric_summary = {}
            for col in numeric_cols[:20]:
                series = pd.to_numeric(value[col], errors="coerce").dropna()
                if not series.empty:
                    numeric_summary[str(col)] = {
                        "mean": float(series.mean()),
                        "std": float(series.std()),
                        "min": float(series.min()),
                        "max": float(series.max()),
                    }
            return {
                "type": "dataframe",
                "shape": [int(value.shape[0]), int(value.shape[1])],
                "columns": [str(c) for c in value.columns],
                "head_rows": int(min(len(value), max_df_rows)),
                "data": value.head(max_df_rows).to_dict(orient="records"),
                "numeric_summary": numeric_summary,
                "truncated": bool(len(value) > max_df_rows),
            }
        if isinstance(value, pd.Series):
            numeric = pd.to_numeric(value, errors="coerce").dropna()
            return {
                "type": "series",
                "length": int(len(value)),
                "head_values": value.head(max_series_rows).tolist(),
                "truncated": bool(len(value) > max_series_rows),
                "stats": {
                    "mean": float(numeric.mean()) if not numeric.empty else None,
                    "std": float(numeric.std()) if not numeric.empty else None,
                    "min": float(numeric.min()) if not numeric.empty else None,
                    "max": float(numeric.max()) if not numeric.empty else None,
                },
            }
        if isinstance(value, np.ndarray):
            arr = np.asarray(value)
            flattened = arr.reshape(-1)
            sample = flattened[:max_array_elements]
            return {
                "type": "ndarray",
                "shape": [int(x) for x in arr.shape],
                "sample": sample.tolist(),
                "sample_size": int(len(sample)),
                "total_elements": int(flattened.size),
                "truncated": bool(flattened.size > max_array_elements),
                "stats": {
                    "mean": float(arr.mean()) if arr.size else None,
                    "std": float(arr.std()) if arr.size else None,
                    "min": float(arr.min()) if arr.size else None,
                    "max": float(arr.max()) if arr.size else None,
                },
            }
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    return value


def _write_output_artifacts(results: Any, user_id: str = "default") -> Dict[str, str]:
    """Write analysis artifacts into output/ for easy user discovery."""
    safe_user = (
        "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(user_id))
        or "default"
    )
    output_dir = PROJECT_ROOT / "output" / safe_user
    output_dir.mkdir(parents=True, exist_ok=True)
    created: Dict[str, str] = {}

    def _safe_name(name: str) -> str:
        cleaned = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name
        )
        return cleaned.strip("_") or "result"

    def _write_json(file_path: Path, payload: Any) -> None:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def _write_analysis_artifact(key: str, value: Any) -> str:
        safe_key = _safe_name(key)
        try:
            import numpy as np
            import pandas as pd

            if isinstance(value, pd.DataFrame):
                out_file = output_dir / f"{safe_key}.csv"
                value.to_csv(out_file, index=True)
                return str(out_file)

            if isinstance(value, pd.Series):
                out_file = output_dir / f"{safe_key}.csv"
                value.to_frame(name=safe_key).to_csv(out_file, index=True)
                return str(out_file)

            if isinstance(value, np.ndarray):
                out_file = output_dir / f"{safe_key}.json"
                _write_json(
                    out_file,
                    {
                        "type": "ndarray",
                        "shape": [int(x) for x in value.shape],
                        "data": value.tolist(),
                    },
                )
                return str(out_file)
        except Exception:
            pass

        out_file = output_dir / f"{safe_key}.json"
        _write_json(out_file, {"value": _to_serializable(value)})
        return str(out_file)

    if not isinstance(results, dict):
        summary_file = output_dir / "analysis_results.json"
        _write_json(summary_file, {"results": _to_serializable(results)})
        created["analysis_results"] = str(summary_file)
        return created

    for key, value in results.items():
        try:
            created[key] = _write_analysis_artifact(str(key), value)
        except Exception:
            # Keep run successful even if one artifact fails to serialize.
            created[key] = "failed_to_export"

    summary_payload = {
        "generated_at": datetime.now().isoformat(),
        "result_keys": list(results.keys()),
        "results": _to_serializable(results),
        "artifacts": created,
    }
    summary_file = output_dir / "analysis_results.json"
    _write_json(summary_file, summary_payload)
    created["analysis_results"] = str(summary_file)

    # --- Retention / compression policy --------------------------------
    # Keep a gzip-compressed versioned copy of every run's summary and
    # rotate out old copies so the output/ directory doesn't grow unbounded.
    _MAX_RETAINED_SUMMARIES = 10
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_gz = output_dir / f"analysis_results_{ts}.json.gz"
    with summary_file.open("rb") as src, gzip.open(versioned_gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    created["analysis_results_versioned"] = str(versioned_gz)

    # Rotate: keep only the N most-recent versioned backups
    backups = sorted(
        output_dir.glob("analysis_results_*.json.gz"), key=lambda p: p.stat().st_mtime
    )
    for old in backups[:-_MAX_RETAINED_SUMMARIES]:
        try:
            old.unlink()
        except OSError:
            pass
    # --------------------------------------------------------------------

    return created


def quick_diagnostics(user_id: str = "default") -> str:
    """Return a concise multi-line diagnosis for the latest run failures."""
    safe_user = (
        "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(user_id))
        or "default"
    )
    base = PROJECT_ROOT / "data" / "users" / safe_user / "processed" / "quality"
    report_file = base / "quality_report.json"
    dead_letter = base / "dead_letter.jsonl"

    lines = [f"Diagnostics for user={safe_user}"]
    if report_file.exists():
        try:
            payload = json.loads(report_file.read_text(encoding="utf-8"))
            files = payload.get("files", {})
            failed = [k for k, v in files.items() if v.get("status") == "failed"]
            lines.append(f"- Failed entities: {len(failed)}")
            if failed:
                lines.append(f"- Top failed: {failed[:5]}")
            summary = payload.get("summary", {})
            missing_sources = summary.get("missing_sources", [])
            lines.append(f"- Missing sources: {missing_sources}")
        except Exception as exc:
            lines.append(f"- quality_report parse error: {exc}")
    else:
        lines.append("- quality_report.json not found")

    if dead_letter.exists():
        try:
            entries = dead_letter.read_text(encoding="utf-8").strip().splitlines()
            lines.append(f"- Dead-letter entries: {len(entries)}")
            if entries:
                last = json.loads(entries[-1])
                lines.append(
                    "- Last error: "
                    f"{last.get('entity')} | {last.get('error_type')} | {last.get('error_message')}"
                )
        except Exception as exc:
            lines.append(f"- dead-letter parse error: {exc}")
    else:
        lines.append("- dead_letter.jsonl not found")

    return "\n".join(lines)


def main() -> None:
    start_time = time.time()
    run_id = str(uuid.uuid4())
    correlation_id = str(uuid.uuid4())
    catalog.set_run_context(run_id=run_id, correlation_id=correlation_id)

    print(APPLICATION_TITLE)
    print(LIVE_STEP_0_WELCOME)
    print(QUICK_START)

    try:
        # Prerequisites check
        print(LIVE_STEP_1_PREREQUISITES_CHECK)

        # Load configuration
        print(LIVE_STEP_2_CONFIG_LOADING)
        config = ProjectConfig.load_from_env()
        config_contract = config.to_serializable_dict()
        config_hash = _hash_payload(config_contract)
        code_version = os.getenv("GIT_COMMIT_SHA", "unversioned")
        pyproject_hash = _hash_file_if_exists(Path("pyproject.toml"))
        logger.info(
            MAIN_CONFIG_LOADED.format(config_details=f"mode={config.mode.value}")
        )
        catalog.log_operation(
            "config_load",
            "main",
            {
                "mode": config.mode.value,
                "random_seed": config.random_seed,
                "enforce_reproducibility": config.enforce_reproducibility,
                "config_hash": config_hash,
            },
            {"run_id": run_id, "correlation_id": correlation_id},
            "Configuration loaded",
        )
        catalog.log_operation(
            "run_contract",
            "main",
            {
                "config_hash": config_hash,
                "pyproject_hash": pyproject_hash,
                "code_version": code_version,
                "seed_policy": (
                    "deterministic" if config.enforce_reproducibility else "stochastic"
                ),
            },
            {"phase": "pre_pipeline"},
            "Run contract initialized",
        )
        print(MAIN_START.format(mode=config.mode.value.upper()))

        # Data fetching notification
        print(LIVE_STEP_3_DATA_FETCHING_START)

        # Initialize Factory
        factory = DataFactory(fred_api_key=config.fred_api_key)

        # Initialize and run full pipeline
        logger.info(MAIN_PIPELINE_START)
        pipeline_start = time.time()
        pipeline = MedallionPipeline(config=config, factory=factory)

        # Choose parallel or sequential
        if config.should_use_parallel_pipeline() and config.mode == RunMode.ACTUAL:
            results = pipeline.run_full_pipeline_parallel()
        else:
            results = pipeline.run_full_pipeline_sequential()

        output_artifacts = _write_output_artifacts(
            results, user_id=getattr(config, "data_user_id", "default")
        )
        ai_brief = generate_ai_pipeline_brief(
            user_id=getattr(config, "data_user_id", "default")
        )
        if ai_brief.get("success"):
            output_artifacts["ai_pipeline_briefing_json"] = str(
                ai_brief.get("json_path", "")
            )
            output_artifacts["ai_pipeline_briefing_md"] = str(
                ai_brief.get("md_path", "")
            )
        else:
            output_artifacts["ai_pipeline_briefing_status"] = (
                f"skipped: {ai_brief.get('error', 'unknown')}")

        pipeline_duration = time.time() - pipeline_start
        data_catalog_hash = _hash_file_if_exists(pipeline.raw_path / "catalog.json")
        results_count = len(results) if hasattr(results, "__len__") else 0
        catalog.log_operation(
            "pipeline_complete",
            "main",
            {
                "duration_seconds": pipeline_duration,
                "results_count": results_count,
                "output_files_written": len(output_artifacts),
            },
            {
                "results_keys": (
                    list(results.keys()) if isinstance(results, dict) else []
                ),
                "data_catalog_hash": data_catalog_hash,
                "output_artifacts": output_artifacts,
            },
            "Full pipeline completed",
        )

        catalog.log_operation(
            "run_contract",
            "main",
            {
                "config_hash": config_hash,
                "data_catalog_hash": data_catalog_hash,
                "pyproject_hash": pyproject_hash,
                "code_version": code_version,
            },
            {"phase": "post_pipeline"},
            "Run contract finalized",
        )

        execution_time = time.time() - start_time
        logger.info(MAIN_PIPELINE_SUCCESS)
        catalog.log_operation(
            "session_complete",
            "main",
            {"total_duration": execution_time, "success": True},
            {},
            "Application session completed successfully",
        )

        # Completion message with metrics
        metrics = catalog.get_metrics_summary()
        if isinstance(results, dict):
            result_keys = list(results.keys())
        else:
            result_keys = []
        print(
            LIVE_STEP_8_COMPLETION.format(
                total_time=f"{execution_time:.2f}",
                total_records=metrics.get("data_processed", 0),
                analyses_count=metrics.get("analyses_completed", 0),
                files_created=len(results) if hasattr(results, "__len__") else 0,
            )
        )

        print(MAIN_RESULTS_SUMMARY.format(result_keys=result_keys))
        print(MAIN_COMPLETION.format(execution_time=f"{execution_time:.2f}"))

        # Save session summary
        catalog.save_session_summary()

    except FetcherError as e:
        print(LIVE_ERROR_API_KEY if "API" in str(e) else LIVE_ERROR_NETWORK)
        catalog.log_error("main", "FetcherError", str(e), "pipeline_execution")
        logger.error(f"Fetcher Error: {e}")
        print(f"Fetcher Error: {_user_error_message(e)}")
        sys.exit(1)
    except ValueError as e:
        catalog.log_error("main", "ConfigError", str(e), "config_loading")
        logger.error(f"Configuration Error: {e}")
        print(_user_error_message(e))
        sys.exit(1)
    except DataPipelineError as e:
        catalog.log_error("main", "DataPipelineError", str(e), "pipeline_execution")
        logger.error(f"Data Pipeline Error: {e}")
        print(f"Data Pipeline Error: {_user_error_message(e)}")
        sys.exit(1)
    except Exception as e:
        catalog.log_error("main", "UnexpectedError", str(e), "application_execution")
        logger.error(f"Unexpected Application Error: {e}")
        print(f"Unexpected Application Error: {e}. Contact support.")
        sys.exit(1)


if __name__ == "__main__":
    main()
