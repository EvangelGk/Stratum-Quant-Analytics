import hashlib
import importlib
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _import_first(*module_names: str) -> Any:
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(f"Unable to import any of: {module_names}")


_fetchers_exc = _import_first(
    "src.exceptions.FetchersExceptions", "exceptions.FetchersExceptions"
)
_medallion_exc = _import_first(
    "src.exceptions.MedallionExceptions", "exceptions.MedallionExceptions"
)
_factory_mod = _import_first("src.Fetchers.Factory", "Fetchers.Factory")
_project_cfg_mod = _import_first("src.Fetchers.ProjectConfig", "Fetchers.ProjectConfig")
_catalog_mod = _import_first("src.logger.Catalog", "logger.Catalog")
_directions_mod = _import_first(
    "src.logger.Messages.DirectionsMess", "logger.Messages.DirectionsMess"
)
_main_msg_mod = _import_first("src.logger.Messages.MainMess", "logger.Messages.MainMess")
_medallion_mod = _import_first("src.Medallion", "Medallion")

FetcherError = _fetchers_exc.FetcherError
DataPipelineError = _medallion_exc.DataPipelineError
DataFactory = _factory_mod.DataFactory
ProjectConfig = _project_cfg_mod.ProjectConfig
RunMode = _project_cfg_mod.RunMode
catalog = _catalog_mod.catalog
LIVE_ERROR_API_KEY = _directions_mod.LIVE_ERROR_API_KEY
LIVE_ERROR_NETWORK = _directions_mod.LIVE_ERROR_NETWORK
LIVE_STEP_0_WELCOME = _directions_mod.LIVE_STEP_0_WELCOME
LIVE_STEP_1_PREREQUISITES_CHECK = _directions_mod.LIVE_STEP_1_PREREQUISITES_CHECK
LIVE_STEP_2_CONFIG_LOADING = _directions_mod.LIVE_STEP_2_CONFIG_LOADING
LIVE_STEP_3_DATA_FETCHING_START = _directions_mod.LIVE_STEP_3_DATA_FETCHING_START
LIVE_STEP_8_COMPLETION = _directions_mod.LIVE_STEP_8_COMPLETION
APPLICATION_TITLE = _main_msg_mod.APPLICATION_TITLE
MAIN_COMPLETION = _main_msg_mod.MAIN_COMPLETION
MAIN_CONFIG_LOADED = _main_msg_mod.MAIN_CONFIG_LOADED
MAIN_PIPELINE_START = _main_msg_mod.MAIN_PIPELINE_START
MAIN_PIPELINE_SUCCESS = _main_msg_mod.MAIN_PIPELINE_SUCCESS
MAIN_RESULTS_SUMMARY = _main_msg_mod.MAIN_RESULTS_SUMMARY
MAIN_START = _main_msg_mod.MAIN_START
QUICK_START = _main_msg_mod.QUICK_START
MedallionPipeline = _medallion_mod.MedallionPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

        if isinstance(value, pd.DataFrame):
            return {
                "type": "dataframe",
                "shape": [int(value.shape[0]), int(value.shape[1])],
                "columns": [str(c) for c in value.columns],
                "data": value.head(200).to_dict(orient="records"),
            }
        if isinstance(value, pd.Series):
            return {
                "type": "series",
                "length": int(len(value)),
                "data": value.head(500).tolist(),
            }
        if isinstance(value, np.ndarray):
            return {
                "type": "ndarray",
                "shape": [int(x) for x in value.shape],
                "data": value.tolist(),
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


def _write_output_artifacts(results: Any) -> Dict[str, str]:
    """Write analysis artifacts into output/ for easy user discovery."""
    output_dir = Path("output")
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
    return created


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

        output_artifacts = _write_output_artifacts(results)

        pipeline_duration = time.time() - pipeline_start
        data_catalog_hash = _hash_file_if_exists(Path("data/raw/catalog.json"))
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
        print(f"Fetcher Error: {e}. Check API keys or network.")
    except ValueError as e:
        catalog.log_error("main", "ConfigError", str(e), "config_loading")
        logger.error(f"Configuration Error: {e}")
        print(str(e))
    except DataPipelineError as e:
        catalog.log_error("main", "DataPipelineError", str(e), "pipeline_execution")
        logger.error(f"Data Pipeline Error: {e}")
        print(f"Data Pipeline Error: {e}. Check data integrity or resources.")
    except Exception as e:
        catalog.log_error("main", "UnexpectedError", str(e), "application_execution")
        logger.error(f"Unexpected Application Error: {e}")
        print(f"Unexpected Application Error: {e}. Contact support.")


if __name__ == "__main__":
    main()
