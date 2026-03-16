import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

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

        pipeline_duration = time.time() - pipeline_start
        data_catalog_hash = _hash_file_if_exists(Path("data/raw/catalog.json"))
        catalog.log_operation(
            "pipeline_complete",
            "main",
            {"duration_seconds": pipeline_duration, "results_count": len(results)},
            {
                "results_keys": list(results.keys()),
                "data_catalog_hash": data_catalog_hash,
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
