from Fetchers.ProjectConfig import ProjectConfig
from Fetchers.Factory import DataFactory
from Medallion import MedallionPipeline
from exceptions.MedallionExceptions import DataPipelineError
from exceptions.FetchersExceptions import FetcherError
from logger.Messages.MainMess import (
    MAIN_START, MAIN_CONFIG_LOADED, MAIN_PIPELINE_START,
    MAIN_PIPELINE_SUCCESS, MAIN_RESULTS_SUMMARY, MAIN_COMPLETION,
    APPLICATION_TITLE, QUICK_START
)
from logger.Messages.DirectionsMess import (
    LIVE_STEP_0_WELCOME, LIVE_STEP_1_PREREQUISITES_CHECK,
    LIVE_STEP_2_CONFIG_LOADING, LIVE_STEP_3_DATA_FETCHING_START,
    LIVE_STEP_8_COMPLETION, LIVE_ERROR_API_KEY, LIVE_ERROR_NETWORK
)
from logger.Catalog import catalog
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    start_time = time.time()

    print(APPLICATION_TITLE)
    print(LIVE_STEP_0_WELCOME)
    print(QUICK_START)

    try:
        # Prerequisites check
        print(LIVE_STEP_1_PREREQUISITES_CHECK)

        # Load configuration
        print(LIVE_STEP_2_CONFIG_LOADING)
        config = ProjectConfig.load_from_env()
        logger.info(MAIN_CONFIG_LOADED.format(config_details=f"mode={config.mode.value}"))
        catalog.log_operation("config_load", "main", {"mode": config.mode.value}, {}, "Configuration loaded")
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
        if config.mode.value == 'production':  # Assuming config has mode
            results = pipeline.run_full_pipeline_parallel()
        else:
            results = pipeline.run_full_pipeline_sequential()

        pipeline_duration = time.time() - pipeline_start
        catalog.log_operation("pipeline_complete", "main",
                           {"duration_seconds": pipeline_duration, "results_count": len(results)},
                           {"results_keys": list(results.keys())}, "Full pipeline completed")

        execution_time = time.time() - start_time
        logger.info(MAIN_PIPELINE_SUCCESS)
        catalog.log_operation("session_complete", "main",
                           {"total_duration": execution_time, "success": True},
                           {}, "Application session completed successfully")

        # Completion message with metrics
        metrics = catalog.get_metrics_summary()
        if isinstance(results, dict):
            result_keys = list(results.keys())
        else:
            result_keys = []
        print(LIVE_STEP_8_COMPLETION.format(
            total_time=f"{execution_time:.2f}",
            total_records=metrics.get('data_processed', 0),
            analyses_count=metrics.get('analyses_completed', 0),
            files_created=len(results) if hasattr(results, '__len__') else 0
        ))

        print(MAIN_RESULTS_SUMMARY.format(result_keys=result_keys))
        print(MAIN_COMPLETION.format(execution_time=f"{execution_time:.2f}"))

        # Save session summary
        catalog.save_session_summary()

    except FetcherError as e:
        print(LIVE_ERROR_API_KEY if "API" in str(e) else LIVE_ERROR_NETWORK)
        catalog.log_error("main", "FetcherError", str(e), "pipeline_execution")
        logger.error(f"Fetcher Error: {e}")
        print(f"Fetcher Error: {e}. Check API keys or network.")
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