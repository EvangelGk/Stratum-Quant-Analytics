from Fetchers.ProjectConfig import ProjectConfig
from Fetchers.Factory import DataFactory
from Medallion import MedallionPipeline
from exceptions.MedallionExceptions import DataPipelineError
from exceptions.FetchersExceptions import FetcherError

def main():
    try:
        # Load configuration
        config = ProjectConfig.load_from_env()
        print(f"--- Running in {config.mode.value.upper()} mode ---")

        # Initialize Factory
        factory = DataFactory(fred_api_key=config.fred_api_key)

        # Initialize and run full pipeline
        pipeline = MedallionPipeline(config=config, factory=factory)

        # Choose parallel or sequential
        if config.mode.value == 'production':  # Assuming config has mode
            results = pipeline.run_full_pipeline_parallel()
        else:
            results = pipeline.run_full_pipeline_sequential()

        print("Full pipeline executed successfully.")
        print(f"Analysis Results Keys: {list(results.keys())}")

    except FetcherError as e:
        print(f"Fetcher Error: {e}. Check API keys or network.")
    except DataPipelineError as e:
        print(f"Data Pipeline Error: {e}. Check data integrity or resources.")
    except Exception as e:
        print(f"Unexpected Application Error: {e}. Contact support.")

if __name__ == "__main__":
    main()