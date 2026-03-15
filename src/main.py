from Fetchers.ProjectConfig import ProjectConfig
from Fetchers.Factory import DataFactory
from Medallion.MedallionPipeline import MedallionPipeline

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

    except Exception as e:
        print(f"Application failed: {e}")

if __name__ == "__main__":
    main()