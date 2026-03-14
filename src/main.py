from Fetchers.ProjectConfig import ProjectConfig
from Fetchers.Factory import DataFactory
from Medallion import MedallionPipeline

def main():
    try:
        # 1.Single Point of Failure
        config = ProjectConfig.load_from_env()
        print(f"--- Running in {config.mode.value.upper()} mode ---")

        # 2. Initialize Factory with the key from config
        # The main function acts as a 'dependency injector'
        factory = DataFactory(fred_api_key=config.fred_api_key)

        # 3. Start the Pipeline
        pipeline = MedallionPipeline(config=config, factory=factory)
        pipeline.run_bronze()
        
        print("Pipeline execution successful.")

    except Exception as e:
        print(f"Application failed: {e}")

if __name__ == "__main__":
    main()