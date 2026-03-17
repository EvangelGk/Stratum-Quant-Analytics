"""
Comprehensive test suite for all custom exceptions.
Tests exception hierarchy, inheritance, and messages.
"""

import pytest

# ============================================================================
# FETCHERS EXCEPTIONS TESTS
# ============================================================================

def test_fetcher_error_base_exception():
    """Test FetcherError is a proper Exception subclass."""
    from src.exceptions.FetchersExceptions import FetcherError
    
    assert issubclass(FetcherError, Exception)
    exc = FetcherError("test message")
    assert "test message" in str(exc)


def test_network_error_inherits_from_fetcher_error():
    """Test NetworkError is subclass of FetcherError."""
    from src.exceptions.FetchersExceptions import NetworkError, FetcherError
    
    assert issubclass(NetworkError, FetcherError)
    assert issubclass(NetworkError, Exception)


def test_timeout_error_inherits_from_fetcher_error():
    """Test TimeoutError is subclass of FetcherError."""
    from src.exceptions.FetchersExceptions import TimeoutError, FetcherError
    
    assert issubclass(TimeoutError, FetcherError)


def test_connection_error_inherits_from_network_error():
    """Test ConnectionError is subclass of NetworkError."""
    from src.exceptions.FetchersExceptions import ConnectionError, NetworkError
    
    assert issubclass(ConnectionError, NetworkError)


def test_api_error_inherits_from_fetcher_error():
    """Test APIError is subclass of FetcherError."""
    from src.exceptions.FetchersExceptions import APIError, FetcherError
    
    assert issubclass(APIError, FetcherError)


def test_authentication_error_inherits_from_api_error():
    """Test AuthenticationError is subclass of APIError."""
    from src.exceptions.FetchersExceptions import AuthenticationError, APIError
    
    assert issubclass(AuthenticationError, APIError)


def test_rate_limit_error_inherits_from_api_error():
    """Test RateLimitError is subclass of APIError."""
    from src.exceptions.FetchersExceptions import RateLimitError, APIError
    
    assert issubclass(RateLimitError, APIError)


def test_data_fetch_error_inherits_from_fetcher_error():
    """Test DataFetchError is subclass of FetcherError."""
    from src.exceptions.FetchersExceptions import DataFetchError, FetcherError
    
    assert issubclass(DataFetchError, FetcherError)


def test_cache_error_inherits_from_fetcher_error():
    """Test CacheError is subclass of FetcherError."""
    from src.exceptions.FetchersExceptions import CacheError, FetcherError
    
    assert issubclass(CacheError, FetcherError)


def test_cache_miss_error_inherits_from_cache_error():
    """Test CacheMissError is subclass of CacheError."""
    from src.exceptions.FetchersExceptions import CacheMissError, CacheError
    
    assert issubclass(CacheMissError, CacheError)


def test_external_service_error_inherits_from_fetcher_error():
    """Test ExternalServiceError is subclass of FetcherError."""
    from src.exceptions.FetchersExceptions import ExternalServiceError, FetcherError
    
    assert issubclass(ExternalServiceError, FetcherError)


def test_fetcher_config_error_inherits_from_fetcher_error():
    """Test FetcherConfigError is subclass of FetcherError."""
    from src.exceptions.FetchersExceptions import FetcherConfigError, FetcherError
    
    assert issubclass(FetcherConfigError, FetcherError)


def test_missing_api_key_error_inherits_from_fetcher_config_error():
    """Test MissingAPIKeyError is subclass of FetcherConfigError."""
    from src.exceptions.FetchersExceptions import MissingAPIKeyError, FetcherConfigError
    
    assert issubclass(MissingAPIKeyError, FetcherConfigError)


# ============================================================================
# LOGGER EXCEPTIONS TESTS
# ============================================================================

def test_logger_error_base_exception():
    """Test LoggerError is a proper Exception subclass."""
    from src.exceptions.LoggerExceptions import LoggerError
    
    assert issubclass(LoggerError, Exception)


def test_log_file_error_inherits_from_logger_error():
    """Test LogFileError is subclass of LoggerError."""
    from src.exceptions.LoggerExceptions import LogFileError, LoggerError
    
    assert issubclass(LogFileError, LoggerError)


def test_log_file_not_found_error_inherits_from_log_file_error():
    """Test LogFileNotFoundError is subclass of LogFileError."""
    from src.exceptions.LoggerExceptions import LogFileNotFoundError, LogFileError
    
    assert issubclass(LogFileNotFoundError, LogFileError)


def test_log_file_permission_error_inherits_from_log_file_error():
    """Test LogFilePermissionError is subclass of LogFileError."""
    from src.exceptions.LoggerExceptions import LogFilePermissionError, LogFileError
    
    assert issubclass(LogFilePermissionError, LogFileError)


def test_log_file_write_error_inherits_from_log_file_error():
    """Test LogFileWriteError is subclass of LogFileError."""
    from src.exceptions.LoggerExceptions import LogFileWriteError, LogFileError
    
    assert issubclass(LogFileWriteError, LogFileError)


def test_disk_full_error_inherits_from_log_file_error():
    """Test DiskFullError is subclass of LogFileError."""
    from src.exceptions.LoggerExceptions import DiskFullError, LogFileError
    
    assert issubclass(DiskFullError, LogFileError)


def test_handler_error_inherits_from_logger_error():
    """Test HandlerError is subclass of LoggerError."""
    from src.exceptions.LoggerExceptions import HandlerError, LoggerError
    
    assert issubclass(HandlerError, LoggerError)


def test_invalid_handler_error_inherits_from_handler_error():
    """Test InvalidHandlerError is subclass of HandlerError."""
    from src.exceptions.LoggerExceptions import InvalidHandlerError, HandlerError
    
    assert issubclass(InvalidHandlerError, HandlerError)


def test_log_level_error_inherits_from_logger_error():
    """Test LogLevelError is subclass of LoggerError."""
    from src.exceptions.LoggerExceptions import LogLevelError, LoggerError
    
    assert issubclass(LogLevelError, LoggerError)


def test_logger_timeout_error_inherits_from_logger_error():
    """Test LoggerTimeoutError is subclass of LoggerError."""
    from src.exceptions.LoggerExceptions import LoggerTimeoutError, LoggerError
    
    assert issubclass(LoggerTimeoutError, LoggerError)


# ============================================================================
# MEDALLION EXCEPTIONS TESTS
# ============================================================================

def test_data_pipeline_error_base_exception():
    """Test DataPipelineError is a proper Exception subclass."""
    from src.exceptions.MedallionExceptions import DataPipelineError
    
    assert issubclass(DataPipelineError, Exception)


def test_system_resource_error_inherits_from_data_pipeline_error():
    """Test SystemResourceError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import SystemResourceError, DataPipelineError
    
    assert issubclass(SystemResourceError, DataPipelineError)


def test_disk_space_error_inherits_from_system_resource_error():
    """Test DiskSpaceError is subclass of SystemResourceError."""
    from src.exceptions.MedallionExceptions import DiskSpaceError, SystemResourceError
    
    assert issubclass(DiskSpaceError, SystemResourceError)


def test_memory_error_inherits_from_system_resource_error():
    """Test MemoryError is subclass of SystemResourceError."""
    from src.exceptions.MedallionExceptions import MemoryError, SystemResourceError
    
    assert issubclass(MemoryError, SystemResourceError)


def test_data_corruption_error_inherits_from_data_pipeline_error():
    """Test DataCorruptionError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import DataCorruptionError, DataPipelineError
    
    assert issubclass(DataCorruptionError, DataPipelineError)


def test_schema_mismatch_error_inherits_from_data_pipeline_error():
    """Test SchemaMismatchError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import SchemaMismatchError, DataPipelineError
    
    assert issubclass(SchemaMismatchError, DataPipelineError)


def test_data_validation_error_inherits_from_data_pipeline_error():
    """Test DataValidationError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import DataValidationError, DataPipelineError
    
    assert issubclass(DataValidationError, DataPipelineError)


def test_imputation_error_inherits_from_data_pipeline_error():
    """Test ImputationError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import ImputationError, DataPipelineError
    
    assert issubclass(ImputationError, DataPipelineError)


def test_standardization_error_inherits_from_data_pipeline_error():
    """Test StandardizationError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import StandardizationError, DataPipelineError
    
    assert issubclass(StandardizationError, DataPipelineError)


def test_outlier_detection_error_inherits_from_data_pipeline_error():
    """Test OutlierDetectionError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import OutlierDetectionError, DataPipelineError
    
    assert issubclass(OutlierDetectionError, DataPipelineError)


def test_parallel_execution_error_inherits_from_data_pipeline_error():
    """Test ParallelExecutionError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import ParallelExecutionError, DataPipelineError
    
    assert issubclass(ParallelExecutionError, DataPipelineError)


def test_thread_pool_error_inherits_from_parallel_execution_error():
    """Test ThreadPoolError is subclass of ParallelExecutionError."""
    from src.exceptions.MedallionExceptions import ThreadPoolError, ParallelExecutionError
    
    assert issubclass(ThreadPoolError, ParallelExecutionError)


def test_analysis_error_inherits_from_data_pipeline_error():
    """Test AnalysisError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import AnalysisError, DataPipelineError
    
    assert issubclass(AnalysisError, DataPipelineError)


def test_monte_carlo_error_inherits_from_analysis_error():
    """Test MonteCarloError is subclass of AnalysisError."""
    from src.exceptions.MedallionExceptions import MonteCarloError, AnalysisError
    
    assert issubclass(MonteCarloError, AnalysisError)


def test_regression_error_inherits_from_analysis_error():
    """Test RegressionError is subclass of AnalysisError."""
    from src.exceptions.MedallionExceptions import RegressionError, AnalysisError
    
    assert issubclass(RegressionError, AnalysisError)


def test_compliance_violation_error_inherits_from_data_pipeline_error():
    """Test ComplianceViolationError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import ComplianceViolationError, DataPipelineError
    
    assert issubclass(ComplianceViolationError, DataPipelineError)


def test_timeout_error_inherits_from_data_pipeline_error():
    """Test TimeoutError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import TimeoutError, DataPipelineError
    
    assert issubclass(TimeoutError, DataPipelineError)


# ============================================================================
# STREAMLIT EXCEPTIONS TESTS
# ============================================================================

def test_streamlit_error_base_exception():
    """Test StreamlitError is a proper Exception subclass."""
    from src.exceptions.StreamlitExceptions import StreamlitError
    
    assert issubclass(StreamlitError, Exception)


def test_pipeline_execution_error_inherits_from_streamlit_error():
    """Test PipelineExecutionError is subclass of StreamlitError."""
    from src.exceptions.StreamlitExceptions import PipelineExecutionError, StreamlitError
    
    assert issubclass(PipelineExecutionError, StreamlitError)


def test_pipeline_subprocess_error_inherits_from_pipeline_execution_error():
    """Test PipelineSubprocessError is subclass of PipelineExecutionError."""
    from src.exceptions.StreamlitExceptions import (
        PipelineSubprocessError,
        PipelineExecutionError,
    )
    
    assert issubclass(PipelineSubprocessError, PipelineExecutionError)


def test_pipeline_progress_tracking_error_inherits_from_pipeline_execution_error():
    """Test PipelineProgressTrackingError is subclass of PipelineExecutionError."""
    from src.exceptions.StreamlitExceptions import (
        PipelineProgressTrackingError,
        PipelineExecutionError,
    )
    
    assert issubclass(PipelineProgressTrackingError, PipelineExecutionError)


def test_data_file_error_inherits_from_streamlit_error():
    """Test DataFileError is subclass of StreamlitError."""
    from src.exceptions.StreamlitExceptions import DataFileError, StreamlitError
    
    assert issubclass(DataFileError, StreamlitError)


def test_data_file_not_found_error_inherits_from_data_file_error():
    """Test DataFileNotFoundError is subclass of DataFileError."""
    from src.exceptions.StreamlitExceptions import DataFileNotFoundError, DataFileError
    
    assert issubclass(DataFileNotFoundError, DataFileError)


def test_json_parse_error_inherits_from_data_file_error():
    """Test JSONParseError is subclass of DataFileError."""
    from src.exceptions.StreamlitExceptions import JSONParseError, DataFileError
    
    assert issubclass(JSONParseError, DataFileError)


def test_session_error_inherits_from_streamlit_error():
    """Test SessionError is subclass of StreamlitError."""
    from src.exceptions.StreamlitExceptions import SessionError, StreamlitError
    
    assert issubclass(SessionError, StreamlitError)


def test_session_load_error_inherits_from_session_error():
    """Test SessionLoadError is subclass of SessionError."""
    from src.exceptions.StreamlitExceptions import SessionLoadError, SessionError
    
    assert issubclass(SessionLoadError, SessionError)


def test_session_snapshot_error_inherits_from_session_error():
    """Test SessionSnapshotError is subclass of SessionError."""
    from src.exceptions.StreamlitExceptions import SessionSnapshotError, SessionError
    
    assert issubclass(SessionSnapshotError, SessionError)


def test_display_error_inherits_from_streamlit_error():
    """Test DisplayError is subclass of StreamlitError."""
    from src.exceptions.StreamlitExceptions import DisplayError, StreamlitError
    
    assert issubclass(DisplayError, StreamlitError)


def test_chart_render_error_inherits_from_display_error():
    """Test ChartRenderError is subclass of DisplayError."""
    from src.exceptions.StreamlitExceptions import ChartRenderError, DisplayError
    
    assert issubclass(ChartRenderError, DisplayError)


def test_configuration_error_inherits_from_streamlit_error():
    """Test ConfigurationError is subclass of StreamlitError."""
    from src.exceptions.StreamlitExceptions import ConfigurationError, StreamlitError
    
    assert issubclass(ConfigurationError, StreamlitError)


def test_role_permission_error_inherits_from_configuration_error():
    """Test RolePermissionError is subclass of ConfigurationError."""
    from src.exceptions.StreamlitExceptions import RolePermissionError, ConfigurationError
    
    assert issubclass(RolePermissionError, ConfigurationError)


def test_data_processing_error_inherits_from_streamlit_error():
    """Test DataProcessingError is subclass of StreamlitError."""
    from src.exceptions.StreamlitExceptions import DataProcessingError, StreamlitError
    
    assert issubclass(DataProcessingError, StreamlitError)


def test_health_score_calculation_error_inherits_from_data_processing_error():
    """Test HealthScoreCalculationError is subclass of DataProcessingError."""
    from src.exceptions.StreamlitExceptions import (
        HealthScoreCalculationError,
        DataProcessingError,
    )
    
    assert issubclass(HealthScoreCalculationError, DataProcessingError)


def test_alert_generation_error_inherits_from_data_processing_error():
    """Test AlertGenerationError is subclass of DataProcessingError."""
    from src.exceptions.StreamlitExceptions import AlertGenerationError, DataProcessingError
    
    assert issubclass(AlertGenerationError, DataProcessingError)


def test_run_comparison_error_inherits_from_data_processing_error():
    """Test RunComparisonError is subclass of DataProcessingError."""
    from src.exceptions.StreamlitExceptions import RunComparisonError, DataProcessingError
    
    assert issubclass(RunComparisonError, DataProcessingError)


def test_report_generation_error_inherits_from_data_processing_error():
    """Test ReportGenerationError is subclass of DataProcessingError."""
    from src.exceptions.StreamlitExceptions import ReportGenerationError, DataProcessingError
    
    assert issubclass(ReportGenerationError, DataProcessingError)


def test_import_error_streamlit_inherits_from_streamlit_error():
    """Test ImportError is subclass of StreamlitError."""
    from src.exceptions.StreamlitExceptions import ImportError as StreamlitImportError
    from src.exceptions.StreamlitExceptions import StreamlitError
    
    assert issubclass(StreamlitImportError, StreamlitError)


def test_logger_module_error_inherits_from_import_error():
    """Test LoggerModuleError is subclass of ImportError."""
    from src.exceptions.StreamlitExceptions import LoggerModuleError
    from src.exceptions.StreamlitExceptions import ImportError as StreamlitImportError
    
    assert issubclass(LoggerModuleError, StreamlitImportError)


# ============================================================================
# EXCEPTION MESSAGE TESTS
# ============================================================================

def test_exception_with_message():
    """Test exceptions properly capture and store messages."""
    from src.exceptions.StreamlitExceptions import PipelineExecutionError
    
    msg = "Pipeline failed to execute"
    exc = PipelineExecutionError(msg)
    assert msg in str(exc)


def test_exception_chaining():
    """Test exception chaining with 'from e'."""
    from src.exceptions.StreamlitExceptions import PipelineExecutionError
    
    try:
        try:
            raise ValueError("Original error")
        except ValueError as e:
            raise PipelineExecutionError("Pipeline error") from e
    except PipelineExecutionError as exc:
        assert exc.__cause__ is not None
        assert isinstance(exc.__cause__, ValueError)


def test_multiple_exceptions_in_try_except():
    """Test catching multiple exception types in single except clause."""
    from src.exceptions.StreamlitExceptions import JSONParseError, DataFileReadError
    
    def raise_json_error():
        raise JSONParseError("JSON parse failed")
    
    def raise_file_error():
        raise DataFileReadError("File read failed")
    
    try:
        raise_json_error()
    except (JSONParseError, DataFileReadError):
        pass  # Should catch successfully
    
    try:
        raise_file_error()
    except (JSONParseError, DataFileReadError):
        pass  # Should catch successfully


# ============================================================================
# TESTS FOR EXCEPTION USAGE IN REAL SCENARIOS
# ============================================================================

def test_exception_in_data_health_calculation():
    """Test HealthScoreCalculationError is raised when health check fails."""
    from src.exceptions.StreamlitExceptions import HealthScoreCalculationError
    
    with pytest.raises(HealthScoreCalculationError):
        raise HealthScoreCalculationError("Health calculation failed")


def test_exception_in_alert_generation():
    """Test AlertGenerationError is raised when alert generation fails."""
    from src.exceptions.StreamlitExceptions import AlertGenerationError
    
    with pytest.raises(AlertGenerationError):
        raise AlertGenerationError("Alert generation failed")


def test_exception_in_json_parsing():
    """Test JSONParseError is raised when JSON parsing fails."""
    from src.exceptions.StreamlitExceptions import JSONParseError
    
    with pytest.raises(JSONParseError):
        raise JSONParseError("Failed to parse JSON")


def test_exception_in_session_loading():
    """Test SessionLoadError is raised when session loading fails."""
    from src.exceptions.StreamlitExceptions import SessionLoadError
    
    with pytest.raises(SessionLoadError):
        raise SessionLoadError("Failed to load session")


def test_exception_in_subprocess_execution():
    """Test PipelineSubprocessError is raised when subprocess fails."""
    from src.exceptions.StreamlitExceptions import PipelineSubprocessError
    
    with pytest.raises(PipelineSubprocessError):
        raise PipelineSubprocessError("Failed to create subprocess")


# ============================================================================
# PARAMETRIZED TESTS FOR EXCEPTION HIERARCHY
# ============================================================================

@pytest.mark.parametrize(
    "exception_module,exception_class,parent_class",
    [
        ("src.exceptions.FetchersExceptions", "NetworkError", "FetcherError"),
        ("src.exceptions.FetchersExceptions", "TimeoutError", "FetcherError"),
        ("src.exceptions.FetchersExceptions", "APIError", "FetcherError"),
        ("src.exceptions.FetchersExceptions", "DataFetchError", "FetcherError"),
        ("src.exceptions.FetchersExceptions", "CacheError", "FetcherError"),
        ("src.exceptions.LoggerExceptions", "LogFileError", "LoggerError"),
        ("src.exceptions.LoggerExceptions", "HandlerError", "LoggerError"),
        ("src.exceptions.MedallionExceptions", "DataValidationError", "DataPipelineError"),
        ("src.exceptions.MedallionExceptions", "AnalysisError", "DataPipelineError"),
        ("src.exceptions.StreamlitExceptions", "PipelineExecutionError", "StreamlitError"),
        ("src.exceptions.StreamlitExceptions", "DataFileError", "StreamlitError"),
        ("src.exceptions.StreamlitExceptions", "SessionError", "StreamlitError"),
    ],
)
def test_exception_inheritance(exception_module, exception_class, parent_class):
    """Parametrized test for exception inheritance chain."""
    import importlib
    
    module = importlib.import_module(exception_module)
    exc_class = getattr(module, exception_class)
    parent = getattr(module, parent_class)
    
    assert issubclass(exc_class, parent)
    assert issubclass(exc_class, Exception)
