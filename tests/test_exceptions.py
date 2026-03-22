"""
Comprehensive test suite for all custom exceptions.
Tests exception hierarchy, inheritance, and messages.
"""

import inspect

import pytest

# ============================================================================
# AI AGENT EXCEPTIONS TESTS
# ============================================================================


def test_ai_agent_error_is_base_exception():
    """Test AIAgentError is a proper Exception subclass."""
    from src.exceptions.AIAgentExceptions import AIAgentError

    assert issubclass(AIAgentError, Exception)
    exc = AIAgentError("ai error message")
    assert "ai error message" in str(exc)


def test_llm_connection_error_inherits_from_ai_agent_error():
    from src.exceptions.AIAgentExceptions import AIAgentError, LLMConnectionError

    assert issubclass(LLMConnectionError, AIAgentError)


def test_llm_timeout_error_inherits_from_ai_agent_error():
    from src.exceptions.AIAgentExceptions import AIAgentError, LLMTimeoutError

    assert issubclass(LLMTimeoutError, AIAgentError)


def test_llm_unavailable_error_inherits_from_llm_connection_error():
    from src.exceptions.AIAgentExceptions import LLMConnectionError, LLMUnavailableError

    assert issubclass(LLMUnavailableError, LLMConnectionError)


def test_ai_agent_config_error_inherits_from_ai_agent_error():
    from src.exceptions.AIAgentExceptions import AIAgentConfigError, AIAgentError

    assert issubclass(AIAgentConfigError, AIAgentError)


def test_llm_authentication_error_inherits_from_ai_agent_config_error():
    from src.exceptions.AIAgentExceptions import AIAgentConfigError, LLMAuthenticationError

    assert issubclass(LLMAuthenticationError, AIAgentConfigError)


def test_model_not_found_error_inherits_from_ai_agent_config_error():
    from src.exceptions.AIAgentExceptions import AIAgentConfigError, ModelNotFoundError

    assert issubclass(ModelNotFoundError, AIAgentConfigError)


def test_llm_response_error_inherits_from_ai_agent_error():
    from src.exceptions.AIAgentExceptions import AIAgentError, LLMResponseError

    assert issubclass(LLMResponseError, AIAgentError)


def test_ai_response_parse_error_inherits_from_llm_response_error():
    from src.exceptions.AIAgentExceptions import AIResponseParseError, LLMResponseError

    assert issubclass(AIResponseParseError, LLMResponseError)


def test_context_window_error_inherits_from_llm_response_error():
    from src.exceptions.AIAgentExceptions import ContextWindowError, LLMResponseError

    assert issubclass(ContextWindowError, LLMResponseError)


def test_ai_context_error_inherits_from_ai_agent_error():
    from src.exceptions.AIAgentExceptions import AIAgentError, AIContextError

    assert issubclass(AIContextError, AIAgentError)


def test_missing_context_error_inherits_from_ai_context_error():
    from src.exceptions.AIAgentExceptions import AIContextError, MissingContextError

    assert issubclass(MissingContextError, AIContextError)


def test_context_serialization_error_inherits_from_ai_context_error():
    from src.exceptions.AIAgentExceptions import AIContextError, ContextSerializationError

    assert issubclass(ContextSerializationError, AIContextError)


def test_ai_output_error_inherits_from_ai_agent_error():
    from src.exceptions.AIAgentExceptions import AIAgentError, AIOutputError

    assert issubclass(AIOutputError, AIAgentError)


def test_ai_agent_exceptions_all_carry_messages():
    """All exception classes must correctly propagate their message."""
    from src.exceptions.AIAgentExceptions import (
        AIAgentError, LLMConnectionError, LLMTimeoutError, LLMUnavailableError,
        AIAgentConfigError, LLMAuthenticationError, ModelNotFoundError,
        LLMResponseError, AIResponseParseError, ContextWindowError,
        AIContextError, MissingContextError, ContextSerializationError,
        AIOutputError,
    )

    classes = [
        AIAgentError, LLMConnectionError, LLMTimeoutError, LLMUnavailableError,
        AIAgentConfigError, LLMAuthenticationError, ModelNotFoundError,
        LLMResponseError, AIResponseParseError, ContextWindowError,
        AIContextError, MissingContextError, ContextSerializationError,
        AIOutputError,
    ]
    for cls in classes:
        exc = cls("test msg")
        assert "test msg" in str(exc), f"{cls.__name__} did not propagate message"


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
    from src.exceptions.FetchersExceptions import FetcherError, NetworkError

    assert issubclass(NetworkError, FetcherError)
    assert issubclass(NetworkError, Exception)


def test_timeout_error_inherits_from_fetcher_error():
    """Test TimeoutError is subclass of FetcherError."""
    from src.exceptions.FetchersExceptions import FetcherError, TimeoutError

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
    from src.exceptions.FetchersExceptions import APIError, AuthenticationError

    assert issubclass(AuthenticationError, APIError)


def test_rate_limit_error_inherits_from_api_error():
    """Test RateLimitError is subclass of APIError."""
    from src.exceptions.FetchersExceptions import APIError, RateLimitError

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
    from src.exceptions.FetchersExceptions import CacheError, CacheMissError

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
    from src.exceptions.FetchersExceptions import FetcherConfigError, MissingAPIKeyError

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
    from src.exceptions.LoggerExceptions import LogFileError, LogFileNotFoundError

    assert issubclass(LogFileNotFoundError, LogFileError)


def test_log_file_permission_error_inherits_from_log_file_error():
    """Test LogFilePermissionError is subclass of LogFileError."""
    from src.exceptions.LoggerExceptions import LogFileError, LogFilePermissionError

    assert issubclass(LogFilePermissionError, LogFileError)


def test_log_file_write_error_inherits_from_log_file_error():
    """Test LogFileWriteError is subclass of LogFileError."""
    from src.exceptions.LoggerExceptions import LogFileError, LogFileWriteError

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
    from src.exceptions.LoggerExceptions import HandlerError, InvalidHandlerError

    assert issubclass(InvalidHandlerError, HandlerError)


def test_log_level_error_inherits_from_logger_error():
    """Test LogLevelError is subclass of LoggerError."""
    from src.exceptions.LoggerExceptions import LoggerError, LogLevelError

    assert issubclass(LogLevelError, LoggerError)


def test_logger_timeout_error_inherits_from_logger_error():
    """Test LoggerTimeoutError is subclass of LoggerError."""
    from src.exceptions.LoggerExceptions import LoggerError, LoggerTimeoutError

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
    from src.exceptions.MedallionExceptions import (
        DataPipelineError,
        SystemResourceError,
    )

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
    from src.exceptions.MedallionExceptions import (
        DataCorruptionError,
        DataPipelineError,
    )

    assert issubclass(DataCorruptionError, DataPipelineError)


def test_schema_mismatch_error_inherits_from_data_pipeline_error():
    """Test SchemaMismatchError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import (
        DataPipelineError,
        SchemaMismatchError,
    )

    assert issubclass(SchemaMismatchError, DataPipelineError)


def test_data_validation_error_inherits_from_data_pipeline_error():
    """Test DataValidationError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import (
        DataPipelineError,
        DataValidationError,
    )

    assert issubclass(DataValidationError, DataPipelineError)


def test_imputation_error_inherits_from_data_pipeline_error():
    """Test ImputationError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import DataPipelineError, ImputationError

    assert issubclass(ImputationError, DataPipelineError)


def test_standardization_error_inherits_from_data_pipeline_error():
    """Test StandardizationError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import (
        DataPipelineError,
        StandardizationError,
    )

    assert issubclass(StandardizationError, DataPipelineError)


def test_outlier_detection_error_inherits_from_data_pipeline_error():
    """Test OutlierDetectionError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import (
        DataPipelineError,
        OutlierDetectionError,
    )

    assert issubclass(OutlierDetectionError, DataPipelineError)


def test_parallel_execution_error_inherits_from_data_pipeline_error():
    """Test ParallelExecutionError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import (
        DataPipelineError,
        ParallelExecutionError,
    )

    assert issubclass(ParallelExecutionError, DataPipelineError)


def test_thread_pool_error_inherits_from_parallel_execution_error():
    """Test ThreadPoolError is subclass of ParallelExecutionError."""
    from src.exceptions.MedallionExceptions import (
        ParallelExecutionError,
        ThreadPoolError,
    )

    assert issubclass(ThreadPoolError, ParallelExecutionError)


def test_analysis_error_inherits_from_data_pipeline_error():
    """Test AnalysisError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import AnalysisError, DataPipelineError

    assert issubclass(AnalysisError, DataPipelineError)


def test_monte_carlo_error_inherits_from_analysis_error():
    """Test MonteCarloError is subclass of AnalysisError."""
    from src.exceptions.MedallionExceptions import AnalysisError, MonteCarloError

    assert issubclass(MonteCarloError, AnalysisError)


def test_regression_error_inherits_from_analysis_error():
    """Test RegressionError is subclass of AnalysisError."""
    from src.exceptions.MedallionExceptions import AnalysisError, RegressionError

    assert issubclass(RegressionError, AnalysisError)


def test_compliance_violation_error_inherits_from_data_pipeline_error():
    """Test ComplianceViolationError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import (
        ComplianceViolationError,
        DataPipelineError,
    )

    assert issubclass(ComplianceViolationError, DataPipelineError)


def test_timeout_error_inherits_from_data_pipeline_error():
    """Test TimeoutError is subclass of DataPipelineError."""
    from src.exceptions.MedallionExceptions import DataPipelineError, TimeoutError

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
    from src.exceptions.StreamlitExceptions import (
        PipelineExecutionError,
        StreamlitError,
    )

    assert issubclass(PipelineExecutionError, StreamlitError)


# ============================================================================
# META COVERAGE TESTS (ANTI-STALE GUARDRAILS)
# ============================================================================


@pytest.mark.parametrize(
    "module_name,base_name",
    [
        ("src.exceptions.FetchersExceptions", "FetcherError"),
        ("src.exceptions.LoggerExceptions", "LoggerError"),
        ("src.exceptions.MedallionExceptions", "DataPipelineError"),
        ("src.exceptions.StreamlitExceptions", "StreamlitError"),
    ],
)
def test_all_custom_exceptions_in_module_inherit_from_declared_base(
    module_name, base_name
):
    """Fail fast if new exception classes are added with wrong inheritance."""
    module = __import__(module_name, fromlist=[base_name])
    base_cls = getattr(module, base_name)
    exception_classes = [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__ and issubclass(cls, Exception)
    ]

    assert exception_classes, f"No exception classes found in {module_name}"
    for cls in exception_classes:
        if cls is base_cls:
            continue
        assert issubclass(cls, base_cls), (
            f"{module_name}.{cls.__name__} must inherit from {base_name}"
        )


def test_custom_exception_message_roundtrip():
    """Ensure custom exceptions preserve human-readable messages."""
    from src.exceptions.MedallionExceptions import ComplianceViolationError

    msg = "guardrail breached"
    exc = ComplianceViolationError(msg)
    assert str(exc) == msg


def test_pipeline_subprocess_error_inherits_from_pipeline_execution_error():
    """Test PipelineSubprocessError is subclass of PipelineExecutionError."""
    from src.exceptions.StreamlitExceptions import (
        PipelineExecutionError,
        PipelineSubprocessError,
    )

    assert issubclass(PipelineSubprocessError, PipelineExecutionError)


def test_pipeline_progress_tracking_error_inherits_from_pipeline_execution_error():
    """Test PipelineProgressTrackingError is subclass of PipelineExecutionError."""
    from src.exceptions.StreamlitExceptions import (
        PipelineExecutionError,
        PipelineProgressTrackingError,
    )

    assert issubclass(PipelineProgressTrackingError, PipelineExecutionError)


def test_data_file_error_inherits_from_streamlit_error():
    """Test DataFileError is subclass of StreamlitError."""
    from src.exceptions.StreamlitExceptions import DataFileError, StreamlitError

    assert issubclass(DataFileError, StreamlitError)


def test_data_file_not_found_error_inherits_from_data_file_error():
    """Test DataFileNotFoundError is subclass of DataFileError."""
    from src.exceptions.StreamlitExceptions import DataFileError, DataFileNotFoundError

    assert issubclass(DataFileNotFoundError, DataFileError)


def test_json_parse_error_inherits_from_data_file_error():
    """Test JSONParseError is subclass of DataFileError."""
    from src.exceptions.StreamlitExceptions import DataFileError, JSONParseError

    assert issubclass(JSONParseError, DataFileError)


def test_session_error_inherits_from_streamlit_error():
    """Test SessionError is subclass of StreamlitError."""
    from src.exceptions.StreamlitExceptions import SessionError, StreamlitError

    assert issubclass(SessionError, StreamlitError)


def test_session_load_error_inherits_from_session_error():
    """Test SessionLoadError is subclass of SessionError."""
    from src.exceptions.StreamlitExceptions import SessionError, SessionLoadError

    assert issubclass(SessionLoadError, SessionError)


def test_session_snapshot_error_inherits_from_session_error():
    """Test SessionSnapshotError is subclass of SessionError."""
    from src.exceptions.StreamlitExceptions import SessionError, SessionSnapshotError

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
    from src.exceptions.StreamlitExceptions import (
        ConfigurationError,
        RolePermissionError,
    )

    assert issubclass(RolePermissionError, ConfigurationError)


def test_data_processing_error_inherits_from_streamlit_error():
    """Test DataProcessingError is subclass of StreamlitError."""
    from src.exceptions.StreamlitExceptions import DataProcessingError, StreamlitError

    assert issubclass(DataProcessingError, StreamlitError)


def test_health_score_calculation_error_inherits_from_data_processing_error():
    """Test HealthScoreCalculationError is subclass of DataProcessingError."""
    from src.exceptions.StreamlitExceptions import (
        DataProcessingError,
        HealthScoreCalculationError,
    )

    assert issubclass(HealthScoreCalculationError, DataProcessingError)


def test_alert_generation_error_inherits_from_data_processing_error():
    """Test AlertGenerationError is subclass of DataProcessingError."""
    from src.exceptions.StreamlitExceptions import (
        AlertGenerationError,
        DataProcessingError,
    )

    assert issubclass(AlertGenerationError, DataProcessingError)


def test_run_comparison_error_inherits_from_data_processing_error():
    """Test RunComparisonError is subclass of DataProcessingError."""
    from src.exceptions.StreamlitExceptions import (
        DataProcessingError,
        RunComparisonError,
    )

    assert issubclass(RunComparisonError, DataProcessingError)


def test_report_generation_error_inherits_from_data_processing_error():
    """Test ReportGenerationError is subclass of DataProcessingError."""
    from src.exceptions.StreamlitExceptions import (
        DataProcessingError,
        ReportGenerationError,
    )

    assert issubclass(ReportGenerationError, DataProcessingError)


def test_import_error_streamlit_inherits_from_streamlit_error():
    """Test ImportError is subclass of StreamlitError."""
    from src.exceptions.StreamlitExceptions import ImportError as StreamlitImportError
    from src.exceptions.StreamlitExceptions import StreamlitError

    assert issubclass(StreamlitImportError, StreamlitError)


def test_logger_module_error_inherits_from_import_error():
    """Test LoggerModuleError is subclass of ImportError."""
    from src.exceptions.StreamlitExceptions import ImportError as StreamlitImportError
    from src.exceptions.StreamlitExceptions import LoggerModuleError

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
    from src.exceptions.StreamlitExceptions import DataFileReadError, JSONParseError

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
        (
            "src.exceptions.MedallionExceptions",
            "DataValidationError",
            "DataPipelineError",
        ),
        ("src.exceptions.MedallionExceptions", "AnalysisError", "DataPipelineError"),
        (
            "src.exceptions.StreamlitExceptions",
            "PipelineExecutionError",
            "StreamlitError",
        ),
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
