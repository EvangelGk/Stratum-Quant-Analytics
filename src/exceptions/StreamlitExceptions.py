"""
Custom Exceptions for Streamlit UI.
Ordered from most critical to least critical.
"""


class StreamlitError(Exception):
    """Base exception for all Streamlit UI errors."""

    pass


# Pipeline Execution Errors
class PipelineExecutionError(StreamlitError):
    """Raised when pipeline execution fails."""

    pass


class PipelineTimeoutError(PipelineExecutionError):
    """Raised when pipeline execution times out."""

    pass


class PipelineSubprocessError(PipelineExecutionError):
    """Raised for subprocess-related issues during pipeline execution."""

    pass


class PipelineProgressTrackingError(PipelineExecutionError):
    """Raised when progress bar tracking fails."""

    pass


# Data File and I/O Errors
class DataFileError(StreamlitError):
    """Raised for data file issues in UI."""

    pass


class DataFileNotFoundError(DataFileError):
    """Raised when required data file is missing."""

    pass


class DataFileReadError(DataFileError):
    """Raised when reading data file fails."""

    pass


class DataFileWriteError(DataFileError):
    """Raised when writing data file fails."""

    pass


class JSONParseError(DataFileError):
    """Raised when JSON parsing fails."""

    pass


class JSONSerializationError(DataFileError):
    """Raised when JSON serialization fails."""

    pass


# Session and State Errors
class SessionError(StreamlitError):
    """Raised for session state issues."""

    pass


class SessionLoadError(SessionError):
    """Raised when loading session history fails."""

    pass


class SessionSnapshotError(SessionError):
    """Raised when recording UI snapshot fails."""

    pass


class SessionExpiredError(SessionError):
    """Raised when session expires or becomes invalid."""

    pass


# Display and Rendering Errors
class DisplayError(StreamlitError):
    """Raised for rendering/display issues."""

    pass


class ChartRenderError(DisplayError):
    """Raised when chart rendering fails."""

    pass


class TabRenderError(DisplayError):
    """Raised when tab rendering fails."""

    pass


class ComponentRenderError(DisplayError):
    """Raised when UI component rendering fails."""

    pass


# Configuration and State Errors
class ConfigurationError(StreamlitError):
    """Raised for configuration issues."""

    pass


class RolePermissionError(ConfigurationError):
    """Raised when role-based permission check fails."""

    pass


class UIConfigError(ConfigurationError):
    """Raised for UI configuration problems."""

    pass


# Data Processing and Calculation Errors
class DataProcessingError(StreamlitError):
    """Raised when data processing fails in UI."""

    pass


class HealthScoreCalculationError(DataProcessingError):
    """Raised when health score calculation fails."""

    pass


class AlertGenerationError(DataProcessingError):
    """Raised when smart alert generation fails."""

    pass


class RunComparisonError(DataProcessingError):
    """Raised when run comparison fails."""

    pass


class ReportGenerationError(DataProcessingError):
    """Raised when report generation fails."""

    pass


# Import and Module Errors
class ImportError(StreamlitError):
    """Raised when module import fails."""

    pass


class LoggerModuleError(ImportError):
    """Raised when logger module import fails."""

    pass


class DependencyMissingError(ImportError):
    """Raised when required dependency is missing."""

    pass


# User Interaction Errors
class UserInputError(StreamlitError):
    """Raised for invalid user input."""

    pass


class InvalidParameterError(UserInputError):
    """Raised when user provides invalid parameters."""

    pass


class ValidationError(UserInputError):
    """Raised when input validation fails."""

    pass


# Miscellaneous Errors
class UnexpectedStreamlitError(StreamlitError):
    """Raised for unexpected Streamlit errors."""

    pass
