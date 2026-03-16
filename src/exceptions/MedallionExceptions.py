"""
Custom Exceptions for Data Pipeline.
Used across Bronze, Silver, and Gold layers for specific error handling.
Ordered from most critical to least critical.
"""


class DataPipelineError(Exception):
    """Base exception for all data pipeline errors."""

    pass


# Critical Infrastructure Errors
class SystemResourceError(DataPipelineError):
    """Raised when system resources are exhausted (e.g., memory, disk space)."""

    pass


class DiskSpaceError(SystemResourceError):
    """Raised when insufficient disk space for data storage."""

    pass


class MemoryError(SystemResourceError):
    """Raised when out of memory during processing."""

    pass


class NetworkError(DataPipelineError):
    """Raised for network connectivity issues in distributed processing."""

    pass


# Data Integrity and Validation Errors
class DataCorruptionError(DataPipelineError):
    """Raised when data files are corrupted or unreadable."""

    pass


class SchemaMismatchError(DataPipelineError):
    """Raised when data doesn't match expected schema."""

    pass


class CatalogNotFoundError(DataPipelineError):
    """Raised when catalog.json is missing."""

    pass


class DataValidationError(DataPipelineError):
    """Raised when data fails schema validation."""

    pass


# Processing Errors
class ImputationError(DataPipelineError):
    """Raised when imputation fails."""

    pass


class StandardizationError(DataPipelineError):
    """Raised when data standardization fails."""

    pass


class OutlierDetectionError(DataPipelineError):
    """Raised when outlier detection or winsorization fails."""

    pass


class TransformationError(DataPipelineError):
    """Raised during data transformation (e.g., log returns)."""

    pass


# File and I/O Errors
class FileNotFoundError(DataPipelineError):
    """Raised when required files are missing."""

    pass


class FileSaveError(DataPipelineError):
    """Raised when saving processed data fails."""

    pass


class PermissionError(DataPipelineError):
    """Raised when file permissions prevent access."""

    pass


class PathError(DataPipelineError):
    """Raised for invalid or non-existent paths."""

    pass


# Parallel Execution Errors
class ParallelExecutionError(DataPipelineError):
    """Raised when parallel tasks fail."""

    pass


class ThreadPoolError(ParallelExecutionError):
    """Raised for thread pool issues."""

    pass


class ProcessPoolError(ParallelExecutionError):
    """Raised for process pool issues."""

    pass


# Analysis-Specific Errors
class AnalysisError(DataPipelineError):
    """Base for analysis errors."""

    pass


class MonteCarloError(AnalysisError):
    """Raised in Monte Carlo simulations."""

    pass


class RegressionError(AnalysisError):
    """Raised in regression analyses."""

    pass


class ForecastingError(AnalysisError):
    """Raised in time series forecasting."""

    pass


# Compliance and Security Errors
class ComplianceViolationError(DataPipelineError):
    """Raised when data violates business compliance rules."""

    pass


class EncryptionError(DataPipelineError):
    """Raised when data encryption/decryption fails."""

    pass


class DecryptionError(EncryptionError):
    """Raised when data decryption fails."""

    pass


# Cloud and External Service Errors
class CloudUploadError(DataPipelineError):
    """Raised when uploading to cloud fails."""

    pass


class CloudDownloadError(DataPipelineError):
    """Raised when downloading from cloud fails."""

    pass


class ExternalServiceError(DataPipelineError):
    """Raised for errors in external services (e.g., APIs)."""

    pass


# Configuration Errors
class ConfigurationError(DataPipelineError):
    """Raised for invalid configuration."""

    pass


class MissingConfigError(ConfigurationError):
    """Raised when required config is missing."""

    pass


# Miscellaneous Errors
class TimeoutError(DataPipelineError):
    """Raised when operations timeout."""

    pass


class UnexpectedError(DataPipelineError):
    """Raised for unexpected errors."""

    pass
