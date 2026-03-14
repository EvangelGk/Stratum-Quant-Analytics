"""
Custom Exceptions for Data Pipeline.
Used across Bronze, Silver, and Gold layers for specific error handling.
"""

class DataPipelineError(Exception):
    """Base exception for data pipeline errors."""
    pass

class CatalogNotFoundError(DataPipelineError):
    """Raised when catalog.json is missing."""
    pass

class DataValidationError(DataPipelineError):
    """Raised when data fails schema validation."""
    pass

class ImputationError(DataPipelineError):
    """Raised when imputation fails."""
    pass

class StandardizationError(DataPipelineError):
    """Raised when data standardization fails."""
    pass

class FileSaveError(DataPipelineError):
    """Raised when saving processed data fails."""
    pass

class OutlierDetectionError(DataPipelineError):
    """Raised when outlier detection or winsorization fails."""
    pass

class ComplianceViolationError(DataPipelineError):
    """Raised when data violates business compliance rules (e.g., too many nulls)."""
    pass