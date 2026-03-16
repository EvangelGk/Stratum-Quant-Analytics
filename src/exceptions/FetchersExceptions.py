"""
Custom Exceptions for Data Fetchers.
Ordered from most critical to least critical.
"""


class FetcherError(Exception):
    """Base exception for all fetcher errors."""

    pass


# Critical Network and Connectivity Errors
class NetworkError(FetcherError):
    """Raised for network connectivity issues."""

    pass


class TimeoutError(FetcherError):
    """Raised when API requests timeout."""

    pass


class ConnectionError(NetworkError):
    """Raised when connection to API fails."""

    pass


# API-Specific Errors
class APIError(FetcherError):
    """Raised for general API errors."""

    pass


class AuthenticationError(APIError):
    """Raised when API authentication fails (e.g., invalid key)."""

    pass


class AuthorizationError(APIError):
    """Raised when API authorization fails."""

    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    pass


class APIUnavailableError(APIError):
    """Raised when API is temporarily unavailable."""

    pass


class InvalidAPIResponseError(APIError):
    """Raised when API returns invalid or malformed response."""

    pass


# Data Fetching Errors
class DataFetchError(FetcherError):
    """Raised when data fetching fails."""

    pass


class InvalidParametersError(DataFetchError):
    """Raised for invalid fetch parameters (e.g., bad ticker)."""

    pass


class NoDataError(DataFetchError):
    """Raised when no data is available for the request."""

    pass


class DataFormatError(DataFetchError):
    """Raised when fetched data has invalid format."""

    pass


# Caching Errors
class CacheError(FetcherError):
    """Raised for caching issues."""

    pass


class CacheMissError(CacheError):
    """Raised when cache lookup fails."""

    pass


class CacheWriteError(CacheError):
    """Raised when writing to cache fails."""

    pass


class CacheExpirationError(CacheError):
    """Raised when cached data is expired."""

    pass


# External Service Errors
class ExternalServiceError(FetcherError):
    """Raised for errors in external services."""

    pass


class ServiceDownError(ExternalServiceError):
    """Raised when external service is down."""

    pass


class ServiceMaintenanceError(ExternalServiceError):
    """Raised during service maintenance."""

    pass


# Configuration Errors
class FetcherConfigError(FetcherError):
    """Raised for fetcher configuration issues."""

    pass


class MissingAPIKeyError(FetcherConfigError):
    """Raised when API key is missing."""

    pass


class InvalidConfigError(FetcherConfigError):
    """Raised for invalid fetcher config."""

    pass


# Miscellaneous Errors
class UnexpectedFetcherError(FetcherError):
    """Raised for unexpected fetcher errors."""

    pass
