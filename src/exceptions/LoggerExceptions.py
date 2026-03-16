"""
Custom Exceptions for Logging System.
Ordered from most critical to least critical.
"""


class LoggerError(Exception):
    """Base exception for all logging errors."""

    pass


# Critical File and I/O Errors
class LogFileError(LoggerError):
    """Raised for log file issues."""

    pass


class LogFileNotFoundError(LogFileError):
    """Raised when log file is missing."""

    pass


class LogFilePermissionError(LogFileError):
    """Raised when log file permissions prevent access."""

    pass


class LogFileWriteError(LogFileError):
    """Raised when writing to log file fails."""

    pass


class DiskFullError(LogFileError):
    """Raised when disk is full for logging."""

    pass


# Handler and Configuration Errors
class HandlerError(LoggerError):
    """Raised for logging handler issues."""

    pass


class InvalidHandlerError(HandlerError):
    """Raised for invalid logging handler."""

    pass


class HandlerConfigError(HandlerError):
    """Raised for handler configuration errors."""

    pass


class StreamHandlerError(HandlerError):
    """Raised for console/stream logging issues."""

    pass


class FileHandlerError(HandlerError):
    """Raised for file logging issues."""

    pass


# Log Level and Message Errors
class LogLevelError(LoggerError):
    """Raised for invalid log levels."""

    pass


class InvalidLogLevelError(LogLevelError):
    """Raised when log level is invalid."""

    pass


class LogMessageError(LoggerError):
    """Raised for log message issues."""

    pass


class EmptyLogMessageError(LogMessageError):
    """Raised for empty log messages."""

    pass


class LogFormatError(LogMessageError):
    """Raised for invalid log format."""

    pass


# External Logging Errors
class ExternalLoggerError(LoggerError):
    """Raised for external logging service issues."""

    pass


class RemoteLoggerError(ExternalLoggerError):
    """Raised for remote logging failures."""

    pass


class CloudLoggerError(ExternalLoggerError):
    """Raised for cloud logging failures."""

    pass


# Configuration Errors
class LoggerConfigError(LoggerError):
    """Raised for logger configuration issues."""

    pass


class MissingLoggerConfigError(LoggerConfigError):
    """Raised when logger config is missing."""

    pass


class InvalidLoggerConfigError(LoggerConfigError):
    """Raised for invalid logger config."""

    pass


# Miscellaneous Errors
class LoggerTimeoutError(LoggerError):
    """Raised when logging operations timeout."""

    pass


class UnexpectedLoggerError(LoggerError):
    """Raised for unexpected logging errors."""

    pass
