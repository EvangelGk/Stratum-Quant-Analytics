import json
import logging
import threading
import time
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


class ApplicationCatalog:
    """
    Comprehensive logging and metrics catalog for the entire Scenario Planner
    application.
    Tracks all operations, data flows, analyses, and system metrics.

    This class is designed for high reliability and observability:
      - Thread-safe recording of metrics
      - Rotating JSON log files
      - Structured session summaries
      - Full operation timeline for post-mortem analysis
    """

    def __init__(
        self,
        log_file: str = "application_catalog.log",
        max_bytes: int = 5_242_880,
        backup_count: int = 3,
    ) -> None:
        self.log_file = Path("logs") / log_file
        self.log_file.parent.mkdir(exist_ok=True)

        self.lock = threading.Lock()
        self.session_id = str(uuid.uuid4())

        # Setup structured logging (JSON logs at file path)
        self.logger = logging.getLogger("ApplicationCatalog")
        self.logger.setLevel(logging.INFO)

        # Ensure we can log without specifying extra fields every time
        class DefaultExtraFilter(logging.Filter):
            def __init__(self, session_id: str):
                super().__init__()
                self.session_id = session_id

            def filter(self, record: logging.LogRecord) -> bool:
                if not hasattr(record, "session_id"):
                    record.session_id = self.session_id
                if not hasattr(record, "operation"):
                    record.operation = "unknown"
                if not hasattr(record, "metrics"):
                    record.metrics = "{}"
                if not hasattr(record, "details"):
                    record.details = "{}"
                return True

        self.logger.addFilter(DefaultExtraFilter(self.session_id))

        # Avoid adding duplicate handlers if instantiated more than once
        if not self.logger.handlers:
            # Rotating file handler to prevent log files from growing indefinitely
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(
                logging.Formatter(
                    '{"timestamp": "%(asctime)s", "session_id": '
                    '"%(session_id)s", "level": "%(levelname)s", '
                    '"component": "%(name)s", "operation": '
                    '"%(operation)s", "metrics": %(metrics)s, '
                    '"details": %(details)s, "message": "%(message)s"}'
                )
            )
            self.logger.addHandler(file_handler)

            # Console handler for user feedback (plain text)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(operation)s - %(message)s")
            )
            self.logger.addHandler(console_handler)

        # Metrics storage
        self.session_metrics: Dict[str, Any] = {
            "session_id": self.session_id,
            "session_start": datetime.now().isoformat(),
            "operations": [],
            "data_metrics": {},
            "performance_metrics": {},
            "error_metrics": {},
        }

    def log_operation(
        self,
        operation: str,
        component: str,
        metrics: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        message: str = "",
    ) -> None:
        """Log an operation with structured metrics and details.

        This function is thread-safe and appends operations into an in-memory timeline.
        If called from multiple threads, it ensures consistent session tracking.

        Args:
            operation: Operation name (e.g., 'data_fetch', 'analysis_run')
            component: Component name (e.g., 'bronze', 'gold', 'fetcher')
            metrics: Quantitative metrics (e.g., {'records': 1000,
                'duration': 5.2})
            details: Qualitative details (e.g., {'source': 'yfinance',
                'ticker': 'AAPL'})
            message: Human-readable message
        """
        extra = {
            "session_id": self.session_id,
            "operation": operation,
            "metrics": json.dumps(metrics or {}),
            "details": json.dumps(details or {}),
        }

        # Log to output (file + console)
        self.logger.info(message, extra=extra)

        # Store in session metrics (thread-safe)
        with self.lock:
            self.session_metrics["operations"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "operation": operation,
                    "component": component,
                    "metrics": metrics or {},
                    "details": details or {},
                }
            )

    def log_data_operation(
        self,
        operation: str,
        source: str,
        records: int = 0,
        files: int = 0,
        duration: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Log data-related operations with specific metrics."""
        metrics = {
            "records_processed": records,
            "files_processed": files,
            "duration_seconds": duration,
            "success": success,
        }

        details = {"source": source, "error": error}

        message = (
            f"Data {operation}: {source} - {records} records, {files} "
            f"files in {duration:.2f}s"
        )

        self.log_operation(
            f"data_{operation}", "data_pipeline", metrics, details, message
        )

    def log_analysis_operation(
        self,
        analysis_type: str,
        target: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        duration: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Log analysis operations with specific metrics."""
        analysis_metrics = {
            "duration_seconds": duration,
            "success": success,
            "analysis_type": analysis_type,
            **(metrics or {}),
        }

        details = {"target": target, "error": error}

        message = (
            f"Analysis {analysis_type}: {target or 'N/A'} completed in {duration:.2f}s"
        )

        self.log_operation(
            "analysis_run", "gold_layer", analysis_metrics, details, message
        )

    def log_system_metrics(
        self,
        component: str,
        cpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None,
        disk_usage: Optional[float] = None,
        network_requests: Optional[int] = None,
    ) -> None:
        """Log system performance metrics."""
        metrics = {
            "cpu_percent": cpu_usage,
            "memory_percent": memory_usage,
            "disk_percent": disk_usage,
            "network_requests": network_requests,
        }

        # Filter out None values
        metrics = {k: v for k, v in metrics.items() if v is not None}

        self.log_operation(
            "system_metrics", component, metrics, {}, f"System metrics for {component}"
        )

    def log_error(
        self,
        component: str,
        error_type: str,
        error_message: str,
        operation: Optional[str] = None,
    ) -> None:
        """Log errors with categorization."""
        metrics = {"error_count": 1, "error_type": error_type}

        details = {"error_message": error_message, "operation": operation}

        self.log_operation(
            "error_occurred",
            component,
            metrics,
            details,
            f"Error in {component}: {error_type}",
        )

        # Update error metrics (thread-safe)
        with self.lock:
            if error_type not in self.session_metrics["error_metrics"]:
                self.session_metrics["error_metrics"][error_type] = 0
            self.session_metrics["error_metrics"][error_type] += 1

    def save_session_summary(self, include_timeline: bool = True) -> Path:
        """Save comprehensive session summary with all metrics.

        The output is a JSON file that contains:
          - Session metadata (start/end time, session id)
          - Aggregated data + analysis metrics
          - Error breakdown
          - (Optional) full operation timeline
          - Environment and configuration snapshot

        Returns:
            Path to the saved summary file.
        """
        summary_file = (
            self.log_file.parent
            / f"session_summary_{self.session_id}_{int(time.time())}.json"
        )

        with self.lock:
            # Calculate session totals
            total_operations = len(self.session_metrics["operations"])
            total_duration = sum(
                op.get("metrics", {}).get("duration_seconds", 0)
                for op in self.session_metrics["operations"]
            )

            # Data processing summary
            data_ops = [
                op
                for op in self.session_metrics["operations"]
                if op["operation"].startswith("data_")
            ]
            total_records = sum(
                op.get("metrics", {}).get("records_processed", 0) for op in data_ops
            )
            total_files = sum(
                op.get("metrics", {}).get("files_processed", 0) for op in data_ops
            )

            # Analysis summary
            analysis_ops = [
                op
                for op in self.session_metrics["operations"]
                if op["operation"] == "analysis_run"
            ]
            successful_analyses = sum(
                1 for op in analysis_ops if op.get("metrics", {}).get("success", False)
            )

            summary = {
                "session_info": {
                    "session_id": self.session_id,
                    "start_time": self.session_metrics["session_start"],
                    "end_time": datetime.now().isoformat(),
                    "total_duration_seconds": total_duration,
                    "total_operations": total_operations,
                },
                "data_metrics": {
                    "total_records_processed": total_records,
                    "total_files_processed": total_files,
                    "data_sources_used": list(
                        set(
                            op.get("details", {}).get("source")
                            for op in data_ops
                            if op.get("details", {}).get("source")
                        )
                    ),
                },
                "analysis_metrics": {
                    "total_analyses_run": len(analysis_ops),
                    "successful_analyses": successful_analyses,
                    "analysis_types": list(
                        set(
                            op.get("metrics", {}).get("analysis_type")
                            for op in analysis_ops
                        )
                    ),
                },
                "error_metrics": self.session_metrics["error_metrics"],
                "performance_metrics": {
                    "average_operation_duration": total_duration / total_operations
                    if total_operations > 0
                    else 0,
                    "operations_per_second": total_operations / total_duration
                    if total_duration > 0
                    else 0,
                },
                "environment": self._collect_environment_snapshot(),
            }

            if include_timeline:
                summary["operations_timeline"] = self.session_metrics["operations"]

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Session summary saved to {summary_file}")
        return summary_file

    def _collect_environment_snapshot(self) -> Dict[str, Any]:
        """Collect runtime environment details for debugging and reproducibility."""
        import os
        import platform

        snapshot = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cwd": os.getcwd(),
            "environment_variables": {
                k: os.environ.get(k)
                for k in ["PYTHONPATH"]
                if os.environ.get(k)
            },
            "timestamp": datetime.now().isoformat(),
        }
        return snapshot

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current session metrics summary."""
        with self.lock:
            return {
                "total_operations": len(self.session_metrics["operations"]),
                "error_count": sum(self.session_metrics["error_metrics"].values()),
                "data_processed": sum(
                    op.get("metrics", {}).get("records_processed", 0)
                    for op in self.session_metrics["operations"]
                ),
                "analyses_completed": sum(
                    1
                    for op in self.session_metrics["operations"]
                    if op["operation"] == "analysis_run"
                    and op.get("metrics", {}).get("success", False)
                ),
            }


# Global catalog instance
catalog = ApplicationCatalog()
