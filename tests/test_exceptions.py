import pytest

from exceptions.MedallionExceptions import (
    AnalysisError,
    DataPipelineError,
    DataValidationError,
)


def test_exception_hierarchy():
    assert issubclass(DataValidationError, DataPipelineError)
    assert issubclass(AnalysisError, DataPipelineError)


def test_exceptions_raise_and_message():
    with pytest.raises(DataValidationError) as exc:
        raise DataValidationError("test validation")
    assert "test validation" in str(exc.value)
