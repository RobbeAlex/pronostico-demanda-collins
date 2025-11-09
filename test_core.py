import pytest
from src.core import DemandForecasterSystem

def test_forecaster_initialization():
    forecaster = DemandForecasterSystem()
    assert forecaster is not None
    assert hasattr(forecaster, 'run_pipeline')

def test_run_pipeline_returns_dict():
    forecaster = DemandForecasterSystem()
    result = forecaster.run_pipeline()
    assert isinstance(result, dict)
