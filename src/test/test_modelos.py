from modelos.prophet_model import ProphetModel
from modelos.arima_model import ARIMAModel

def test_prophet_predict():
    modelo = ProphetModel("test_prophet")
    modelo.fit("datos simulados")
    df = modelo.predict(3)
    assert df.shape[0] == 3

def test_arima_evaluate():
    modelo = ARIMAModel("test_arima")
    modelo.fit("datos simulados")
    resultado = modelo.evaluate()
    assert "MAE" in resultado
