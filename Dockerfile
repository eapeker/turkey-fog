# Fog prediction tool: SEVIRI (+ ERA5-Land) -> night-time fog probability at one point.
# Contains only what prediction needs: code, config, the two deployed models and offline examples.
# No data, no credentials: EUMETSAT and Earth Engine logins are mounted at run time.
FROM python:3.11-slim

WORKDIR /app

COPY requirements-predict.txt .
RUN pip install --no-cache-dir -r requirements-predict.txt

COPY config.yaml pyproject.toml ./
COPY src/ src/
COPY scripts/predict.py scripts/
COPY models/mlp.joblib models/mlp_era5.joblib models/
COPY examples/ examples/

ENTRYPOINT ["python", "scripts/predict.py"]
CMD ["--help"]
