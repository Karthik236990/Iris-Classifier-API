FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model at image build time so the container starts ready to serve.
# (For larger real-world models you'd instead pull a pre-trained artifact
# from object storage here rather than training inside the image.)
RUN python -m src.data.make_dataset && \
    python -m src.features.build_features && \
    python -m src.models.train_model

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
