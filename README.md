---
title: FastapiCivic
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
# Honda Civic Generation Image Classification API

à¹‚à¸›à¸£à¹€à¸ˆà¸„à¸™à¸µà¹‰à¸—à¸³à¸•à¸²à¸¡à¹‚à¸ˆà¸—à¸¢à¹Œ Project Assignment à¹€à¸£à¸·à¹ˆà¸­à¸‡ High-Throughput Image Classification Service à¹‚à¸”à¸¢à¹€à¸¥à¸·à¸­à¸à¸«à¸±à¸§à¸‚à¹‰à¸­à¸ˆà¸³à¹à¸™à¸à¸£à¸¸à¹ˆà¸™ Honda Civic 8 à¸„à¸¥à¸²à¸ª à¹„à¸”à¹‰à¹à¸à¹ˆ `EG`, `EK`, `ES`, `FD`, `FB`, `FC`, `FE`, `FK`

## Labels

| Label | Generation | Approx. year range | à¸ˆà¸¸à¸”à¸ªà¸±à¸‡à¹€à¸à¸•à¸«à¸¥à¸±à¸ |
| --- | --- | --- | --- |
| EG | 5th gen | 1991-1995 | à¸•à¸±à¸§à¸–à¸±à¸‡à¹€à¸•à¸µà¹‰à¸¢ à¸—à¸£à¸‡à¹‚à¸„à¹‰à¸‡ à¹„à¸Ÿà¸«à¸™à¹‰à¸²à¹€à¸£à¸µà¸¢à¸§ |
| EK | 6th gen | 1995-2000 | à¹€à¸ªà¹‰à¸™à¸•à¸±à¸§à¸–à¸±à¸‡à¹€à¸£à¸µà¸¢à¸š à¹„à¸Ÿà¸«à¸™à¹‰à¸²à¹‚à¸•à¸‚à¸¶à¹‰à¸™ |
| ES | 7th gen | 2000-2005 | à¸—à¸£à¸‡à¸‹à¸µà¸”à¸²à¸™à¸ªà¸¹à¸‡à¸‚à¸¶à¹‰à¸™ à¸à¸£à¸°à¸ˆà¸±à¸‡à¸«à¸™à¹‰à¸²à¹€à¸£à¸µà¸¢à¸š |
| FD | 8th gen | 2005-2011 | à¸«à¸™à¹‰à¸²à¸›à¸±à¸”à¸ªà¸­à¸‡à¸Šà¸±à¹‰à¸™ à¸—à¸£à¸‡à¸¥à¸´à¹ˆà¸¡ à¹„à¸Ÿà¸«à¸™à¹‰à¸²à¸„à¸¡ |
| FB | 9th gen | 2011-2015 | à¹€à¸ªà¹‰à¸™à¸‚à¹‰à¸²à¸‡à¹€à¸£à¸µà¸¢à¸š à¹„à¸Ÿà¸—à¹‰à¸²à¸¢à¸à¸§à¹‰à¸²à¸‡ |
| FC | 10th gen | 2015-2021 | à¸—à¸£à¸‡ fastback à¹„à¸Ÿà¸«à¸™à¹‰à¸² LED à¸„à¸¡ |
| FE | 11th gen | 2021-present | à¹€à¸ªà¹‰à¸™à¸•à¸±à¸§à¸–à¸±à¸‡à¸ªà¸¸à¸ à¸²à¸ž à¸à¸£à¸°à¸ˆà¸±à¸‡à¸«à¸™à¹‰à¸²à¹à¸™à¸§à¸™à¸­à¸™ |
| FK | 10th gen hatchback | 2016-2021 | à¸•à¸±à¸§à¸–à¸±à¸‡ hatchback à¸—à¹‰à¸²à¸¢à¸ªà¸±à¹‰à¸™ à¹€à¸ªà¹‰à¸™à¸«à¸¥à¸±à¸‡à¸„à¸²à¸¥à¸²à¸” |

## Project Structure

```text
app/                         FastAPI application
models/                      optimized ONNX model output
scripts/                     export, quantize, benchmark, report rendering scripts
tests/                       pytest unit tests
.github/workflows/ci-cd.yml  CI/CD workflow
artifacts/                   JMeter, Postman, cURL artifacts
docs/                        report and presentation sources
```

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Predict:

```bash
curl.exe -X POST "http://127.0.0.1:8000/predict" --form "file=@samplefc.png"
```

## Docker

```bash
docker build -t civic-gen-classifier .
docker run --rm -p 7860:7860 civic-gen-classifier
```

Cloud cURL example after deploying to Hugging Face Spaces:

```bash
curl.exe -X POST "https://<hf-username>-<space-name>.hf.space/predict" --form "file=@samplefc.png"
```

## Model Optimization Workflow

The selected Hugging Face base model is `google/vit-base-patch16-224`. For a real Civic generation classifier, fine-tune it on a labeled dataset with folders `EG/`, `EK/`, `ES/`, `FD/`, `FB/`, `FC/`, `FE/`, and `FK/`, then export and quantize:

```bash
pip install -r requirements-optimization.txt
python scripts/export_onnx.py --model-id google/vit-base-patch16-224 --output models/civic_gen.onnx
python scripts/quantize.py --input models/civic_gen.onnx --output models/civic_gen_quantized.onnx
python scripts/benchmark.py --image samplefc.png --runs 100
```

If `models/civic_gen_quantized.onnx` is not present, the API uses a deterministic lightweight fallback classifier so endpoint tests and load tests can still run.

## Tests

```bash
pytest -q
```

The GitHub Actions workflow runs tests on every push and pull request, builds the Docker image, and deploys to Hugging Face Spaces when all tests pass on `main`.

