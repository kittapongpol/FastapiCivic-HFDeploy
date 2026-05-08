# Honda Civic Generation Image Classification API

โปรเจคนี้ทำตามโจทย์ Project Assignment เรื่อง High-Throughput Image Classification Service โดยเลือกหัวข้อจำแนกรุ่น Honda Civic 8 คลาส ได้แก่ `EG`, `EK`, `ES`, `FD`, `FB`, `FC`, `FE`, `FK`

## Labels

| Label | Generation | Approx. year range | จุดสังเกตหลัก |
| --- | --- | --- | --- |
| EG | 5th gen | 1991-1995 | ตัวถังเตี้ย ทรงโค้ง ไฟหน้าเรียว |
| EK | 6th gen | 1995-2000 | เส้นตัวถังเรียบ ไฟหน้าโตขึ้น |
| ES | 7th gen | 2000-2005 | ทรงซีดานสูงขึ้น กระจังหน้าเรียบ |
| FD | 8th gen | 2005-2011 | หน้าปัดสองชั้น ทรงลิ่ม ไฟหน้าคม |
| FB | 9th gen | 2011-2015 | เส้นข้างเรียบ ไฟท้ายกว้าง |
| FC | 10th gen | 2015-2021 | ทรง fastback ไฟหน้า LED คม |
| FE | 11th gen | 2021-present | เส้นตัวถังสุภาพ กระจังหน้าแนวนอน |
| FK | 10th gen hatchback | 2016-2021 | ตัวถัง hatchback ท้ายสั้น เส้นหลังคาลาด |

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
