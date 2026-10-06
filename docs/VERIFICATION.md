# Kiểm chứng triển khai — 2026-10-06

Môi trường máy kiểm tra: Windows, CPU, **không có CUDA**. Python3.12.12 trong
hai virtualenv, torch2.6.0+cpu; modern Transformers4.51.3 / legacy4.44.2.
Đây là kiểm chứng adapter/runner trên CPU, không phải kết quả Colab GPU.

| Kiểm tra thực tế | Trạng thái | Giới hạn |
| --- | --- | --- |
| `scripts/build_dataset.py --check` | 28group /56query /336pairs | Fixture tổng hợp; không thêm nhãn retrieval mới |
| `python -m unittest discover -s tests -q` | 32test passed | Mock API không chứng minh quota/key thật |
| BM25 và TF-IDF full-corpus | Mỗi bên56/56 complete/full | 168documentIDs |
| MiniLM multilingual dense retrieval | 56/56 complete/full | Native cap128, CPU/FP32 |
| BM25 + MiniLM RRF | 56/56 complete/full | Parent hashes và metric đã validate |
| mMARCO rerank từ BM25 top20 | 56/56 complete/full | CPU/FP32, cap512 |
| mMARCO rerank từ RRF-MiniLM top20 | 56/56 complete/full | Như trên |
| mMARCO oracle6doc | 56/56 complete/full | Không phải end-to-end retrieval |
| Qwen3 Reranker0.6B | 4/4 smoke complete, full_dataset=false | Không được dùng làm full benchmark |
| GTE multilingual pinned cross-repo code | 4/4 smoke complete, full_dataset=false | Legacy4.44.2, không xformers |
| Jina v2 pinned custom code | 4/4 smoke complete, full_dataset=false | Legacy4.44.2, einops, flash tắt |
| Suite reporter | Xuất7 validated run artifacts | MD, summary/per-query/breakdown CSV, bootstrap CI, manifest |
| Notebook | Mọi cell Python + GPU subprocess program compile | Chưa thực thi notebook trong Colab thật |
| Qwen4B/8B, Gemma9B; embedding còn lại | Source/ID/revision đối chiếu | Chưa inference-verified ở lượt này |
| Cohere3model | Retry/index/pacing/redaction test passed | Không có key để chạy API thật |

Artifact kiểm tra nằm ở `.research/pinned-validation/`,
`.research/qwen-pinned-smoke/`, `.research/gte-pinned-smoke/`,
`.research/jina-pinned-smoke/` trên máy thực hiện; là thư mục ignored và
**không** giả vờ các artifact này đã có trên Colab của người đọc.
Không commit trọng số hay key. Những model không kiểm tra thật không có
score so sánh được công bố trong lượt này. Kết quả GPU phải lấy từ ZIP
do notebook chạy trên GPU của bạn xuất ra.

Lệnh kiểm tra mẫu (PowerShell):

```powershell
.\.venv-validation-modern\Scripts\python.exe -m benchmark.suite retrieve --profile cpu --embedders minilm-multi --device cpu --dtype float32 --local-files-only --output .research/pinned-validation --resume
.\.venv-validation-modern\Scripts\python.exe -m benchmark.suite rerank --profile cpu --models mmarco-minilm --pools bm25 rrf-minilm-multi --device cpu --dtype float32 --local-files-only --output .research/pinned-validation --resume
.\.venv-validation-modern\Scripts\python.exe -m benchmark.suite report --output .research/pinned-validation
```

`--local-files-only` ở các lệnh này dùng cache trên máy kiểm tra. Người đọc
chưa có weights cần bỏ flag đó; không mặc nhiên chạy được offline. JSON
của bài full được reporter kiểm tra query selection, corpus/dataset hashes,
ranking/metric, candidate pool và parent hashes trước xuất báo cáo.
