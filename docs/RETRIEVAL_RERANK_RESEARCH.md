# Retrieval + reranking: catalog, phương pháp và Colab

Đối chiếu nguồn nhà phát hành ngày **2026-10-06**. Đây là protocol và adapter,
không phải kết quả chạy hết model. Revision cố định nằm trong
[`benchmark/catalog.py`](../benchmark/catalog.py); JSON artifact mới là bằng chứng
model đã hoàn tất. Không sửa các kết quả CPU lịch sử trong README.

## 1. Ba phép đo khác nhau

| Bài đo | Model nhận gì? | Điều cần đọc |
| --- | --- | --- |
| Retrieval | Query; index toàn bộ 168 tài liệu, không nhận grade/rationale | Có tìm được bằng chứng? Recall@K, candidate recall, Hole@K |
| End-to-end | Reranker nhận query và top-K từ một retriever | Có cải thiện thứ hạng trên cùng pool? nDCG/MRR/Strict@1 trước–sau |
| Oracle rerank | Query và sáu tài liệu gốc, có chứa positive nếu answerable | Khả năng phân biệt hard negative khi bỏ qua lỗi retrieval |

Pipeline thực thi: corpus hợp nhất và sắp xếp ID ổn định → index một lần/model
→ score mọi tài liệu/query → lấy top-20 → chấm lại bằng reranker → sort ổn định.
Reranker chỉ nhận chuỗi văn bản, không nhận nhãn, rationale, ID hay score retriever.
Mỗi model chạy trong subprocess để giải phóng RAM/VRAM trước model tiếp theo.

BEIR nhấn mạnh kiểm tra lexical/dense/reranking trên nhiều miền, thay vì suy
rộng từ một dataset. Suite này học cách tách các bước đó, **không tự nhận là BEIR**.
[BEIR paper](https://arxiv.org/abs/2104.08663),
[BEIR evaluation](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py).

## 2. Retrieval: 9 embedding, tổng cộng 20 phương pháp

| CLI name | Checkpoint / phương pháp | Pooling và prompt | Native cap dùng để chặn |
| --- | --- | --- | ---: |
| `bm25` | BM25, Unicode word tokenizer, k1=1.2, b=0.75 | IDF trên corpus, không phải sáu ứng viên | Full text |
| `tfidf` | Sublinear TF, smoothed IDF, cosine | IDF trên corpus | Full text |
| `minilm-multi` | [Multilingual MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | Masked mean + L2 | 128 |
| `mpnet-multi` | [Multilingual MPNet](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) | Masked mean + L2 | 128 |
| `e5-small` | [Multilingual E5 small](https://huggingface.co/intfloat/multilingual-e5-small) | `query: ` / `passage: `, mean + L2 | 512 |
| `e5-base` | [Multilingual E5 base](https://huggingface.co/intfloat/multilingual-e5-base) | Như trên, kể cả query tiếng Việt | 512 |
| `e5-large` | [Multilingual E5 large](https://huggingface.co/intfloat/multilingual-e5-large) | Như trên | 512 |
| `bge-m3-dense` | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | CLS + L2; **dense-only** | 8192 |
| `qwen-embed-0.6b` | [Qwen3 Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Last valid token, left padding, query instruction + L2 | 32768 |
| `qwen-embed-4b` | [Qwen3 Embedding 4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | Như trên | 32768 |
| `qwen-embed-8b` | [Qwen3 Embedding 8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | Như trên | 32768 |
| `rrf-<embedder>` ×9 | BM25 + từng dense retriever | Tổng `1/(60+rank)`, rank bắt đầu từ 1 | Theo hai parent |

Native cap không có nghĩa notebook mặc định chạy hết context: nó dùng
`EMBEDDING_MAX_LENGTH=512`, effective cap = min(requested, native).
MiniLM/MPNet theo cấu hình SentenceTransformer 128, không nhầm với architecture
512. Audit ghi cả token trước/sau truncation. BGE-M3 embedding **khác**
`BAAI/bge-reranker-v2-m3`; sparse và ColBERT của BGE-M3 chưa được triển khai.
Qwen dùng instruction chính thức kiểu retrieval, không tối ưu bằng nhãn test.
[E5 report](https://arxiv.org/abs/2402.05672),
[Qwen report](https://arxiv.org/abs/2506.05176).

RRF chỉ kết hợp thứ hạng, không cộng trực tiếp BM25 score với cosine có thang
khác nhau. Hai parent phải complete, cùng corpus/query/code/config; artifact
fusion giữ SHA cả hai parent. Các tài liệu trùng text nhưng khác ID vẫn là hai
tài liệu trong corpus theo fixture, không tự gộp mất nhãn.

## 3. Reranking: 15 model

| CLI name | Model ID / nguồn | Cách chấm và lưu ý | Môi trường |
| --- | --- | --- | --- |
| `bge-m3` | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Classification logit, multilingual | modern |
| `bge-base` | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | English/Chinese baseline, cap 512 | modern |
| `bge-large` | [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | English/Chinese baseline, cap 512 | modern |
| `msmarco-l6` | [MS MARCO MiniLM L6](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | English baseline, cap 512 | modern |
| `msmarco-l12` | [MS MARCO MiniLM L12](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2) | English baseline, cap 512 | modern |
| `mmarco-minilm` | [mMARCO multilingual MiniLM](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) | Multilingual, classifier cap 512 | modern |
| `qwen0.6b` | [Qwen3 Reranker 0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | Final-token yes–no logit, không generate | modern |
| `qwen4b` | [Qwen3 Reranker 4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | Như trên; GPU lớn | modern |
| `qwen8b` | [Qwen3 Reranker 8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) | Như trên; GPU lớn | modern |
| `gte-multi` | [GTE multilingual reranker](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) | Custom classifier; model/code SHA riêng; không yêu cầu xformers | legacy |
| `jina-v2` | [Jina reranker v2 multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) | Custom classifier, flash attention tắt; cap 1024 của checkpoint; **CC-BY-NC-4.0** | legacy |
| `gemma` | [BGE Gemma2 lightweight](https://huggingface.co/BAAI/bge-reranker-v2.5-gemma2-lightweight) | Gemma2 9B; scalar layerwise head; layer28, compress2, layer24 | legacy |
| `cohere-pro` | `rerank-v4.0-pro` | Hosted vendor score, không lộ revision weights | hosted |
| `cohere-fast` | `rerank-v4.0-fast` | Như trên | hosted |
| `cohere-v3.5` | `rerank-v3.5` | Như trên; đối chứng API thế hệ trước | hosted |

Các model English/Chinese không được quảng cáo thành model Việt tốt: đây là
đối chứng, kết quả VI/EN tách trong `breakdown.csv`. Model class và prompt theo
nhà phát hành; không thay bằng chat model tự chấm điểm. Classification raw logits
và yes–no logits dùng cho xếp hạng, không coi là xác suất đúng đã calibration.
API mapping giữ nguyên index tài liệu; không lấy thứ tự results từ Cohere làm
thứ tự score đầu vào. [Cohere API](https://docs.cohere.com/reference/rerank).

Custom code là code bên ngoài cần chủ động cho phép `--trust-remote-code`; SHA
model và code được pin. GTE trỏ cross-repository tới `Alibaba-NLP/new-impl`,
không dùng nhầm revision weights cho module code. Jina cần `einops`, tắt flash
để không bắt T4 cài flash-attn. Gemma phiên bản mới nhất của Transformers không
tương thích các import cũ: notebook cho nó Python/Transformers riêng.

## 4. Metric và giới hạn nhãn

- Các cutoff 1/3/5/10/20/50/100: nDCG, MRR, Recall, Hole.
- Relevant = grade ≥2; strict = grade3. nDCG có gain `2^grade-1`;
  IDCG dùng **toàn bộ known positives**, kể cả tài liệu bị retrieval bỏ sót.
- `candidate_recall` = số known relevant trong top-K / tổng known relevant.
  Khi giữ nguyên top-K pool, reranker không thể tăng chỉ số này.
- `Hole@K` = phần tài liệu top-K chưa được chấm. Unjudged được tính zero gain
  như **quy ước tính**, không phải kết luận rằng tài liệu đó thực sự irrelevant.
- Core44 query là bảng chất lượng chính; long-context4 và unanswerable8 báo
  riêng. Unanswerable không nhận nDCG/Recall giả bằng 1 hoặc 0.
- Δ nDCG so cùng query/pool trước–sau; bootstrap2000 mẫu ở cấp `group_id`,
  giữ cặp Việt/Anh cùng cụm. Có số query improved/harmed/unchanged.

[Danh mục metric BEIR](https://github.com/beir-cellar/beir/wiki/Metrics-available)
có nDCG/MRR/Recall/Hole, nhưng BEIR/trec_eval dùng quy ước gain khác.
Không đối chiếu trực tiếp giá trị nDCG fixture này với BEIR leaderboard.

**Quan trọng:** fixture chỉ chấm 6/168 ID cho từng query. Retrieval ngoài nhóm
gốc có thể đưa lên tài liệu hữu ích chưa có nhãn. Nhãn unanswerable cũng mới
được xác nhận trong pool gốc, chưa chứng minh corpus toàn cục không có đáp án.
Muốn kết luận production cần chấm thêm top results ngoài pool, giữ test riêng,
nhiều query/người chấm và dataset miền thực. Chạy nhiều model không sửa được
giới hạn đó. Không học threshold từ chính fixture này.

## 5. Chạy toàn bộ trên Colab

[Mở notebook](https://colab.research.google.com/github/Achitechture-suggestive-system/rerank-benchmark/blob/fix/cohere-rate-limit/notebooks/retrieval_rerank_colab.ipynb).
Không cần copy lệnh shell dài: notebook dùng subprocess nên tránh ghép lặp
`python -m unittest ... -m benchmark.run ...` hay lỗi `export` khác process.

1. Runtime → Change runtime type → GPU. Dùng A100 cho bản đầy đủ; T415GB
   không đủ Gemma9B FP16. Mốc ≥24GiB chỉ là điều kiện tối thiểu theo thiết kế,
   không phải đo VRAM hay cam kết mọi context fit. A10040GB có dư hơn.
2. Cell đầu đặt `RUN_MODE="full"`, `FULL_MATRIX=True`, giữ
   `RUN_ID` ổn định. Mặc định cap embedding512/reranker1024, top20, batch1.
3. Chạy lần lượt cell: mountDrive/source → môi trường → GPU probe → plan →
   retrieval → reranker modern → GTE/Jina → Gemma → Cohere → report/ZIP.
4. Cohere Secret tên **COHERE_API_KEY**, bật Notebook access; nếu chưa có,
   notebook cho nhập ẩn hoặc để trống để ghi skipped. Không có API key thì
   không thể đo ba API model. Không commit key hoặc dùng key giả.
5. Chạy lại cell lỗi với cùng config/root để resume. Đổi code/data/K/window
   phải dùng RUN_ID mới; checkpoint kiểm tra hash, prefix query, scoring config,
   packages và parent. Không sửa JSON thủ công để chuyển failed thành complete.

Notebook cài `uv==0.9.26`, Python3.12.12, CUDA torch2.6.0 từ index cu124;
modern Transformers4.51.3, legacy4.44.2. Các requirements được pin; package
transitive thực tế lưu theo mỗi môi trường. Python native3.13 không ảnh hưởng.
[uv Python environments](https://docs.astral.sh/uv/concepts/python-versions/),
[PyTorch official cu124 command](https://pytorch.org/get-started/previous-versions/).
Không phụ thuộc runtime lịch sử Colab2025.07 đã quá hạn một năm.
[Colab runtime version policy](https://research.google.com/colaboratory/runtime-version-faq.html).

Cache Hugging Face mặc định là temporary cache **riêng do từng worker tạo**,
được dọn khi worker kết thúc để tránh đầy disk khi tải toàn bộ trọng số.
Không xóa global cache có sẵn. Đặt `KEEP_MODEL_CACHE=True` nếu disk đủ và muốn
giữ weights giữa các phiên model. Ngắt toàn runtime có thể mất cache nhưng
checkpoint trên Drive còn. Notebook không reset/xóa checkout có thay đổi.

### Quy mô full matrix và Cohere

| Cấu hình full GPU | Retrieval run | End-to-end pipelines | Oracle run | Tổng JSON run |
| --- | ---: | ---: | ---: | ---: |
| Local đầy đủ + API mặc định | 20 | 20×12 + 1×3 =243 | 15 | 278 |
| Mọi cặp, `API_ALL_POOLS=True` | 20 | 20×15 =300 | 15 | 335 |

Đây là đếm protocol, **không phải 278/335 kết quả đã chạy**. Default `full`
chạy12 local reranker qua20 retriever; ba API model chỉ RRF-BGE-M3 + oracle.
Mỗi hosted reranker cần113 request (56×2+1 warmup), tổng339 trước retries.
Toàn bộ cặp API cần3531 request (3×(56×21+1)), vượt1000/tháng của trial key.
Trial rerank10 request/phút: pacing6.2s; production quota khác.
[Cohere rate limits](https://docs.cohere.com/v2/docs/rate-limits).

Log cũ dừng sau9 query có thể phù hợp với warmup+9=10 request/phút, nhưng
**không thể xác định HTTP429 chỉ từ stdout**. Cần `reason` trong JSON. Retry
xử lý429/5xx/network với backoff/Retry-After; 401/402 hay hết quota không tự
biến thành chạy thành công. Retries cũng tiêu tốn thời gian/request. Không thể
cam kết Colab miễn phí sống đủ lâu cho335 run; hãy resume hoặc chia phiên.

### Terminal Colab: lệnh tương đương

Chỉ dùng khi đã cài môi trường từ notebook, không dùng `!` ở terminal:

```bash
RUN_ROOT=/content/drive/MyDrive/rerank-benchmark-results/2026-10-06-suite-v1
PY=/content/benchmark-envs/modern/bin/python
LEGACY=/content/benchmark-envs/legacy/bin/python
cd /content/retrieval-rerank-suite

COMMON=(--output "$RUN_ROOT" --device cuda --dtype float16 --batch-size 1 \
  --embedding-batch-size 4 --max-length 1024 --embedding-max-length 512 \
  --candidate-k 20 --threads 4 --allow-large-models --resume)

"$PY" -m benchmark.suite retrieve --profile large "${COMMON[@]}"
"$PY" -m benchmark.suite rerank --profile large --pools all "${COMMON[@]}"
"$LEGACY" -m benchmark.suite rerank --profile custom --trust-remote-code --pools all "${COMMON[@]}"
"$LEGACY" -m benchmark.suite rerank --profile gemma --pools all "${COMMON[@]}"
# Key phải có trong chính terminal environment nếu dùng dòng dưới.
"$PY" -m benchmark.suite rerank --profile hosted --pools rrf-bge-m3-dense "${COMMON[@]}"
"$PY" -m benchmark.suite report --output "$RUN_ROOT"
```

Notebook vẫn là cách được chuẩn bị đầy đủ, có plan, quota estimate, preflight,
snapshot dataset, package freeze và tải ZIP. `--profile all` là danh mục tổng;
**không** cài tất cả vào một version Transformers rồi mong Gemma/Qwen cùng chạy.

Các tensor trước truncation trong audit có thể vượt architecture max length
và tokenizer in warning; chúng chỉ được **đếm token**, không forward tensor
dài đó vào model. Input inference là bản đã cắt đúng effective cap. Nếu vẫn
thấy `IndexError`/CUDA OOM thật, đọc `reason`, không coi warning là lý do bỏ query.

## 6. Lấy và đọc kết quả

Cell cuối tạo ZIP, copy ZIP vào Drive và gọi `files.download`. Trong ZIP:

- `report.md`: trạng thái từng artifact; quality chỉ từ complete/full;
  retrieval/end-to-end/oracle tách riêng; Δ và confidence interval.
- `summary.csv`: bảng full/core; `breakdown.csv`: VI/EN, category và track.
- `per-query.csv`: kể cả diagnostics, Hole, top1, latency, API request/wait time.
- `retrieval/*.json`, `rerank/<model>/<pool>.json`: đầy đủ score/ranking/audit,
  SHA dataset/corpus/code/weights/code-remote/parent, config, hardware/sessions.
- `manifest.json`: file hashes, status counts, planned artifacts chưa có.
- `plan.json`, `dataset.jsonl`, `gpu-preflight.json`, `*-packages.txt`:
  ngữ cảnh tái lập. Model không thuộc profile T4 ghi trong excluded list.

Online latency là **ước tính cộng** retrieval query time + rerank time đo ở
các run riêng, không phải deployment throughput. RRF cộng hai parent query
time và fusion. Index document/load/warmup tách khỏi query latency. Cohere
wait/backoff tách khỏi HTTP request time (request còn gồm network). Không so
CPU/local với hosted API hoặc khác GPU/dtype/window như một phép đo server
thuần. Chỉ một lần đo/query, không phải load test nhiều phiên đồng thời.

Benchmark chỉ đầy đủ cho **kế hoạch đã chọn** khi không còn missing/failed/
skipped/starting, và các artifact complete đủ56query, full_dataset=true.
Muốn toàncatalog15reranker×20retriever cần fullGPU + API_ALL_POOLS=True + key/quota.
Không tự gán score cho model lỗi hoặc bỏ những query chấm khó.
