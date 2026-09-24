### 4.1. Chất lượng và trạng thái chạy

Bảng chất lượng dùng **44 câu core / 22 nhóm độc lập**. Hai ngôn ngữ của cùng tình huống không phải hai mẫu độc lập.

| Mô hình | Trạng thái |
| --- | --- |
| BGE M3 | `complete` |
| BM25 | `complete` |
| Cohere 4.0 pro | `skipped` |
| BGE Gemma2 lightweight | `skipped` |
| Qwen3 4B | `skipped` |

**Điểm chất lượng — chỉ các lần chạy đầy đủ**

| Mô hình | nDCG@5 | MRR@5 | Hit@1 | Strict@1 | Hard-neg win |
| --- | ---: | ---: | ---: | ---: | ---: |
| BGE M3 | 0.7651 | 0.8292 | 72.73% | 56.82% | 81.82% |
| BM25 | 0.5656 | 0.5500 | 27.27% | 18.18% | 51.52% |

> **Cách đọc:** Hit@1 nhận grade ≥ 2; Strict@1 chỉ nhận grade = 3. Kết quả được sinh từ JSON; mô hình chưa chạy không được gán điểm.

- `cohere-pro`: COHERE_API_KEY is not configured.
- `gemma`: Large model loading disabled for this resource-constrained run; rerun on a suitable machine with --allow-large-models.
- `qwen4b`: Large model loading disabled for this resource-constrained run; rerun on a suitable machine with --allow-large-models.

### 4.2. Thời gian xử lý

| Mô hình | p50 (s/query) | p95 (s/query) | Cặp/giây |
| --- | ---: | ---: | ---: |
| BGE M3 | 2.348 | 14.395 | 1.89 |
| BM25 | 0.000 | 0.001 | 20254.51 |

p50/p95 tính trên toàn bộ query, một lượt sau warm-up; có tokenization, không có thời gian load. Cohere nếu chạy gồm cả network. Đây không phải phép đo throughput production hay so sánh latency trên cùng phần cứng.

### 4.3. Phân tích từng mô hình

#### BGE M3

**Theo nhóm năng lực**

| Nhóm | Số query | nDCG@5 | Strict@1 |
| --- | ---: | ---: | ---: |
| causal | 4 | 0.9324 | 100.00% |
| code | 4 | 0.7148 | 50.00% |
| conjunction | 2 | 0.9049 | 100.00% |
| crosslingual | 4 | 0.8493 | 50.00% |
| entity | 4 | 0.8524 | 75.00% |
| evidence_quality | 2 | 0.6609 | 0.00% |
| injection | 4 | 0.8556 | 75.00% |
| logic | 2 | 0.2731 | 0.00% |
| multihop | 4 | 0.8748 | 100.00% |
| negation | 4 | 0.5752 | 0.00% |
| numeric | 4 | 0.6582 | 50.00% |
| quantifier | 2 | 0.6275 | 0.00% |
| temporal | 4 | 0.8706 | 75.00% |

**Theo ngôn ngữ — chỉ core**

| Ngôn ngữ | nDCG@5 | Strict@1 |
| --- | ---: | ---: |
| `vi` | 0.7505 | 59.09% |
| `en` | 0.7797 | 54.55% |

Không diễn giải đây là đánh giá tổng quát mọi dữ liệu tiếng Việt/Anh.

**Tài liệu dài và token bị cắt**

- Long-context: nDCG@5 **0.6519**, Strict@1 **0.00%** trên 4 câu.
- Số cặp bị cắt: toàn bộ **8**, core **0**.

Nếu bằng chứng bị cắt mất, lỗi thuộc cả chính sách cửa sổ và mô hình; không kết luận mô hình không hiểu bằng chứng chưa được thấy.

**Một số ca sai có thể kiểm tra lại**

- **`or-auth-vi`** — top 1 grade **0**, đáp án đầy đủ ở hạng **6**.
  Hai nhánh đều sai. Điểm top 1: `1.367872`; điểm đáp án: `-3.450614`.

- **`or-auth-en`** — top 1 grade **0**, đáp án đầy đủ ở hạng **5**.
  Hai nhánh đều sai. Điểm top 1: `3.795702`; điểm đáp án: `-1.576579`.

- **`numeric-units-en`** — top 1 grade **0**, đáp án đầy đủ ở hạng **4**.
  Nhầm giây/phút. Điểm top 1: `1.777738`; điểm đáp án: `-1.137777`.

- **`negation-cache-vi`** — top 1 grade **0**, đáp án đầy đủ ở hạng **5**.
  Không cấm không đồng nghĩa với không lưu. Điểm top 1: `-0.591446`; điểm đáp án: `-5.206386`.

**Câu hỏi không có đáp án**

Unanswerable: **8** câu không có tài liệu đúng. Giá trị lớn nhất trong các điểm top 1 là `0.169746`.

Chỉ là chẩn đoán trong thang điểm của mô hình; chưa có ngưỡng abstain được hiệu chuẩn và không đưa các câu này vào nDCG/MRR.

**Runtime và artifact**

- Thiết bị: `cpu`; dtype: `float32`.
- Load: 44.57s; warm-up: 0.48s.
- Inference: 336 cặp / 177.91s = 1.89 cặp/s.
- Artifact: [bge-m3.json](local-2026-09-24/bge-m3.json).

#### BM25

**Theo nhóm năng lực**

| Nhóm | Số query | nDCG@5 | Strict@1 |
| --- | ---: | ---: | ---: |
| causal | 4 | 0.5376 | 0.00% |
| code | 4 | 0.4399 | 0.00% |
| conjunction | 2 | 0.7105 | 50.00% |
| crosslingual | 4 | 0.4109 | 0.00% |
| entity | 4 | 0.6370 | 25.00% |
| evidence_quality | 2 | 0.5548 | 0.00% |
| injection | 4 | 0.5696 | 25.00% |
| logic | 2 | 0.4352 | 0.00% |
| multihop | 4 | 0.6709 | 50.00% |
| negation | 4 | 0.4066 | 0.00% |
| numeric | 4 | 0.5429 | 0.00% |
| quantifier | 2 | 0.6278 | 0.00% |
| temporal | 4 | 0.8422 | 75.00% |

**Theo ngôn ngữ — chỉ core**

| Ngôn ngữ | nDCG@5 | Strict@1 |
| --- | ---: | ---: |
| `vi` | 0.5135 | 13.64% |
| `en` | 0.6178 | 22.73% |

Không diễn giải đây là đánh giá tổng quát mọi dữ liệu tiếng Việt/Anh.

**Tài liệu dài và token bị cắt**

- Long-context: nDCG@5 **0.5680**, Strict@1 **25.00%** trên 4 câu.
- Số cặp bị cắt: toàn bộ **0**, core **0**.

Nếu bằng chứng bị cắt mất, lỗi thuộc cả chính sách cửa sổ và mô hình; không kết luận mô hình không hiểu bằng chứng chưa được thấy.

**Một số ca sai có thể kiểm tra lại**

- **`negation-cache-vi`** — top 1 grade **0**, đáp án đầy đủ ở hạng **6**.
  Không cấm không đồng nghĩa với không lưu. Điểm top 1: `0.258996`; điểm đáp án: `0.000000`.

- **`numeric-units-vi`** — top 1 grade **0**, đáp án đầy đủ ở hạng **6**.
  Nhầm bit/byte. Điểm top 1: `2.702237`; điểm đáp án: `0.198784`.

- **`causal-incident-vi`** — top 1 grade **0**, đáp án đầy đủ ở hạng **6**.
  Không chứng minh can thiệp loại timeout. Điểm top 1: `1.823516`; điểm đáp án: `0.188351`.

- **`crosslingual-retry-en`** — top 1 grade **0**, đáp án đầy đủ ở hạng **6**.
  Trái yêu cầu. Điểm top 1: `4.666727`; điểm đáp án: `1.788805`.

**Câu hỏi không có đáp án**

Unanswerable: **8** câu không có tài liệu đúng. Giá trị lớn nhất trong các điểm top 1 là `5.422367`.

Chỉ là chẩn đoán trong thang điểm của mô hình; chưa có ngưỡng abstain được hiệu chuẩn và không đưa các câu này vào nDCG/MRR.

**Runtime và artifact**

- Thiết bị: `cpu`; dtype: `n/a`.
- Load: 0.00s; warm-up: 0.00s.
- Inference: 336 cặp / 0.02s = 20254.51 cặp/s.
- Artifact: [bm25.json](local-2026-09-24/bm25.json).

### 4.4. So sánh theo từng query

#### BGE M3 so với BM25

- Δ nDCG@5 core: **+0.1995**.
- Bootstrap CI 95%: **[+0.1287, +0.2720]**.
- Theo query: **34 tăng**, **9 giảm**, **1 không đổi**.

Bootstrap 2000 lần theo 22 nhóm tình huống, seed cố định. Khoảng này chỉ mô tả biến thiên trong bộ fixture tổng hợp, không suy rộng ra production.

| Query minh họa | Δ nDCG@5 |
| --- | ---: |
| `numeric-units-en` | -0.3121 |
| `entity-owner-en` | -0.3072 |
| `or-auth-vi` | -0.2351 |
| `causal-incident-vi` | +0.7173 |
| `crosslingual-retry-en` | +0.7081 |
| `crosslingual-delete-vi` | +0.5118 |

### 4.5. Giới hạn kết luận của lần chạy

Chỉ so sánh chất lượng các mô hình có trạng thái complete. Chưa có đủ kết quả Qwen 4B, Gemma2 lightweight và Cohere thì chưa thể xếp hạng bốn mô hình. BM25 là mốc từ vựng, không thay thế một trong bốn mô hình được yêu cầu. Nhãn do tác giả fixture xây dựng và chưa qua đánh giá mù bởi nhiều người.

**Dataset SHA-256**

```text
c03286ee60f320652abb94a5863e07bebebf09a8c5ca35ff386a8f6a680b1a42
```

Mỗi JSON lưu revision mô hình, phiên bản thư viện, cấu hình, thời điểm, runtime, score/rank và token audit.
