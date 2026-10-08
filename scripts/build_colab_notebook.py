"""Deterministically build the checked-in Colab notebook; no notebook execution."""
import json
import textwrap
from pathlib import Path


cells = []


def cell(kind, source):
    item = dict(id=f"cell-{len(cells):02d}", cell_type=kind, metadata={}, source=textwrap.dedent(source).strip().splitlines(keepends=True))
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell("markdown", """
    # Chạy retrieval + reranking đầy đủ trên Colab

    Chọn **Runtime → Change runtime type → GPU** trước khi chạy.
    `RUN_MODE="full"` cần GPU ≥24 GiB, nên dùng **A100 40GB** để có dư bộ nhớ.
    T4 chạy `RUN_MODE="t4"`: không giả vờ đã đo Qwen 4B/8B hoặc Gemma 9B.
    Có 9 embedding, BM25/TF-IDF, 9 RRF (20 retriever) và 15 reranker trong catalog.
    Model mới là adapter đã đối chiếu nguồn, **không phải tất cả đã inference-verified**.

    Mỗi model chạy trong tiến trình riêng; Python 3.12 + CUDA torch được cài trong
    môi trường cô lập. Gemma/GTE/Jina dùng Transformers 4.44.2; Qwen dùng 4.51.3.
    Không cài đè torch của kernel Colab. Không ghép lệnh unittest và benchmark.
    Chạy lần lượt các cell dưới đây. Sau khi ngắt runtime, giữ nguyên `RUN_ID`
    và scoring config để tiếp tục. Không dùng `--limit` cho kết quả đầy đủ.

    Corpus gồm 168 ID nhưng chỉ 6 tài liệu/query có nhãn. Hole@K cho biết phần
    chưa được chấm; đây là fixture tổng hợp, không phải leaderboard BEIR/MTEB.
""")
cell("code", '''
    # 1. Cấu hình: đổi thành "full" khi dùng A100, hoặc giữ "t4" trên Tesla T4.
    RUN_MODE = "t4"               # "t4" / "full"
    RUN_ID = "2026-10-06-suite-v1" # Giữ nguyên khi resume; đổi khi code/data/window thay đổi.
    FULL_MATRIX = True             # Local: mọi retriever × mọi reranker + oracle 6-doc.
    RUN_HOSTED = True              # Cohere cần key và quota; có thể bỏ qua cell API.
    API_ALL_POOLS = False          # True = nhiều request; có thể vượt trial 1000/tháng.
    CANDIDATE_K = 20
    MAX_LENGTH = 1024
    EMBEDDING_MAX_LENGTH = 512
    GIT_REF = "fix/cohere-rate-limit" # Có thể thay bằng commit SHA để đóng băng source.
    KEEP_MODEL_CACHE = False       # False: giải phóng cache riêng từng worker để tiết kiệm disk.

    import os, sys, json, subprocess, shutil, zipfile
    from pathlib import Path
    from google.colab import drive
    assert RUN_MODE in ("t4", "full")
    assert RUN_ID and Path(RUN_ID).name == RUN_ID and RUN_ID not in (".", "..")
    drive.mount("/content/drive")
    RUN_ROOT = Path("/content/drive/MyDrive/rerank-benchmark-results") / RUN_ID
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    REPO = Path("/content/retrieval-rerank-suite")
    print("Kết quả lưu tại:", RUN_ROOT)
''')
cell("code", '''
    # 2. Lấy source. Không reset hay xóa checkout đang có thay đổi.
    REPO_URL = "https://github.com/Achitechture-suggestive-system/rerank-benchmark.git"
    if not REPO.exists():
        subprocess.run(["git", "clone", REPO_URL, str(REPO)], check=True)
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=REPO, text=True).strip()
    assert origin.rstrip("/") == REPO_URL, "Checkout này không thuộc repo benchmark"
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    assert not dirty.strip(), "Checkout có thay đổi: hãy giữ lại chúng và dùng checkout khác"
    subprocess.run(["git", "fetch", "origin", GIT_REF], cwd=REPO, check=True)
    subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=REPO, check=True)
    COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    print("Source commit:", COMMIT)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from benchmark.catalog import EMBEDDERS, RERANKERS, PROFILES
    # Source cố định trong một kernel: restart kernel nếu muốn chuyển sang code mới.
    from benchmark.run import source_hash
    assert Path(sys.modules["benchmark.catalog"].__file__).resolve().is_relative_to(REPO)
''')
cell("code", '''
    # 3. Tạo Python 3.12 độc lập, cài CUDA torch trước Transformers.
    subprocess.run([sys.executable, "-m", "pip", "install", "uv==0.9.26"], check=True)
    UV = [sys.executable, "-m", "uv"]
    subprocess.run(UV + ["python", "install", "3.12.12"], check=True)
    ENVS = {}
    for name, requirements in (("modern", "requirements-colab-modern.txt"), ("legacy", "requirements-colab-legacy.txt")):
        venv = Path("/content/benchmark-envs") / name
        if not (venv / "bin/python").exists():
            subprocess.run(UV + ["venv", "--python", "3.12.12", str(venv)], check=True)
        py = str(venv / "bin/python")
        subprocess.run(UV + ["pip", "install", "--python", py, "torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu124"], check=True)
        subprocess.run(UV + ["pip", "install", "--python", py, "-r", str(REPO / requirements)], check=True)
        ENVS[name] = py
        frozen = subprocess.check_output(UV + ["pip", "freeze", "--python", py], text=True)
        (RUN_ROOT / (name + "-packages.txt")).write_text(frozen, encoding="utf-8")
    print("Môi trường benchmark:", ENVS)
''')
cell("code", '''
    # 4. Kiểm tra GPU bằng đúng Python sẽ chạy benchmark (không chỉ nvidia-smi).
    probe = """import json, sys, torch, transformers
    assert torch.cuda.is_available(), 'CUDA unavailable: choose GPU and verify CUDA torch wheel'
    x = torch.ones((2, 2), device='cuda'); y = x @ x; torch.cuda.synchronize()
    p = torch.cuda.get_device_properties(0)
    print(json.dumps(dict(python=sys.version, torch=torch.__version__, transformers=transformers.__version__,
                         cuda=torch.version.cuda, gpu=p.name, vram_gib=p.total_memory/2**30, test_sum=y.sum().item())))
    """
    GPU_INFO = {}
    for name, py in ENVS.items():
        result = subprocess.check_output([py, "-c", probe], text=True)
        GPU_INFO[name] = json.loads(result.strip().splitlines()[-1])
        print(name, GPU_INFO[name])
    if RUN_MODE == "full":
        assert GPU_INFO["modern"]["vram_gib"] >= 23, "Full FP16 cần GPU >=24GB; T4 hãy dùng RUN_MODE='t4'"
    (RUN_ROOT / "gpu-preflight.json").write_text(json.dumps(GPU_INFO, indent=2), encoding="utf-8")
    subprocess.run([ENVS["modern"], "-m", "unittest", "discover", "-s", "tests", "-q"], cwd=REPO, check=True)
''')
cell("code", '''
    # 5. Lập kế hoạch. Full matrix Cohere cần chủ động bật vì có quota/phí.
    PROFILE = "large" if RUN_MODE == "full" else "t4"
    EMBED_NAMES = PROFILES[PROFILE]["embedders"]
    LOCAL_MODELS = PROFILES[PROFILE]["rerankers"] + PROFILES["custom"]["rerankers"]
    if RUN_MODE == "full":
        LOCAL_MODELS += ["gemma"]
    RETRIEVERS = ["bm25", "tfidf"] + EMBED_NAMES + ["rrf-" + name for name in EMBED_NAMES]
    LOCAL_POOLS = RETRIEVERS if FULL_MATRIX else ["bm25", "rrf-bge-m3-dense"]
    API_POOLS = RETRIEVERS if API_ALL_POOLS else ["rrf-bge-m3-dense"]
    API_MODELS = PROFILES["hosted"]["rerankers"] if RUN_HOSTED else []
    expected = ["retrieval/" + name + ".json" for name in RETRIEVERS]
    expected += [f"rerank/{model}/{pool}.json" for model in LOCAL_MODELS for pool in LOCAL_POOLS + ["oracle"]]
    expected += [f"rerank/{model}/{pool}.json" for model in API_MODELS for pool in API_POOLS + ["oracle"]]
    plan = dict(commit=COMMIT, implementation_sha256=source_hash(), run_mode=RUN_MODE,
                candidate_k=CANDIDATE_K, max_length=MAX_LENGTH, embedding_max_length=EMBEDDING_MAX_LENGTH,
                retrievers=RETRIEVERS, local_models=LOCAL_MODELS, hosted_models=API_MODELS,
                local_pools=LOCAL_POOLS, api_pools=API_POOLS, expected_artifacts=expected,
                excluded_embeddings=[name for name in EMBEDDERS if name not in EMBED_NAMES],
                excluded_by_resource=[name for name in RERANKERS if name not in LOCAL_MODELS + API_MODELS])
    plan_path = RUN_ROOT / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        for key in ("commit", "implementation_sha256", "candidate_k", "max_length", "embedding_max_length"):
            assert old[key] == plan[key], f"{key} đã đổi: dùng RUN_ID mới, không trộn kết quả"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    shutil.copy2(REPO / "data/evidence_depth.jsonl", RUN_ROOT / "dataset.jsonl")
    calls = len(API_MODELS) * (56 * (len(API_POOLS) + 1) + 1)
    print(len(RETRIEVERS), "retriever;", len(LOCAL_MODELS) + len(API_MODELS), "reranker;", len(expected), "artifact dự kiến")
    print("Embedding ngoài profile:", plan["excluded_embeddings"])
    print("Reranker ngoài profile / API đã tắt:", plan["excluded_by_resource"])
    print("Cohere requests trước retries:", calls, "(~", round(calls * 6.2 / 60), "phút pacing)")
    if calls > 1000:
        print("CẢNH BÁO: vượt quota trial 1000 API calls/tháng; cần production key/quota hoặc API_ALL_POOLS=False")
    COMMON = ["--output", str(RUN_ROOT), "--device", "cuda", "--dtype", "float16",
              "--candidate-k", str(CANDIDATE_K), "--max-length", str(MAX_LENGTH),
              "--embedding-max-length", str(EMBEDDING_MAX_LENGTH), "--batch-size", "1",
              "--embedding-batch-size", "4", "--threads", "4", "--resume"]
    if not KEEP_MODEL_CACHE:
        COMMON += ["--ephemeral-cache", "/content/benchmark-downloads"]
    if RUN_MODE == "full":
        COMMON += ["--allow-large-models"]
    def run_stage(environment, stage, *extra):
        command = [ENVS[environment], "-m", "benchmark.suite", stage, *COMMON, *extra]
        # Key và env nằm trong cùng Python process; không dùng export trong cell shell khác.
        result = subprocess.run(command, cwd=REPO, env=os.environ.copy())
        print("Exit code:", result.returncode, "— xem JSON/report để biết model failed/skipped")
        return result.returncode
    run_stage("modern", "plan", "--profile", PROFILE, "--pools", *LOCAL_POOLS)
''')
cell("code", '''
    # 6. Retrieval: BM25, TF-IDF, các embedding và BM25+dense RRF trên TOÀN corpus.
    run_stage("modern", "retrieve", "--profile", PROFILE)
''')
cell("code", '''
    # 7. Reranker chuẩn: BGE, MS MARCO, mMARCO, Qwen các size phù hợp GPU.
    # Cùng top-K của từng retriever; oracle đo độc lập trên 6 tài liệu gốc.
    run_stage("modern", "rerank", "--profile", PROFILE, "--pools", *LOCAL_POOLS)
''')
cell("code", '''
    # 8. GTE/Jina: chạy module nhà phát hành ở revision đã pin.
    # Jina license CC-BY-NC-4.0; không mặc nhiên dùng thương mại.
    run_stage("legacy", "rerank", "--profile", "custom", "--trust-remote-code", "--pools", *LOCAL_POOLS)
''')
cell("code", '''
    # 9. Gemma 9B chỉ ở chế độ full; không thay checkpoint bằng bản quantized khác.
    if RUN_MODE == "full":
        run_stage("legacy", "rerank", "--profile", "gemma", "--pools", *LOCAL_POOLS,
                  "--cutoff-layer", "28", "--compress-ratio", "2", "--compress-layers", "24")
    else:
        print("Gemma không chạy trên T4. Dùng A100 + RUN_MODE='full' để hoàn tất model lớn.")
''')
cell("code", '''
    # 10. Cohere: thêm Secret COHERE_API_KEY và bật Notebook access, hoặc nhập ẩn.
    # Không dán key vào code/output/Git. Trial ~10 request/phút; pacing mặc định 6.2s.
    if RUN_HOSTED:
        from google.colab import userdata
        from getpass import getpass
        try:
            key = userdata.get("COHERE_API_KEY")
        except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
            key = getpass("Cohere key (để trống = ghi skipped): ")
        if key:
            os.environ["COHERE_API_KEY"] = key.strip()
        else:
            os.environ.pop("COHERE_API_KEY", None)
        del key
        run_stage("modern", "rerank", "--profile", "hosted", "--pools", *API_POOLS,
                  "--cohere-min-interval", "6.2", "--cohere-max-attempts", "8")
''')
cell("code", '''
    # 11. Kiểm tra và lấy kết quả. Missing/failed/skipped ≠ benchmark đầy đủ.
    from google.colab import files
    code = run_stage("modern", "report")
    assert code == 0, "Report chưa hợp lệ: đọc lỗi hash/query/pool, không tự sửa JSON cho qua"
    manifest = json.loads((RUN_ROOT / "manifest.json").read_text(encoding="utf-8"))
    print("Trạng thái:", manifest["status_counts"])
    print("Planned artifacts chưa có:", len(manifest["missing_planned_artifacts"]))
    from IPython.display import Markdown, display
    display(Markdown((RUN_ROOT / "report.md").read_text(encoding="utf-8")))
    archive = Path("/content") / (RUN_ID + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(RUN_ROOT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(RUN_ROOT))
    shutil.copy2(archive, RUN_ROOT.parent / archive.name)
    print("Drive ZIP:", RUN_ROOT.parent / archive.name)
    files.download(str(archive))
''')
cell("markdown", """
    ## Đọc kết quả và chạy tiếp

    - `report.md`: trạng thái từng run, retrieval, end-to-end, oracle và delta/CI.
    - `summary.csv`: các run complete/full; `breakdown.csv`: theo ngôn ngữ, category, track.
    - `per-query.csv`: toàn bộ query, Hole@K, latency, API wait/request time.
    - `retrieval/*.json`, `rerank/<model>/<pool>.json`: rankings, scores, revision,
      tokenizer audit, hardware, checkpoint từng query.
    - `manifest.json`: hashes và model/pipeline đã lên kế hoạch nhưng chưa chạy.

    Khi lỗi giữa chừng: kiểm tra `reason` trong JSON, xử lý tài nguyên/key/quota rồi
    chạy lại cell model đó với cùng `RUN_ID`. Nếu đổi code/dataset/context/K,
    hãy chọn `RUN_ID` mới. Không ghép smoke `--limit` với full run.
    Có thể chuyển T4 → A100 để bổ sung model lớn bằng `RUN_MODE="full"` trên
    cùng source/scoring config; hardware của mỗi phiên được giữ trong JSON.
    FULL_MATRIX=True so sánh toàn bộ cặp local, rất lâu; không cam kết Colab
    miễn phí chạy xong trong một phiên. `API_ALL_POOLS=True` mở toàn bộ cặp
    Cohere nhưng cần quota/phí phù hợp; mặc định chỉ RRF-BGE-M3 + oracle.
""")

root = Path(__file__).resolve().parents[1]
target = root / "notebooks/retrieval_rerank_colab.ipynb"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(dict(cells=cells, nbformat=4, nbformat_minor=5,
                                metadata=dict(colab=dict(name=target.name),
                                              kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                              language_info=dict(name="python"))), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(target)
