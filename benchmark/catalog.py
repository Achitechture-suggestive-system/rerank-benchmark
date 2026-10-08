"""Source-verified model registry. Registry membership is not an inference result.

Revisions were checked against the publishers' Hugging Face repositories on
2026-10-06. Resource floors are conservative scheduling hints, not measurements.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    family: str
    max_tokens: int
    vram_gib: float = 2
    custom_code: bool = False
    code_revision: str = ""
    environment: str = "modern"
    language_scope: str = "multilingual"
    license: str = ""

    @property
    def source(self):
        if self.family == "cohere":
            return "https://docs.cohere.com/reference/rerank"
        return "https://huggingface.co/" + self.model_id


EMBEDDERS = {
    "minilm-multi": ModelSpec("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "e8f8c211226b894fcb81acc59f3b34ba3efd5f42", "mean", 128),
    "mpnet-multi": ModelSpec("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "4328cf26390c98c5e3c738b4460a05b95f4911f5", "mean", 128),
    "e5-small": ModelSpec("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3", "e5", 512),
    "e5-base": ModelSpec("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a", "e5", 512),
    "e5-large": ModelSpec("intfloat/multilingual-e5-large", "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3", "e5", 512, 3),
    "bge-m3-dense": ModelSpec("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", "cls", 8192, 3),
    "qwen-embed-0.6b": ModelSpec("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "qwen-embed", 32768, 3),
    "qwen-embed-4b": ModelSpec("Qwen/Qwen3-Embedding-4B", "5cf2132abc99cad020ac570b19d031efec650f2b", "qwen-embed", 32768, 12),
    "qwen-embed-8b": ModelSpec("Qwen/Qwen3-Embedding-8B", "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", "qwen-embed", 32768, 24),
}

RERANKERS = {
    "bge-m3": ModelSpec("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "classifier", 8192, 3),
    "bge-base": ModelSpec("BAAI/bge-reranker-base", "2cfc18c9415c912f9d8155881c133215df768a70", "classifier", 512, language_scope="English/Chinese baseline"),
    "bge-large": ModelSpec("BAAI/bge-reranker-large", "55611d7bca2a7133960a6d3b71e083071bbfc312", "classifier", 512, 3, language_scope="English/Chinese baseline"),
    "msmarco-l6": ModelSpec("cross-encoder/ms-marco-MiniLM-L-6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a", "classifier", 512, language_scope="English baseline"),
    "msmarco-l12": ModelSpec("cross-encoder/ms-marco-MiniLM-L-12-v2", "7b0235231ca2674cb8ca8f022859a6eba2b1c968", "classifier", 512, language_scope="English baseline"),
    "mmarco-minilm": ModelSpec("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "1427fd652930e4ba29e8149678df786c240d8825", "classifier", 512),
    "qwen0.6b": ModelSpec("Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473", "qwen", 32768, 3),
    "qwen4b": ModelSpec("Qwen/Qwen3-Reranker-4B", "22e683669bc0f0bd69640a1354a6d0aebcfeede5", "qwen", 32768, 12),
    "qwen8b": ModelSpec("Qwen/Qwen3-Reranker-8B", "77d193c791ed757ca307ee72715aa132723da912", "qwen", 32768, 24),
    "gte-multi": ModelSpec("Alibaba-NLP/gte-multilingual-reranker-base", "8215cf04918ba6f7b6a62bb44238ce2953d8831c", "classifier", 8192, 3, True, "40ced75c3017eb27626c9d4ea981bde21a2662f4", "custom"),
    "jina-v2": ModelSpec("jinaai/jina-reranker-v2-base-multilingual", "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9", "classifier", 1024, 3, True, environment="custom", license="CC-BY-NC-4.0"),
    "gemma": ModelSpec("BAAI/bge-reranker-v2.5-gemma2-lightweight", "fabc9f6f51698e30890300fa3917f0560fdbcd75", "gemma", 8192, 24, True, environment="gemma"),
    "cohere-pro": ModelSpec("rerank-v4.0-pro", "hosted", "cohere", 32768, 0, environment="hosted"),
    "cohere-fast": ModelSpec("rerank-v4.0-fast", "hosted", "cohere", 32768, 0, environment="hosted"),
    "cohere-v3.5": ModelSpec("rerank-v3.5", "hosted", "cohere", 4096, 0, environment="hosted"),
}

PROFILES = {
    "t4": {"embedders": [k for k, v in EMBEDDERS.items() if v.vram_gib < 12],
           "rerankers": [k for k, v in RERANKERS.items() if v.environment == "modern" and v.vram_gib < 12]},
    "large": {"embedders": list(EMBEDDERS),
              "rerankers": [k for k, v in RERANKERS.items() if v.environment == "modern"]},
    "custom": {"embedders": [], "rerankers": ["gte-multi", "jina-v2"]},
    "gemma": {"embedders": [], "rerankers": ["gemma"]},
    "hosted": {"embedders": [], "rerankers": ["cohere-pro", "cohere-fast", "cohere-v3.5"]},
    "all": {"embedders": list(EMBEDDERS), "rerankers": list(RERANKERS)},
    "cpu": {"embedders": [], "rerankers": []},
}


def catalog_dict():
    return {"checked_date": "2026-10-06", "embedders": {k: asdict(v) for k, v in EMBEDDERS.items()},
            "rerankers": {k: asdict(v) for k, v in RERANKERS.items()}, "profiles": PROFILES}
