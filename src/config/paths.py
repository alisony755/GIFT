from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GLOVE_EMBEDDING_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "glove"
    / "embedding_glove.p"
)

TRANSE_EMBEDDING_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "NELL_KG"
    / "entity2vec.TransE"
)

TRANSE_MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "NELL_KG"
    / "ent2ids_refined"
)