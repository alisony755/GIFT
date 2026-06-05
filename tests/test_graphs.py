from src.embeddings.transe_loader import TransELoader
from src.graph.word_graph_builder import WordGraphBuilder
from src.graph.entity_graph_builder import EntityGraphBuilder
from src.graph.pos_graph_builder import POSGraphBuilder
import os

# Run with python3 -m tests.test_graphs

# Anchor to project root
ROOT = os.path.dirname(os.path.dirname(__file__))

transe_path = os.path.join(
    ROOT,
    "data/external/NELL_KG/entity2vec.TransE"
)

mapping_path = os.path.join(
    ROOT,
    "data/external/NELL_KG/ent2ids_refined"
)

word_builder = WordGraphBuilder(
    glove_path="data/external/glove/glove.pkl"
)

entity_builder = EntityGraphBuilder(
    transe_path,
    mapping_path
)

pos_builder = POSGraphBuilder()

corpus_tokens = [
    ["apple", "banana"],
    ["banana", "orange"]
]

corpus_entities = [
    ["Apple_Inc"],
    ["Orange_Fruit"]
]

corpus_pos = [
    ["NN", "NN"],
    ["NN", "VB"]
]

word_graph = word_builder.build(corpus_tokens)
entity_graph = entity_builder.build(corpus_entities)
pos_graph = pos_builder.build(corpus_pos)

print("WORD: ", word_graph.X.shape, word_graph.A.shape)
print("ENTITY: ", entity_graph.X.shape, entity_graph.A.shape)
print("POS: ", pos_graph.X.shape, pos_graph.A.shape)

loader = TransELoader(
    "data/external/NELL_KG/entity2vec.TransE",
    "data/external/NELL_KG/ent2ids_refined"
)

print(len(loader.entity_to_id))
print(loader.embeddings.shape)
print(loader.get_embedding("man u")[:5])
