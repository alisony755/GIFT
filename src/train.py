import numpy as np
import torch

from src.graph.word_graph_builder import WordGraphBuilder
from src.graph.entity_graph_builder import EntityGraphBuilder
from src.graph.pos_graph_builder import POSGraphBuilder

from src.text_representation.gcn_encoder import GCNEncoder
from src.text_representation.td_matrix_builder import TDMatrixBuilder
from src.text_representation.text_encoder import TextEncoder

from src.representation.svd_views import SVDViewGenerator

from src.contrastive.projection_head import ProjectionHead
from src.contrastive.contrastive_loss import ContrastiveLoss
from src.contrastive.cluster_contrastive_loss import ClusterContrastiveLoss

from src.clustering.constrained_seed_kmeans import ConstrainedSeedKMeans


class GIFTTrainer:
    def __init__(self, config):
        self.config = config

        self.td_builder = TDMatrixBuilder()
        self.encoder = TextEncoder()
        self.svd = SVDViewGenerator(rank_ratio=config["rank_ratio"])

        self.contrastive_loss = ContrastiveLoss(temperature=config["temp"])
        self.cluster_loss = ClusterContrastiveLoss(temperature=config["temp"])

        self.projection = None  # Initialized after dim known
    
    # Build word, entity, and POS graphs
    def build_graphs(self, corpus_tokens, corpus_entities, corpus_pos_tags):
        # Word graph
        word_graph = self.word_graph_builder.build(
            corpus_tokens
        )

        # Entity graph
        entity_graph = self.entity_graph_builder.build(
            corpus_entities
        )

        # POS graph
        pos_graph = self.pos_graph_builder.build(
            corpus_pos_tags
        )

        return (word_graph, entity_graph, pos_graph)

    def run_gcn(self, word_graph, entity_graph, pos_graph):
        H_w = GCNEncoder(
            input_dim=word_graph.X.shape[1],
            hidden_dim=self.config["hidden_dim"]
        )(
            torch.tensor(word_graph.X, dtype=torch.float32),
            torch.tensor(word_graph.A, dtype=torch.float32)
        )

        H_e = GCNEncoder(
            input_dim=entity_graph.X.shape[1],
            hidden_dim=self.config["hidden_dim"]
        )(
            torch.tensor(entity_graph.X, dtype=torch.float32),
            torch.tensor(entity_graph.A, dtype=torch.float32)
        )

        H_p = GCNEncoder(
            input_dim=pos_graph.X.shape[1],
            hidden_dim=self.config["hidden_dim"]
        )(
            torch.tensor(pos_graph.X, dtype=torch.float32),
            torch.tensor(pos_graph.A, dtype=torch.float32)
        )

        return (
            H_w.detach().cpu().numpy(),
            H_e.detach().cpu().numpy(),
            H_p.detach().cpu().numpy()
        )
    
    def build_td_matrices(self, texts, vocab, entities, pos_tags):
        M_w = self.td_builder.build_word_matrix(
            texts,
            vocab
        )

        M_e = self.td_builder.build_entity_matrix(
            texts,
            entities
        )

        M_p = self.td_builder.build_pos_matrix(
            pos_tags,
            list(set(sum(pos_tags, [])))
        )

        return M_w, M_e, M_p

    def run_svd(self, M_w, M_e, M_p):
        M_wr = self.svd.build_augmented_matrix(M_w)
        M_er = self.svd.build_augmented_matrix(M_e)
        M_pr = self.svd.build_augmented_matrix(M_p)

        return M_wr, M_er, M_pr

    def build_embeddings(self, M_w, M_e, M_p, M_wr, M_er, M_pr, H_w, H_e, H_p):
        Z_org = self.encoder.build_original(
            M_w, M_e, M_p,
            H_w, H_e, H_p
        )

        Z_aug = self.encoder.build_augmented(
            M_wr, M_er, M_pr,
            H_w, H_e, H_p
        )

        return Z_org, Z_aug

    def run_kmeans(self, Z_org, labeled_idx, labels):
        kmeans = ConstrainedSeedKMeans()

        weak_labels = kmeans.fit_predict(
            Z_org,
            labeled_idx,
            labels
        )

        return weak_labels

    def forward(self, batch):
        texts = batch["texts"]
        vocab = batch["vocab"]
        entities = batch["entities"]
        pos_tags = batch["pos"]
        features = batch["features"]

        # Graphs
        graphs = self.build_graphs(texts)

        # GCN
        H_w, H_e, H_p = self.run_gcn(graphs, features)

        # TD matrices
        M_w, M_e, M_p = self.build_td_matrices(
            texts, vocab, entities, pos_tags
        )

        # SVD augmentation
        M_wr, M_er, M_pr = self.run_svd(M_w, M_e, M_p)

        # Z_org, Z_aug
        Z_org, Z_aug = self.build_embeddings(
            M_w, M_e, M_p,
            M_wr, M_er, M_pr,
            H_w, H_e, H_p
        )

        # k-means
        weak_labels = self.run_kmeans(
            Z_org,
            batch["labeled_idx"],
            batch["labels"]
        )

        # Train model
        outputs = self.model(
            torch.tensor(Z_org, dtype=torch.float32),
            torch.tensor(Z_aug, dtype=torch.float32),
            torch.tensor(weak_labels),
            batch["labeled_idx"],
            batch["labels_tensor"]
        )

        return outputs["loss"], outputs["logits"]