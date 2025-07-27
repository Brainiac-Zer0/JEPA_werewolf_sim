import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


INPUT_DIM = 808  # Adjust based on your feature packaging
LATENT_DIM = 32  # Latent state dimension for belief encoding

# ───────────────────────────────────────────────────────────────────────────────
# TEXT ENCODER MODULE
# Uses a transformer-based model to encode messages into dense vectors
# NOTE: Replace this with a stub or one-hot dictionary encoder if needed
# for speed or interpretability.
# ───────────────────────────────────────────────────────────────────────────────
class MessageEncoder(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.output_dim = self.transformer.config.hidden_size

    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.transformer(**encoded_input)
        return self.mean_pooling(model_output, encoded_input['attention_mask'])

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)


# ───────────────────────────────────────────────────────────────────────────────
# MLP-BASED BELIEF STATE ENCODER (f_enc)
# Takes structured features + message embeddings and produces latent state z_t
# ───────────────────────────────────────────────────────────────────────────────
# TODO: Try other architectures:
#  1. GRU over sequential message tokens (temporal memory)
#  2. Attention-based encoder over message pool
#  3. Graph neural encoder (if agents are treated as nodes)
# ───────────────────────────────────────────────────────────────────────────────
class MLPBeliefEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, input_vector):
        return self.encoder(input_vector)


# ───────────────────────────────────────────────────────────────────────────────
# WORLD MODEL MODULE (f_wm)
# Predicts next latent state z_{t+1} from current latent z_t and action a_t
# ───────────────────────────────────────────────────────────────────────────────
# NOTE: Action can be one-hot, index-embedded, or encoded text
# ───────────────────────────────────────────────────────────────────────────────
class ActionEncoder(nn.Module):
    def __init__(self, num_actions, action_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, action_dim)

    def forward(self, action_idx):
        return self.embedding(action_idx)


class WorldModelMLP(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, z_t, a_t_embed):
        combined = torch.cat([z_t, a_t_embed], dim=-1)
        return self.model(combined)


# ───────────────────────────────────────────────────────────────────────────────
# FEATURE PACKAGING HELPER
# Converts agent state info into a flat tensor input to MLPBeliefEncoder
# ───────────────────────────────────────────────────────────────────────────────
def package_features(agent_alive, round_num, self_msg_embed, neighbor_msg_embed, vote_vector, memory_summary):
    """
    Parameters:
        agent_alive (bool): whether the agent is alive
        round_num (int or float): current round number (should normalize if >10)
        self_msg_embed (Tensor): shape (D,)
        neighbor_msg_embed (Tensor): shape (D,)
        vote_vector (Tensor): shape (K,), one-hot or soft count
        memory_summary (Tensor): shape (latent_dim,)
    Returns:
        Tensor: shape (final_input_dim,) for f_enc
    """
    alive_flag = torch.tensor([1.0 if agent_alive else 0.0])
    round_norm = torch.tensor([round_num / 10.0])
    return torch.cat([
        alive_flag,
        round_norm,
        self_msg_embed,
        neighbor_msg_embed,
        vote_vector,
        memory_summary
    ], dim=0)
