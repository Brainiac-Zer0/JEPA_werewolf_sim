import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from functools import lru_cache

INPUT_DIM = 808
LATENT_DIM = 32

# ───────────────────────────────────────────────────────────────────────────────
# TEXT ENCODER MODULE  (shared; includes tiny LRU for single-string calls)
# ───────────────────────────────────────────────────────────────────────────────
class MessageEncoder(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.output_dim = self.transformer.config.hidden_size

    @lru_cache(maxsize=4096)
    def _encode_cached_one(self, text: str):
        encoded_input = self.tokenizer([text], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.transformer(**encoded_input)
        return self.mean_pooling(model_output, encoded_input['attention_mask'])[0]  # D

    def forward(self, texts):
        # Fast path for a single string (cacheable)
        if isinstance(texts, str):
            return self._encode_cached_one(texts).unsqueeze(0)
        # Batch path (not cached)
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.transformer(**encoded_input)
        return self.mean_pooling(model_output, encoded_input['attention_mask'])

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

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

class PlannerHead(nn.Module):
    def __init__(self, latent_dim, num_agents):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_agents)
        )

    def forward(self, z):
        return self.net(z)

def package_features(agent_alive, round_num, self_msg_embed, neighbor_msg_embed, vote_vector, memory_summary):
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
