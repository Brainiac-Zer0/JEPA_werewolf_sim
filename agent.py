import torch
import torch.nn.functional as F
from encoders import MessageEncoder, MLPBeliefEncoder, WorldModelMLP, ActionEncoder, package_features, INPUT_DIM, LATENT_DIM

NUM_AGENTS = 6  # Used for vote vector size
MAX_MEMORY = 5  # Max stored latent states for belief memory

class BaseAgent:
    def __init__(self, name, encoder=None, world_model=None, action_encoder=None):
        self.name = name
        self.role = None       # Assigned by roles.py
        self.alive = True
        self.last_message = ""
        self.llm_fn = None     # Assigned externally

        # 🔧 JEPA Modules (shared or individual)
        self.message_encoder = MessageEncoder()
        self.encoder = encoder if encoder is not None else MLPBeliefEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
        self.world_model = world_model if world_model is not None else WorldModelMLP(latent_dim=LATENT_DIM, action_dim=8)
        self.action_encoder = action_encoder if action_encoder is not None else ActionEncoder(num_actions=6, action_dim=8)

        # 🧠 Memory buffers
        self.vote_history = []        # List of voted agent names
        self.latent_history = []      # List of past z_t vectors
        self.heard_messages = {}      # Dict of agent name → message

    def observe(self, agents):
        observed = []
        for a in agents:
            if a.alive and a.name != self.name:
                observed.append((a.name, a.last_message))
                self.heard_messages[a.name] = a.last_message
        return observed

    def vote(self, agents):
        alive = [a for a in agents if a.alive and a.name != self.name]
        if alive:
            chosen = alive[0]
            self.vote_history.append(chosen.name)
            if len(self.vote_history) > MAX_MEMORY:
                self.vote_history.pop(0)
            return chosen
        return self

    def choose_night_target(self, agents):
        candidates = [a for a in agents if a.alive and a.name != self.name]
        return candidates[0].name if candidates else None

    def receive_messages(self, agents):
        pass

    def encode_current_belief(self, round_num, agents):
        # Encode own message
        self_msg_embed = self.message_encoder(self.last_message).squeeze()

        # Encode messages from nearby agents
        neighbor_data = self.observe(agents)
        neighbor_msgs = [msg for _, msg in neighbor_data if msg]
        if neighbor_msgs:
            neighbor_msg_embed = self.message_encoder(neighbor_msgs).mean(dim=0)
        else:
            neighbor_msg_embed = torch.zeros_like(self_msg_embed)

        # --- Voting history vector ---
        vote_vector = torch.zeros(NUM_AGENTS)
        for name in self.vote_history[-MAX_MEMORY:]:
            try:
                idx = int(name.split('_')[1])
                vote_vector[idx] += 1.0
            except:
                continue
        if vote_vector.sum() > 0:
            vote_vector /= vote_vector.sum()

        # --- Belief memory vector ---
        if self.latent_history:
            memory_summary = torch.stack(self.latent_history).mean(dim=0)
        else:
            memory_summary = torch.zeros(32)

        # Package all features
        x = package_features(
            agent_alive=self.alive,
            round_num=round_num,
            self_msg_embed=self_msg_embed,
            neighbor_msg_embed=neighbor_msg_embed,
            vote_vector=vote_vector,
            memory_summary=memory_summary
        )

        z = self.encoder(x)

        self.latent_history.append(z.detach())
        if len(self.latent_history) > MAX_MEMORY:
            self.latent_history.pop(0)

        return z

    def decode_z(self, z):
        mean = z.mean().item()
        std = z.std().item()
        lines = []

        if mean > 0.2:
            lines.append("I have a bad feeling about someone.")
        elif mean < -0.2:
            lines.append("Things seem quiet... too quiet.")
        else:
            lines.append("The group seems uncertain.")

        if std > 0.5:
            lines.append("I'm unsure who to trust.")
        else:
            lines.append("I feel confident in my suspicions.")

        return " ".join(lines)

    def speak(self):
        if self.llm_fn:
            z = self.encode_current_belief(round_num=0, agents=[])
            response = self.llm_fn(z, self)
            self.last_message = response
            return response
        return "..."
