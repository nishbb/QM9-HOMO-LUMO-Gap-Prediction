import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set, global_add_pool
from torch_scatter import scatter



class EdgeNetwork(nn.Module):
    def __init__(self, edge_dim, node_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, 64), nn.ReLU(),
            nn.Linear(64, node_dim * node_dim),
        )
        self.node_dim = node_dim

    def forward(self, e):
        return self.net(e).view(-1, self.node_dim, self.node_dim)


class MPNNBaseline(nn.Module):
    def __init__(self, node_in=11, edge_in=4, hidden=64, n_layers=3):
        super().__init__()
        self.node_emb = nn.Linear(node_in, hidden)
        self.convs = nn.ModuleList([
            NNConv(hidden, hidden, EdgeNetwork(edge_in, hidden), aggr='mean')
            for _ in range(n_layers)
        ])
        self.set2set = Set2Set(hidden, processing_steps=3)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _encode(self, data):
        h = F.relu(self.node_emb(data.x.float()))
        for conv in self.convs:
            h = F.relu(conv(h, data.edge_index, data.edge_attr.float()))
        return self.set2set(h, data.batch)   # (B, 2*hidden)

    def forward(self, data):
        return self.mlp(self._encode(data))  # (B, 1)

    def embed(self, data):
        return self._encode(data)



class EGNNLayer(nn.Module):
    def __init__(self, hidden, edge_attr_dim=4):
        super().__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(2 * hidden + 1 + edge_attr_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1, bias=False),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h, pos, edge_index, edge_attr):
        row, col = edge_index
        diff    = pos[row] - pos[col]                                # (E, 3)
        dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)              # (E, 1)

        m_ij = self.phi_e(torch.cat([h[row], h[col],
                                     dist_sq, edge_attr], dim=-1))   # (E, H)

        # Equivariant coordinate update
        w       = self.phi_x(m_ij)                                   # (E, 1)
        agg_pos = scatter(diff * w, col, dim=0,
                          dim_size=h.size(0), reduce='mean')          # (N, 3)
        pos_new = pos + agg_pos

        # Node update
        agg_m = scatter(m_ij, col, dim=0,
                        dim_size=h.size(0), reduce='mean')            # (N, H)
        h_new = self.phi_h(torch.cat([h, agg_m], dim=-1)) + h        # residual

        return h_new, pos_new


class EGNNModel(nn.Module):
    def __init__(self, node_in=11, edge_in=4, hidden=128, n_layers=4, use_pos=True):
        super().__init__()
        self.use_pos  = use_pos
        self.node_emb = nn.Linear(node_in, hidden)
        self.layers   = nn.ModuleList([
            EGNNLayer(hidden, edge_attr_dim=edge_in) for _ in range(n_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def _encode(self, data):
        h   = F.silu(self.node_emb(data.x.float()))
        pos = data.pos.float()
        if not self.use_pos:
            pos = torch.zeros_like(pos)
        for layer in self.layers:
            h, pos = layer(h, pos, data.edge_index, data.edge_attr.float())
        return global_add_pool(h, data.batch)   # (B, hidden)

    def forward(self, data):
        return self.readout(self._encode(data))

    def embed(self, data):
        return self._encode(data)
