import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split
import numpy as np
import pickle as pkl
import argparse

from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---- PyTorch Geometric imports ----
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  # NOTE: PyG DataLoader
from torch_geometric.nn import MessagePassing


# ====================================================================
# DATASET: loads simulations and returns a *graph* per simulation
# ====================================================================

class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for black hole trajectories, returning PyG Data objects.

    For each simulation (graph):
      - Nodes = black holes
      - Node features: [x, y, z, vx, vy, vz] (normalized)
      - Targets (y): final positions [x_f, y_f, z_f] (normalized)
      - Fully-connected directed edges (i -> j, i != j)
      - Edge features: [dx, dy, dz, distance]

    We still compute normalization stats over the whole dataset (positions, velocities).
    """

    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            data = pkl.load(f)

        self.ICs = data['ICs']           # list of dicts with "data": list of BH objects
        self.Ns = data['Ns']             # number of BHs per simulation
        self.Ms = data['Ms']             # masses  (not used yet, but you can add)
        self.Rs = data['Rs']             # scale radii (not used yet, but you can add)
        self.Finals = data['Final_Data'] # final positions

        print(f"Loaded {len(self.ICs)} simulations")
        print(f"N range: {int(self.Ns.min())} - {int(self.Ns.max())}")

        self.max_n = int(self.Ns.max())

        # Compute normalization stats for pos/vel
        self._compute_normalization()

    def _compute_normalization(self):
        all_pos = []
        all_vel = []

        for ic_dict in self.ICs:
            ic_list = ic_dict["data"]
            for bh in ic_list:
                all_pos.append(bh.position)
                all_vel.append(bh.velocity)

        all_pos = np.array(all_pos)
        all_vel = np.array(all_vel)

        self.pos_mean = all_pos.mean(axis=0)
        self.pos_std = all_pos.std(axis=0) + 1e-8
        self.vel_mean = all_vel.mean(axis=0)
        self.vel_std = all_vel.std(axis=0) + 1e-8

        print(f"Position scale (avg std): {self.pos_std.mean():.3f}")
        print(f"Velocity scale (avg std): {self.vel_std.mean():.3f}")

    def __len__(self):
        return len(self.ICs)

    def __getitem__(self, idx):
        """
        Returns a torch_geometric.data.Data object for a single simulation.

        Data.x        : [N, 6]   node features (pos+vel, normalized)
        Data.edge_index : [2, E] edge list (i->j)
        Data.edge_attr  : [E, 4] edge features (dx, dy, dz, dist)
        Data.y        : [N, 3]   target final positions (normalized)
        """
        idx = int(idx)

        ic_dict = self.ICs[idx]
        final_dict = self.Finals[idx]

        ic_list = ic_dict["data"]
        # Final data may be dict or list depending on how you stored it
        final_list = final_dict["data"] if isinstance(final_dict, dict) else final_dict

        n = len(ic_list)  # number of BHs in this simulation

        # Prepare arrays
        input_features = np.zeros((n, 6), dtype=np.float32)
        target_positions = np.zeros((n, 3), dtype=np.float32)

        # Fill with normalized data
        for i in range(n):
            bh_ic = ic_list[i]
            bh_final = final_list[i]

            # Normalize (x - mean)/std
            pos = (bh_ic.position - self.pos_mean) / self.pos_std
            vel = (bh_ic.velocity - self.vel_mean) / self.vel_std
            pos_final = (bh_final.position - self.pos_mean) / self.pos_std

            input_features[i, :3] = pos
            input_features[i, 3:] = vel
            target_positions[i] = pos_final

        # Convert to tensors
        x = torch.from_numpy(input_features)           # [N, 6]
        y = torch.from_numpy(target_positions)         # [N, 3]
        pos = x[:, :3]                                 # use initial positions for edges

        # Build fully-connected directed graph (i -> j, i != j)
        idx_nodes = torch.arange(n, dtype=torch.long)
        row = idx_nodes.repeat_interleave(n)           # sender indices
        col = idx_nodes.repeat(n)                      # receiver indices
        mask = row != col                              # remove self-loops
        row, col = row[mask], col[mask]
        edge_index = torch.stack([row, col], dim=0)    # [2, E]

        # Edge features: relative vector + distance
        src = pos[row]          # sender positions
        dst = pos[col]          # receiver positions
        rel = dst - src         # [E, 3]
        dist = torch.norm(rel, dim=-1, keepdim=True) + 1e-8
        edge_attr = torch.cat([rel, dist], dim=-1)     # [E, 4]

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        return data


# ====================================================================
# MPNN LAYER + MODEL
# ====================================================================

class MPNNLayer(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, node_hidden_dim):
        # aggregation can be "add", "mean", or "max"
        super().__init__(aggr="add")

        # φ: message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_in_dim + edge_in_dim, node_hidden_dim),
            nn.ReLU(),
            nn.Linear(node_hidden_dim, node_hidden_dim),
            nn.ReLU(),
        )

        # ψ: update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(node_in_dim + node_hidden_dim, node_in_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, node_in_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_in_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        x_i: features of receiver node
        x_j: features of sender node
        edge_attr: edge features e_ij
        """
        m_in = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(m_in)

    def update(self, aggr_out, x):
        """
        aggr_out: aggregated messages for each node
        x: original node features
        """
        u_in = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(u_in)


class MPNNModel(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, node_hidden_dim,
                 num_layers, node_out_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            MPNNLayer(node_in_dim, edge_in_dim, node_hidden_dim)
            for _ in range(num_layers)
        ])

        # Final per-node readout MLP (maps node features → final position)
        self.readout = nn.Sequential(
            nn.Linear(node_in_dim, node_in_dim),
            nn.ReLU(),
            nn.Linear(node_in_dim, node_out_dim)
        )

    def forward(self, data):
        x = data.x               # [total_nodes_in_batch, node_in_dim]
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        out = self.readout(x)    # [total_nodes_in_batch, node_out_dim]
        return out


# ====================================================================
# TRAINING / EVALUATION LOOPS
# ====================================================================

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)   # [num_nodes_in_batch, node_out_dim]
        loss = loss_fn(pred, batch.y)  # batch.y: same shape

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            total_loss += loss.item()

    return total_loss / len(loader)

def visualize_one_graph(model, loader, dataset, device, save_prefix=None):
    """
    Take one batch from loader, pick the first graph in that batch,
    compare predicted vs true final positions, and make a couple of plots.
    """
    model.eval()

    # Get one batch
    batch = next(iter(loader))
    batch = batch.to(device)

    with torch.no_grad():
        pred_norm = model(batch)   # [total_nodes_in_batch, 3]
        true_norm = batch.y        # [total_nodes_in_batch, 3]

    # Figure out which nodes belong to the first graph in the batch
    # PyG stores this as batch.batch: [num_nodes] with graph indices
    if hasattr(batch, "batch"):
        mask = (batch.batch == 0)      # nodes of graph 0
    else:
        # If batch_size == 1, all nodes belong to one graph
        mask = torch.ones(pred_norm.size(0), dtype=torch.bool, device=device)

    pred_norm = pred_norm[mask]   # [N_nodes_graph0, 3]
    true_norm = true_norm[mask]

    # Denormalize back to physical units (kpc)
    pos_mean = torch.from_numpy(dataset.pos_mean).to(device)  # [3]
    pos_std  = torch.from_numpy(dataset.pos_std).to(device)   # [3]

    pred = pred_norm * pos_std + pos_mean    # [N, 3]
    true = true_norm * pos_std + pos_mean

    pred = pred.cpu().numpy()
    true = true.cpu().numpy()

    # -------------------------------------------------
    # 1) Scatter plot: true vs predicted in x–y plane
    # -------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(true[:, 0], true[:, 1], label="True", alpha=0.7)
    plt.scatter(pred[:, 0], pred[:, 1], label="Pred", alpha=0.7, marker="x")
    plt.xlabel("x [kpc]")
    plt.ylabel("y [kpc]")
    plt.title("Final positions: true vs predicted (one simulation)")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_xy.png", dpi=150)
    plt.show()

    # -------------------------------------------------
    # 2) 3D scatter (optional)
    # -------------------------------------------------
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Scatter true and pred
        ax.scatter(true[:,0], true[:,1], true[:,2], label="True", color="blue", s=20)
        ax.scatter(pred[:,0], pred[:,1], pred[:,2], label="Pred", color="orange", s=20)

        # Draw lines from true → pred
        for i in range(len(true)):
            ax.plot(
                [true[i,0], pred[i,0]],
                [true[i,1], pred[i,1]],
                [true[i,2], pred[i,2]],
                color="gray",
                alpha=0.5
            )

        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.set_zlabel("z [kpc]")
        ax.set_title("Final positions: True vs Predicted\n(Connected per-particle)")
        ax.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("3D plot skipped:", e)

    # -------------------------------------------------
    # 3) Error histogram
    # -------------------------------------------------
    err = np.linalg.norm(pred - true, axis=1)  # per-BH position error [kpc]

    plt.figure(figsize=(6, 4))
    plt.hist(err, bins=30)
    plt.xlabel("Position error [kpc]")
    plt.ylabel("Count")
    plt.title("Per-black-hole final position error")
    plt.tight_layout()
    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_err_hist.png", dpi=150)
    plt.show()

    print(f"Mean |error| [kpc]: {err.mean():.3f}")
    print(f"Median |error| [kpc]: {np.median(err):.3f}")

# ====================================================================
# MAIN
# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the pickled dataset")
    parser.add_argument("--fig_n", type=str, help = "Name of Figures", default = "mpnn_example")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.2)
    args = parser.parse_args()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("Using device:", device)

    # Load dataset (each item is a PyG Data graph)
    dataset = TrajectoryDataset(args.data)

    # Train/val split
    n_total = len(dataset)
    n_val = int(n_total * args.val_frac)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model dimensions
    node_in_dim = 6   # [x,y,z,vx,vy,vz]
    edge_in_dim = 4   # [dx,dy,dz,dist]
    node_out_dim = 3  # [x_final,y_final,z_final]

    model = MPNNModel(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        node_hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        node_out_dim=node_out_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    print("Model:", model)

    best_val = float("inf")    

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_mpnn.pt")

        print(f"Epoch {epoch:03d} | "
              f"train_loss={train_loss:.6f} | "
              f"val_loss={val_loss:.6f} | "
              f"best_val={best_val:.6f}")

    print("Training finished. Best validation loss:", best_val)
    # Load best model weights (just in case they’re not the last epoch)
    model.load_state_dict(torch.load("best_mpnn.pt", map_location=device))

    # Make sure we use a loader with a small batch size for nice plots
    # You can reuse val_ds but with batch_size=1
    viz_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    visualize_one_graph(model, viz_loader, dataset, device, save_prefix=args.fig_n)


if __name__ == "__main__":
    main()
