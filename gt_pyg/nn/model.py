# Standard
from typing import List, Optional

# Third party
import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn.aggr import MultiAggregation

# GT-PyG
from .gt_conv import GTConv
from .mlp import MLP


class GraphTransformerNet(nn.Module):
    """
    Graph Transformer Network.

    Reference:
      1. A Generalization of Transformer Networks to Graphs
         https://arxiv.org/abs/2012.09699
    """

    def __init__(
        self,
        node_cont_dim_in: int,
        node_cat_counts: List[int],
        edge_cat_counts: List[int] = None,
        cat_hidden_dim: int = 8,
        pe_in_dim: Optional[int] = None,
        hidden_dim: int = 128,
        output_dim: int = 1,
        norm: str = "bn",
        gate=False,
        qkv_bias=False,
        num_gt_layers: int = 4,
        num_mlp_layers: int = 1,
        num_heads: int = 8,
        gt_aggregators: List[str] = ["sum"],
        aggregators: List[str] = ["sum"],
        act: str = "relu",
        dropout: float = 0.0,
    ):
        """
        Args:
            node_dim_in (int): Dimension of input node features.
            edge_dim_in (int, optional): Dimension of input edge features.
                                         Default is None.
            pe_in_dim (int, optional): Dimension of positional encoding input.
                                       Default is None.
            hidden_dim (int, optional): Dimension of hidden layers.
                                        Default is 128.
            gate (bool, optional): Use a gate attantion mechanism.
                                   Default is False
            qkv_bias (bool, optional): Bias in the attention mechanism.
                                       Default is False
            norm (str, optional): Normalization method.
                                  Default is "bn" (batch norm).
            num_gt_layers (int, optional): Number of Graph Transformer layers.
                                           Default is 4.
            num_heads (int, optional): Number of attention heads. Default is 8.
            gt_aggregators (List[str], optional): Aggregation methods for the messages aggregation.
                                           Default is ["sum"].
            aggregators (List[str], optional): Aggregation methods for global pooling.
                                               Default is ["sum"].
            act (str, optional): Activation function.
                                 Default is "relu".
            dropout (float, optional): Dropout probability.
                                       Default is 0.0.
        """

        super(GraphTransformerNet, self).__init__()

        self.node_embs = nn.ModuleList(
            [nn.Embedding(cat_count, cat_hidden_dim) for cat_count in node_cat_counts]
        )
        self.node_linear = nn.Linear(
            cat_hidden_dim * len(node_cat_counts) + node_cont_dim_in, hidden_dim
        )
        self.edge_embs = nn.ModuleList(
            [nn.Embedding(cat_count, cat_hidden_dim) for cat_count in edge_cat_counts]
        )
        self.edge_linear = nn.Linear(cat_hidden_dim * len(edge_cat_counts), hidden_dim)

        if pe_in_dim:
            self.pe_emb = nn.Linear(pe_in_dim, hidden_dim, bias=False)
        else:
            self.pe_emb = self.register_parameter("pe_emb", None)

        self.gt_layers = nn.ModuleList()
        for _ in range(num_gt_layers):
            self.gt_layers.append(
                GTConv(
                    node_in_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    edge_in_dim=hidden_dim,
                    num_heads=num_heads,
                    act=act,
                    dropout=dropout,
                    norm=norm,
                    gate=gate,
                    qkv_bias=qkv_bias,
                    aggregators=gt_aggregators,
                )
            )

        self.global_pool = MultiAggregation(aggregators, mode="cat")

        num_aggrs = len(aggregators)
        self.mu_mlp = MLP(
            input_dim=num_aggrs * hidden_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dim,
            num_hidden_layers=num_mlp_layers,
            dropout=dropout,
            act=act,
        )
        self.log_var_mlp = MLP(
            input_dim=num_aggrs * hidden_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dim,
            num_hidden_layers=num_mlp_layers,
            dropout=dropout,
            act=act,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the embedding parameters of the model using Xavier uniform initialization.

        Note: The input and the output of the embedding layers does not pass through the activation layer,
              so the variance estimation differs by a factor of two from the default
              kaiming_uniform initialization.
        """
        for emb in self.node_embs + self.edge_embs:
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.node_linear.weight)
        nn.init.xavier_uniform_(self.edge_linear.weight)
        nn.init.xavier_uniform_(self.pe_emb.weight)

    def forward(
        self,
        batch: Batch,
        zero_var: bool = False,
    ) -> Tensor:
        """
        Forward pass of the Graph Transformer Network.

        Args:
            x (Tensor): Input node features.
            edge_index (Tensor): Graph edge indices.
            edge_attr (Tensor): Edge features.
            pe (Tensor): Positional encoding.
            batch (Batch): Batch indices.
            zero_var (bool, optional): Flag to zero out the log variance.
                                       Default is False.

        Returns:
            Tensor: The output of the forward pass.
        """
        # concat node categorical embeddings
        x_cat = [
            self.node_embs[i](batch.x_cat[:, i]) for i in range(len(self.node_embs))
        ]
        x_cat = torch.cat(x_cat, dim=-1)
        x = torch.cat([x_cat, batch.x_cont], dim=-1)
        x = self.node_linear(x)

        # concat edge categorical embeddings
        edge_attr = torch.cat(
            [
                self.edge_embs[i](batch.edge_attr[:, i])
                for i in range(len(self.edge_embs))
            ],
            dim=-1,
        )
        edge_attr = self.edge_linear(edge_attr)

        if self.pe_emb is not None:
            x = x + self.pe_emb(batch.pe)

        for gt_layer in self.gt_layers:
            (x, edge_attr) = gt_layer(
                x=x, edge_index=batch.edge_index, edge_attr=edge_attr
            )

        x = self.global_pool(x, batch.batch)
        mu = self.mu_mlp(x)
        log_var = self.log_var_mlp(x)
        if zero_var:
            std = torch.zeros_like(log_var)
        else:
            std = torch.exp(0.5 * log_var)

        if self.training:
            eps = torch.randn_like(std)
            return mu + std * eps, std
        else:
            return mu, std

    def num_parameters(self) -> int:
        """
        Calculate the total number of trainable parameters in the model.

        Returns:
            int: The total number of trainable parameters.
        """
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        count = sum([p.numel() for p in trainable_params])
        return count
