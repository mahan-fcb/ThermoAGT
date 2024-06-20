import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing, DiffGroupNorm
from torch_geometric.utils import softmax as tg_softmax
from torch.nn import Sequential, Linear, Parameter

# Define globalATTENTION
class globalATTENTION(torch.nn.Module):
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2):
        super(globalATTENTION, self).__init__()
        self.act = act
        self.fc_layers = fc_layers
        self.batch_norm = batch_norm
        self.batch_track_stats = batch_track_stats == "True"
        self.dropout_rate = dropout_rate

        self.global_mlp = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(self.fc_layers + 1):
            if i == 0:
                lin = torch.nn.Linear(dim+20, dim)
            else:
                lin = torch.nn.Linear(dim, dim if i != self.fc_layers else 1)
            self.global_mlp.append(lin)

            if batch_norm == "True":
                bn = DiffGroupNorm(dim, 10, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

    def forward(self, x, batch, glbl_x):
        out = torch.cat([x, glbl_x], dim=-1)
        for i in range(len(self.global_mlp)):
            out = self.global_mlp[i](out)
            if i != len(self.global_mlp) - 1:
                out = getattr(F, self.act)(out)
            else:
                out = tg_softmax(out, batch)
        return out

# Define MLP
class MLP(nn.Module):
    def __init__(self, hs, act=None):
        super().__init__()
        self.hs = hs
        self.act = act
        num_layers = len(hs)
        layers = []
        for i in range(num_layers - 1):
            layers += [nn.Linear(hs[i], hs[i + 1])]
            if act is not None and i < num_layers - 2:
                layers += [act]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# Define AGATLayer
class AGATLayer(MessagePassing):
    def __init__(self, dim, activation, use_batch_norm, track_stats, dropout, fc_layers=2, **kwargs):
        super().__init__(aggr='add', flow='target_to_source', **kwargs)
        self.activation_func = getattr(F, activation)
        self.dropout = dropout
        self.dim = dim
        self.heads = 4
        self.weight = Parameter(torch.Tensor(dim * 2, self.heads * dim))
        self.attention = Parameter(torch.Tensor(1, self.heads, 2 * dim))
        self.bias = Parameter(torch.Tensor(dim)) if kwargs.get('add_bias', True) else None
        self.bn = nn.BatchNorm1d(self.heads) if use_batch_norm.lower() == "true" else None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attention)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        combined_x_i = self.activation_func(torch.matmul(torch.cat([x_i, edge_attr], dim=-1), self.weight)).view(-1, self.heads, self.dim)
        combined_x_j = self.activation_func(torch.matmul(torch.cat([x_j, edge_attr], dim=-1), self.weight)).view(-1, self.heads, self.dim)
        alpha = self.activation_func((torch.cat([combined_x_i, combined_x_j], dim=-1) * self.attention).sum(dim=-1))
        if self.bn:
            alpha = self.activation_func(self.bn(alpha))
        alpha = tg_softmax(alpha, edge_index_i)
        return (combined_x_j * F.dropout(alpha, p=self.dropout, training=self.training).view(-1, self.heads, 1)).transpose(0, 1)

    def update(self, aggr_out):
        return aggr_out.mean(dim=0) + (self.bias if self.bias is not None else 0)

# Define model
class ThermoAGTGA(torch.nn.Module):
    def __init__(self, out_dims=64, heads=4, pool="global_add_pool", pool_order="early", batch_norm="True", batch_track_stats="True", act="softplus", dropout_rate=0.0, pre_fc_count=1):
        super(ThermoAGTGA, self).__init__()
        self.batch_track_stats = batch_track_stats == "True"
        self.batch_norm = batch_norm == "True"
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout()
        self.global_att_LAYER = globalATTENTION(64, act, batch_norm, batch_track_stats, dropout_rate)
        self.pre_lin_list_E = torch.nn.ModuleList()
        self.pre_lin_list_N = torch.nn.ModuleList()

        for _ in range(pre_fc_count):
            embed_atm = nn.Sequential(MLP([20, 64, 64], act=nn.SiLU()), nn.LayerNorm(64))
            self.pre_lin_list_N.append(embed_atm)
            embed_bnd = nn.Sequential(MLP([50, 64, 64], act=nn.SiLU()), nn.LayerNorm(64))
            self.pre_lin_list_E.append(embed_bnd)

        self.conv1_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for _ in range(3):
            conv1 = AGATLayer(64, act, batch_norm, batch_track_stats, dropout_rate)
            self.conv1_list.append(conv1)
            if self.batch_norm:
                bn = DiffGroupNorm(64, 10, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

        self.lin_out = torch.nn.Linear(64, 1)

    def forward(self, wild_data, mutant_data):
        for i in range(1):
            out_w = self.pre_lin_list_N[i](wild_data.x[:, 1:21])
            out_w = F.softplus(out_w)
            out_ew = self.pre_lin_list_E[i](wild_data.edge_attr)
            out_ew = F.softplus(out_ew)

        prev_out_w = out_w
        for i in range(len(self.conv1_list)):
            out_w = self.conv1_list[i](out_w, wild_data.edge_index, out_ew)
            if self.batch_norm:
                out_w = self.bn_list[i](out_w)
        out_w = torch.add(out_w, prev_out_w)
        out_w = F.dropout(out_w, p=0.1, training=self.training)
        #out_aw       = self.global_att_LAYER(out_w,wild_data.batch,wild_data.x[:,21:])
        #out_w       = (out_w)*out_aw   
        out_w = getattr(torch_geometric.nn, self.pool)(out_w, wild_data.batch)

        for i in range(1):
            out_m = self.pre_lin_list_N[i](mutant_data.x[:, 1:21])
            out_m = F.softplus(out_m)
            out_em = self.pre_lin_list_E[i](mutant_data.edge_attr)
            out_em = F.softplus(out_em)

        prev_out_m = out_m
        for i in range(len(self.conv1_list)):
            out_m = self.conv1_list[i](out_m, mutant_data.edge_index, out_em)
            if self.batch_norm:
                out_m = self.bn_list[i](out_m)
        out_m = torch.add(out_m, prev_out_m)
        out_m = F.dropout(out_m, p=0.1, training=self.training)
        #out_am       = self.global_att_LAYER(out_m,mutant_data.batch,mutant_data.x[:,21:41])
        #out_m       = (out_m)*out_am
        out_m = getattr(torch_geometric.nn, self.pool)(out_m, mutant_data.batch)

        out = out_m - out_w
        out = self.lin_out(out)
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out

