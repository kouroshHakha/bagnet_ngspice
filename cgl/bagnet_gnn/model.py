import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from cgl.models.gnn import DeepGENNet, Node2GraphEmb

# TODO: test

class SymLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        assert in_features % 2 == 0
        assert out_features % 2 == 0

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((out_features // 2, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features // 2, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    @property
    def weight_layer(self):
        return torch.cat([self.weight, self.weight.flip(0, 1)], 0)

    @property
    def bias_layer(self):
        return torch.cat([self.bias, self.bias.flip(0)], 0)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # assumes input is already structured in format of x1,...,xk, xk, ..., x1
        return F.linear(input, self.weight_layer, self.bias_layer)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class MLPBinaryClassifier(nn.Module):
    def __init__(self, feature_dim, n_layers, hidden_dim, drop_out=0.):
        super().__init__()

        assert hidden_dim % 2 == 0

        dims = [feature_dim] + [hidden_dim] * n_layers

        nets = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            nets.append(nn.Linear(in_dim, out_dim, bias=True))
            nets.append(nn.ReLU())
            nets.append(nn.Dropout(drop_out))
        nets.append(nn.Linear(hidden_dim, 2, bias=True))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        nn_out = self.net(x)
        output = {'prob': nn_out.softmax(-1), 'logit': nn_out}
        return output
        
class ComparisonHead(nn.Module):

    def __init__(self, feature_dim, n_layers, hidden_dim, drop_out=0.):
        super().__init__()

        assert hidden_dim % 2 == 0

        dims = [2 * feature_dim] + [hidden_dim] * n_layers

        nets = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            nets.append(SymLinear(in_dim, out_dim, bias=True))
            nets.append(nn.ReLU())
            nets.append(nn.Dropout(drop_out))
        nets.append(SymLinear(hidden_dim, 2, bias=True))
        self.net = nn.Sequential(*nets)

    
    def forward(self, feature_a, feature_b):

        nn_in = torch.cat([feature_a, feature_b.flip(-1)], -1)
        nn_out = self.net(nn_in)
        output = {'prob': nn_out.softmax(-1), 'logit': nn_out}
        return output



class FeatureExtractorLinear(nn.Module):

    def __init__(self, input_features, n_layers, output_features, hidden_dim, drop_out=0.):
        super().__init__()

        self.out_features = output_features
        dims = [input_features] + [hidden_dim] * n_layers + [output_features]

        nets = []
        # TODO: whether should we include relu + dropout in the output of feature extractor
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            nets.append(nn.Linear(in_dim, out_dim, bias=True))
            nets.append(nn.ReLU())
            nets.append(nn.Dropout(drop_out))
        # nets.append(nn.Linear(hidden_dim, output_features, bias=True))

        self.net = nn.Sequential(*nets)

    def forward(self, input_features):
        nn_out = self.net(input_features)
        output = {'feature': nn_out}
        return output

class FeatureExtractorGNN(nn.Module):

    def __init__(self, gnn_ckpt_path, output_features, freeze=False, use_pooling=False, rand_init=False):
        super().__init__()
        self.gnn = DeepGENNet.load_from_checkpoint(gnn_ckpt_path)

        if rand_init:
            self.gnn.reset_parameters()

        self.freeze = freeze
        self.use_pooling = use_pooling
        self.out_features = output_features

        if self.freeze:
            print('Backbone is frozen.')
            self.gnn.freeze()

        if self.use_pooling:
            self.node2graph = None
        else:
            # n_embs = 8, n_layers=3
            self.node2graph = Node2GraphEmb(output_features // 8, 3, output_features, self.gnn.config.hidden_channels)


    def forward(self, batch):
        
        input_struct = self.gnn.get_input_struct(batch)
        node_embs = self.gnn.get_node_features(input_struct, return_masked=False)

        n_nodes = len(batch.x) // batch.num_graphs
        node_embs = torch.stack([node_embs[i::n_nodes] for i in range(n_nodes)], 1)
        # node_embs = torch.stack(torch.split(node_embs, batch.num_graphs, 0), dim=1)
        if self.use_pooling:
            graph_embs = node_embs.mean(1)
        else:
            graph_embs = self.node2graph(node_embs)

        output = {'feature': graph_embs}
        return output


class BagNetComparisonModel(nn.Module):

    def __init__(
        self, 
        comparison_kwrds,
        feature_exractor_config,
        comparison_model_config,
        is_gnn=False,
    ):
        super().__init__()

        self.is_gnn = is_gnn
        if is_gnn:
            self.feature_extractor = FeatureExtractorGNN(**feature_exractor_config)
            # self.feat_dout = nn.Dropout(p=0.2)
        else:
            self.feature_extractor = FeatureExtractorLinear(**feature_exractor_config)

        feature_dim = self.feature_extractor.out_features

        self.comparison_heads = nn.ModuleDict()
        for key in comparison_kwrds:
            self.comparison_heads[key] = ComparisonHead(feature_dim, **comparison_model_config)


    def compute_loss(self, target_labels, nn_output):
        output_logits = nn_output['logit']
        loss = nn.CrossEntropyLoss()(input=output_logits, target=target_labels)
        return loss

    def forward(self, input_dict, compute_loss=True):

        input_a = input_dict['input_a']
        input_b = input_dict['input_b']

        features_a = self.feature_extractor(input_a)['feature']
        features_b = self.feature_extractor(input_b)['feature']

        # if self.is_gnn:
        #     features_a = self.feat_dout(features_a)
        #     features_b = self.feat_dout(features_b)

        outputs = {}
        for key in self.comparison_heads:
            outputs[key] = self.comparison_heads[key](features_a, features_b)

        losses = {}
        if compute_loss:
            for key in input_dict:
                if key not in ('input_a', 'input_b'):
                    losses[key] = self.compute_loss(input_dict[key], outputs[key])

        return dict(outputs=outputs, losses=losses)



class BagNetComparisonModel_v2(nn.Module):

    def __init__(
        self, 
        comparison_kwrds,
        feature_exractor_config,
        comparison_model_config,
        meet_spec_config,
        is_gnn=False,
    ):
        super().__init__()

        self.is_gnn = is_gnn
        if is_gnn:
            self.feature_extractor = FeatureExtractorGNN(**feature_exractor_config)
        else:
            self.feature_extractor = FeatureExtractorLinear(**feature_exractor_config)

        feature_dim = self.feature_extractor.out_features

        self.comparison_heads = nn.ModuleDict()
        for key in comparison_kwrds:
            self.comparison_heads[key] = ComparisonHead(feature_dim, **comparison_model_config)

        self.meet_spec_nn = nn.ModuleDict()
        for key in comparison_kwrds:
            self.meet_spec_nn[key] = MLPBinaryClassifier(feature_dim, **meet_spec_config)


    def compute_loss(self, target_labels, nn_output):
        output_logits = nn_output['logit']
        loss = nn.CrossEntropyLoss()(input=output_logits, target=target_labels)
        return loss

    def forward(self, input_dict, compute_loss=True):
        input_a = input_dict['input_a']
        input_b = input_dict['input_b']

        features_a = self.feature_extractor(input_a)['feature']
        features_b = self.feature_extractor(input_b)['feature']

        outputs = {'a_better_than_b': {}, 'a_meets_specs': {}, 'b_meets_specs': {}}
        for key in self.comparison_heads:
            outputs['a_better_than_b'][key] = self.comparison_heads[key](features_a, features_b)

        for key in self.meet_spec_nn:
            outputs['a_meets_specs'][key] = self.meet_spec_nn[key](features_a)
            outputs['b_meets_specs'][key] = self.meet_spec_nn[key](features_b)

        if compute_loss:
            losses = {'comp': {}, 'meet_spec': {}, }
            for key in self.comparison_heads:
                # if key not in ('input_a', 'input_b'):
                    losses['comp'][key] = nn.CrossEntropyLoss()(input=outputs['a_better_than_b'][key]['logit'], target=input_dict['a_better_than_b'][key])
                    losses['meet_spec'][key] = 0.5 * (nn.CrossEntropyLoss()(input=outputs['a_meets_specs'][key]['logit'], target=input_dict['a_meets_specs'][key]) + \
                        nn.CrossEntropyLoss()(input=outputs['b_meets_specs'][key]['logit'], target=input_dict['b_meets_specs'][key]))
        else:
            losses = {}

        return dict(outputs=outputs, losses=losses)

