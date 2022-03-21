
from cgl.bagnet_gnn.model import BagNetComparisonModel, ComparisonHead, SymLinear
from cgl.bagnet_gnn.data import MAX_SPECS, MIN_SPECS
import torch


from cgl.utils.pdb import register_pdb_hook
register_pdb_hook()

COMPARISON_KWRDS = MAX_SPECS + MIN_SPECS

class TestSymLinear:

    def test_fwd_symmetry(self):
        layer = SymLinear(20, 10, bias=True)
        x1 = torch.randn(16, 10)
        x2 = torch.randn(16, 10)
        x_a = torch.cat([x1, x2.flip(-1)], -1)
        x_b = x_a.flip(-1)

        layer.eval()
        out_a = layer(x_a)
        out_b = layer(x_b)

        assert torch.all(torch.isclose(out_a, out_b.flip(-1), atol=1e-6))

    def test_bwd_symmetry(self):
        # in case of a softmax applied to output the grad diff of dim=0 vs. dim=1 should be a negative sign
        layer = SymLinear(20, 2, bias=True)
        x1 = torch.randn(16, 10)
        x2 = torch.randn(16, 10)

        x_a = torch.cat([x1, x2.flip(-1)], -1)
        x_b = x_a.flip(-1)
        out_a = layer(x_a)
        out_b = layer(x_b)

        # get grads of the first round
        out_a.softmax(-1)[:, 0].mean().backward()
        w_grad_a = layer.weight.grad.clone()
        b_grad_a = layer.bias.grad.clone()

        # reset grads
        layer.weight.grad.data = torch.zeros_like(layer.weight.grad.data)
        layer.bias.grad.data = torch.zeros_like(layer.bias.grad.data)

        out_b.softmax(-1)[:, 0].mean().backward()
        w_grad_b = layer.weight.grad.clone()
        b_grad_b = layer.bias.grad.clone()

        assert torch.all(torch.isclose(w_grad_a, -w_grad_b, atol=1e-6))
        assert torch.all(torch.isclose(b_grad_a, -b_grad_b, atol=1e-6))


class TestComparisonHead:

    def test_fwd(self):

        layer = ComparisonHead(20, 3, 128, drop_out=0.0)
        x1 = torch.randn(16, 20)
        x2 = torch.randn(16, 20)

        layer.eval()
        output_a = layer(x1, x2)
        output_b = layer(x2, x1)

        assert torch.all(torch.isclose(output_a['prob'][:, 0], output_b['prob'][:, 1], atol=1e-6))
        assert torch.all(torch.isclose(output_a['prob'][:, 1], output_b['prob'][:, 0], atol=1e-6))


class TestBagNetLinear:

    def setup_model(self):
        self.input_dim = 8
        self.feature_dim = 20

        feature_ext_config = dict(
            input_features=self.input_dim,
            output_features=self.feature_dim,
            hidden_dim=20,
            n_layers=2,
            drop_out=0.2,
        )

        comparison_config = dict(
            hidden_dim=20,
            n_layers=1,
            drop_out=0.2,
        )

        model = BagNetComparisonModel(
            comparison_kwrds=COMPARISON_KWRDS,
            feature_exractor_config=feature_ext_config,
            comparison_model_config=comparison_config, 
            is_gnn=False
        )

        return model


    def test_fwd(self):
        
        model = self.setup_model()

        x1 = torch.randn(16, self.input_dim)
        x2 = torch.randn(16, self.input_dim)

        specs = {}
        for key in COMPARISON_KWRDS:
            specs[key] = torch.randint(2, size=(16,))

        input_dict = dict(input_a=x1, input_b=x2, **specs)
        output_dict = model(input_dict)





if __name__ == '__main__':
    
    symLinear_testunit = TestSymLinear()
    symLinear_testunit.test_fwd_symmetry()
    symLinear_testunit.test_bwd_symmetry()

    comparisonModel_testunit = TestComparisonHead()
    comparisonModel_testunit.test_fwd()

    bagnetlinear_testunit = TestBagNetLinear()
    bagnetlinear_testunit.test_fwd()

