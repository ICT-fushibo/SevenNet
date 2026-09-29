import unittest

import torch
from md_benchmark.stress_capture import sevennet_stress
from md_benchmark.stress_test_support import assert_replay_stable

from sevenn import _keys as KEY
from sevenn.md_stages.opt4_fusion import CompactSupportCutoff
from sevenn.nn.edge_embedding import PolynomialCutoff
from sevenn.nn.force_output import ForceStressOutputFromEdge


class SevenNetStressTests(unittest.TestCase):
    def test_far_padding_cutoff_value_vjp_replay(self):
        for dtype in (torch.float32, torch.float64):
            radius = torch.tensor(
                [1.0, 4.9, 5.0, 30.0, 1e6],
                device='cuda:0',
                dtype=dtype,
                requires_grad=True,
            )
            cutoff = CompactSupportCutoff(PolynomialCutoff(5.0), 5.0).to('cuda:0')

            def body(cutoff=cutoff, radius=radius):
                value = cutoff(radius)
                gradient = torch.autograd.grad(value.sum(), radius)[0]
                return value.detach(), gradient.detach()

            value, gradient = body()
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertEqual(float(value[2:].abs().max()), 0)
            self.assertEqual(float(gradient[2:].abs().max()), 0)
            assert_replay_stable(body)

    def test_native_edge_gradient_stress_and_cuda_replay(self):
        for dtype in (torch.float32, torch.float64):
            edges = torch.tensor(
                [[1.0, 0.2, 0.4], [-1.0, -0.2, -0.4], [1.0, 0.2, 0.4], [5.0, 0, 0]],
                dtype=dtype,
                device='cuda:0',
                requires_grad=True,
            )
            mask = torch.tensor([1.0, 1, 1, 0], dtype=dtype, device='cuda:0')
            module = ForceStressOutputFromEdge(compute_stress=True).eval()
            module._is_batch_data = False
            numbers = torch.ones(3, dtype=torch.long, device=edges.device)
            index = torch.tensor([[0, 1, 0, 2], [1, 0, 1, 2]], device=edges.device)
            volume = torch.tensor(60.0, dtype=dtype, device=edges.device)

            def body(
                module=module,
                numbers=numbers,
                edges=edges,
                index=index,
                volume=volume,
                mask=mask,
            ):
                output = module(
                    {
                        KEY.ATOMIC_NUMBERS: numbers,
                        KEY.EDGE_VEC: edges,
                        KEY.EDGE_IDX: index,
                        KEY.CELL_VOLUME: volume,
                        KEY.PRED_TOTAL_ENERGY: (0.5 * edges.square().sum(-1) * mask)
                        .sum()
                        .reshape(1),
                    }
                )
                # Mirror _WholeStepPotential._static_forward: retain values,
                # never the eager stress graph (rij * dE/dr still has grad_fn).
                # Otherwise this default-stream AccumulateGrad survives into
                # the side-stream capture (cudaErrorStreamCaptureImplicit).
                return output[KEY.PRED_FORCE].detach(), sevennet_stress(
                    output[KEY.PRED_STRESS]
                ).detach()

            force, stress = body()
            self.assertFalse(force.requires_grad)
            self.assertIsNone(stress.grad_fn)
            self.assertFalse(stress.requires_grad)
            with torch.no_grad():
                reference = edges.T @ (edges * mask[:, None]) / 60
            torch.testing.assert_close(stress, reference)
            assert_replay_stable(body)
            mask.zero_()
            force, stress = body()
            self.assertEqual(float(stress.abs().max()), 0)
            self.assertEqual(float(force.abs().max()), 0)
            assert_replay_stable(body)


if __name__ == '__main__':
    unittest.main()
