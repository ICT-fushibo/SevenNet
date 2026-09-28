import unittest

import torch
from md_benchmark.stress_capture import sevennet_stress
from md_benchmark.stress_test_support import assert_replay_stable

from sevenn import _keys as KEY
from sevenn.nn.force_output import ForceStressOutputFromEdge


class SevenNetStressTests(unittest.TestCase):
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
                return output[KEY.PRED_FORCE], sevennet_stress(
                    output[KEY.PRED_STRESS]
                )

            _, stress = body()
            torch.testing.assert_close(
                stress, edges.T @ (edges * mask[:, None]) / 60
            )
            assert_replay_stable(body)
            mask.zero_()
            force, stress = body()
            self.assertEqual(float(stress.abs().max()), 0)
            self.assertEqual(float(force.abs().max()), 0)
            assert_replay_stable(body)


if __name__ == '__main__':
    unittest.main()
