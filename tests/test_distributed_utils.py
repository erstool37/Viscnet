import os
import sys
import unittest
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))


class DistributedUtilsTests(unittest.TestCase):
    def test_ddp_setup_sets_current_cuda_device_to_local_rank(self):
        from utils import ddp

        env = {"RANK": "2", "WORLD_SIZE": "8", "LOCAL_RANK": "3"}
        with mock.patch.dict(os.environ, env):
            with mock.patch("utils.ddp.torch.cuda.is_available", return_value=True):
                with mock.patch("utils.ddp.torch.cuda.set_device") as set_device:
                    with mock.patch("utils.ddp.dist.init_process_group") as init_process_group:
                        rank, world_size, local_rank = ddp.ddp_setup()

        self.assertEqual((rank, world_size, local_rank), (2, 8, 3))
        set_device.assert_called_once_with(3)
        init_process_group.assert_called_once_with("nccl", rank=2, world_size=8, init_method="env://")

    def test_broadcast_object_list_for_device_passes_explicit_device(self):
        from utils.ddp import broadcast_object_list_for_device

        payload = [None]
        with mock.patch("utils.ddp.dist") as dist:
            dist.get_world_size.return_value = 8

            result = broadcast_object_list_for_device(payload, src=0, device="cuda:3")

        self.assertIs(result, payload)
        dist.broadcast_object_list.assert_called_once_with(payload, src=0, device="cuda:3")


if __name__ == "__main__":
    unittest.main()
