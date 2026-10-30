"""Offline checks for the qualification-only command renderer in tensorfold.launch."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tensorfold import launch
from tensorfold.variants import build, patch

EXPERIMENTAL = '1.2.6-resumable'


def scratch(prefix):
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=os.environ.get('DS41F_TEST_ARTIFACTS')))
    print('Retained ' + EXPERIMENTAL + ' check resource: ' + str(path))
    return path


def synthetic_receipt(built, recipe_name):
    spec, identity = patch.recipe(recipe_name)
    binaries = {}
    for name in ['tensorfold-native', 'tf-dsv41-lanes']:
        (built / name).write_bytes(b'test-only-never-executable')
        binaries[name] = dict(path=name, sha256=patch.digest((built / name).read_bytes()))
    receipt = dict(schema='ds41f.native-build.v1', recipe=spec['recipe'],
                  source_tree_sha256=spec['source_tree_sha256'],
                  recipe_manifest_sha256=identity,
                  source_archive_sha256=spec['base_archive_sha256'],
                  adaptive_learning=spec['adaptive_learning'], binaries=binaries)
    (built / 'BUILD.json').write_text(json.dumps(receipt))
    return receipt


def topology(root, api_host='127.0.0.1'):
    masks = {0: [0, 12, 3], 1: [3, 0, 12], 2: [12, 3, 0]}
    return {
        'master_address': '10.20.30.40',
        'master_port': 29660,
        'api_host': api_host,
        'api_port': 8003,
        'nccl_interface': 'eth0',
        'nccl_hcas': ['mlx5_0', 'mlx5_1', 'mlx5_2', 'mlx5_3'],
        'nccl_address_range': '10.20.30.0/24',
        'ranks': [
            {'rank': rank, 'root': str(root),
             'rdma_devices': ['mlx5_0', 'mlx5_1', 'mlx5_2', 'mlx5_3'],
             'gid_index': 3, 'peer_masks': masks[rank]}
            for rank in range(3)
        ],
    }


def installed(root):
    built = root / 'built'
    built.mkdir()
    synthetic_receipt(built, 'v126-resumable')
    result = build.install(built, root, EXPERIMENTAL)
    config = topology(root)
    (root / 'cluster.json').write_text(json.dumps(config))
    return config, Path(result['installed'])


class QualificationCommand(unittest.TestCase):
    def test_normal_command_rejects_unqualified_profile_before_missing_files(self):
        root = scratch('ds41f-126r-qual-guard-')
        config, _ = installed(root)
        for rank in range(3):
            with self.assertRaises(ValueError):
                launch.command(config, EXPERIMENTAL, rank, validate_files=True)

    def test_qualification_command_reaches_missing_file_validation(self):
        root = scratch('ds41f-126r-qual-files-')
        config, _ = installed(root)
        for rank in range(3):
            with self.assertRaises(FileNotFoundError):
                launch.qualification_command(config, EXPERIMENTAL, rank)

    def test_qualification_command_still_rejects_mutated_binary_or_receipt(self):
        root = scratch('ds41f-126r-qual-mutation-')
        config, dest = installed(root)
        binary = dest / 'tensorfold-native'
        original = binary.read_bytes()
        binary.write_bytes(original + b'x')
        with self.assertRaises(ValueError):
            launch.qualification_command(config, EXPERIMENTAL, 0)
        binary.write_bytes(original)
        receipt_path = dest / 'BUILD.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['source_tree_sha256'] = '0' * 64
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):
            launch.qualification_command(config, EXPERIMENTAL, 0)

    def test_qualification_command_without_validation_renders_loopback_benchmark_head(self):
        root = scratch('ds41f-126r-qual-render-')
        config = topology(root, api_host='0.0.0.0')
        head_argv, head_env = launch.qualification_command(config, EXPERIMENTAL, 0, validate_files=False)
        self.assertEqual(head_argv[head_argv.index('--host') + 1], '127.0.0.1')
        self.assertIn('--benchmark-control', head_argv)
        public_head_argv, public_head_env = launch.command(config, EXPERIMENTAL, 0,
                                                         benchmark=True, validate_files=False)
        self.assertEqual(head_argv, public_head_argv)
        self.assertEqual(head_env, public_head_env)
        for rank in (1, 2):
            argv, env = launch.qualification_command(config, EXPERIMENTAL, rank, validate_files=False)
            public_argv, public_env = launch.command(config, EXPERIMENTAL, rank,
                                                   benchmark=True, validate_files=False)
            self.assertEqual(argv, public_argv)
            self.assertEqual(env, public_env)


if __name__ == '__main__':
    unittest.main()
