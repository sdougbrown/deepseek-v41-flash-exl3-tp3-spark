"""Offline behavioral regression checks for the experimental 1.2.6-resumable recipe/profile."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tensorfold import launch
from tensorfold.variants import build, patch

ROOT = Path(__file__).resolve().parents[1]
STOCK = '1.2.6'
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


def topology(root):
    masks = {0: [0, 12, 3], 1: [3, 0, 12], 2: [12, 3, 0]}
    return {
        'master_address': '10.20.30.40',
        'master_port': 29660,
        'api_host': '127.0.0.1',
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


def profile(name):
    return json.loads((ROOT / 'tensorfold/profiles' / (name + '.json')).read_bytes())


class ResumableProfile(unittest.TestCase):
    def _run_main(self, config_path, variant, rank, run, exec_calls):
        argv = ['launch.py', '--config', str(config_path), '--variant', variant, '--rank', str(rank)]
        if run:
            argv.append('--run')
        with mock.patch.object(sys, 'argv', argv), \
             mock.patch.object(os, 'execvpe', side_effect=lambda *a, **k: exec_calls.append(a)), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            launch.main()
        return json.loads(out.getvalue())

    def test_dry_launch_never_execs_and_derives_from_experimental_profile(self):
        root = scratch('ds41f-126r-dry-')
        config_path = root / 'cluster.json'
        config_path.write_text(json.dumps(topology(root)))
        exp_profile, stock_profile = profile(EXPERIMENTAL), profile(STOCK)
        exec_calls = []
        for rank in range(3):
            exp = self._run_main(config_path, EXPERIMENTAL, rank, False, exec_calls)
            stock = self._run_main(config_path, STOCK, rank, False, exec_calls)
            self.assertEqual(exp['started'], False)
            self.assertEqual(stock['started'], False)
            for key, value in exp_profile['runtime_environment'].items():
                self.assertEqual(exp['environment'].get(key), value)
            diff = {k: v for k, v in exp_profile['runtime_environment'].items()
                   if stock_profile['runtime_environment'].get(k) != v}
            self.assertEqual(exp['environment'], {**stock['environment'], **diff})
            self.assertEqual(exp['argv'][1:], stock['argv'][1:])
            self.assertNotEqual(exp['argv'][0], stock['argv'][0])
            self.assertEqual(Path(exp['argv'][0]).parent, root / exp_profile['binary_subdir'])
            self.assertEqual(Path(stock['argv'][0]).parent, root / stock_profile['binary_subdir'])
            self.assertEqual(exp['environment']['TF_DS_RANK_CACHE'], str(root / exp_profile['rank_cache_subdir']))
            self.assertEqual(exp['environment']['TF_DS_ENGRAM'], str(root / exp_profile['engram_subdir']))
            if rank == 0:
                self.assertEqual(exp['argv'][2], str(root / exp_profile['head_model_subdir']))
                for key, value in exp_profile['head_environment'].items():
                    self.assertEqual(exp['environment'].get(key), value)
            else:
                self.assertNotIn(next(iter(exp_profile['head_environment'])), exp['environment'])
        self.assertEqual(exec_calls, [])

    def test_unqualified_experimental_profile_rejected_before_missing_paths_and_run_never_execs(self):
        root = scratch('ds41f-126r-guard-')
        built = root / 'built'
        built.mkdir()
        synthetic_receipt(built, 'v126-resumable')
        build.install(built, root, EXPERIMENTAL)
        config_path = root / 'cluster.json'
        config_path.write_text(json.dumps(topology(root)))
        config = json.loads(config_path.read_text())
        for rank in range(3):
            with self.assertRaises(ValueError):
                launch.command(config, EXPERIMENTAL, rank, validate_files=True)
        exec_calls = []
        for rank in range(3):
            with self.assertRaises(ValueError):
                self._run_main(config_path, EXPERIMENTAL, rank, True, exec_calls)
        self.assertEqual(exec_calls, [])
        self.assertFalse((root / 'kit').exists())
        self.assertFalse((root / 'weights').exists())

    def test_stock_receipt_rejected_under_experimental_selector_and_experimental_installs_isolated(self):
        root = scratch('ds41f-126r-install-')
        exp_profile, stock_profile = profile(EXPERIMENTAL), profile(STOCK)
        stock_built = root / 'stock-built'
        stock_built.mkdir()
        synthetic_receipt(stock_built, 'v126')
        with self.assertRaises(ValueError):
            build.install(stock_built, root, EXPERIMENTAL)
        self.assertFalse((root / exp_profile['binary_subdir']).exists())
        exp_built = root / 'exp-built'
        exp_built.mkdir()
        synthetic_receipt(exp_built, 'v126-resumable')
        result = build.install(exp_built, root, EXPERIMENTAL)
        self.assertEqual(result['recipe'], 'v126-resumable')
        self.assertEqual(result['services_touched'], False)
        dest = Path(result['installed'])
        self.assertEqual(dest, root / exp_profile['binary_subdir'])
        for name in ['tensorfold-native', 'tf-dsv41-lanes']:
            self.assertEqual((dest / name).read_bytes(), b'test-only-never-executable')
        self.assertTrue((dest / 'BUILD.json').is_file())
        self.assertFalse((root / stock_profile['binary_subdir']).exists())

    def test_graph_owner_receipt_installs_isolated_and_preserves_resumable_install(self):
        root = scratch('ds41f-126r-owner-install-')
        selector = '1.2.6-resumable-graph-owner'
        old_built = root / 'old-built'
        old_built.mkdir()
        synthetic_receipt(old_built, 'v126-resumable')
        old_result = build.install(old_built, root, EXPERIMENTAL)
        old_dest = Path(old_result['installed'])
        old_files = {p.name: p.read_bytes() for p in old_dest.iterdir()}
        new_dest = root / profile(selector)['binary_subdir']
        with self.assertRaises(ValueError):
            build.install(old_built, root, selector)
        self.assertFalse(new_dest.exists())
        new_built = root / 'new-built'
        new_built.mkdir()
        synthetic_receipt(new_built, 'v126-resumable-graph-owner')
        result = build.install(new_built, root, selector)
        self.assertEqual(Path(result['installed']), new_dest)
        self.assertEqual(result['services_touched'], False)
        self.assertEqual(result['recipe'], 'v126-resumable-graph-owner')
        for name in ['tensorfold-native', 'tf-dsv41-lanes', 'BUILD.json']:
            self.assertEqual((new_dest / name).read_bytes(), (new_built / name).read_bytes())
        self.assertEqual({p.name: p.read_bytes() for p in old_dest.iterdir()}, old_files)
        self.assertEqual({p.name for p in root.iterdir() if p.name.startswith('bin-')},
                         {old_dest.name, new_dest.name})


if __name__ == '__main__':
    unittest.main()
