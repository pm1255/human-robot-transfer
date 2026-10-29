import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('model_assets', Path(__file__).parents[1] / 'tools/model_assets.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ModelAssetTests(unittest.TestCase):
    def test_integrity_distinguishes_missing_size_and_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = dict(local_path='weights.bin', bytes=3,
                         sha256='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
                         delivery='mirror')
            self.assertEqual(module.check_asset(root, asset), 'missing')
            (root / 'weights.bin').write_bytes(b'a')
            self.assertEqual(module.check_asset(root, asset), 'size_mismatch')
            (root / 'weights.bin').write_bytes(b'abd')
            self.assertEqual(module.check_asset(root, asset), 'hash_mismatch')
            (root / 'weights.bin').write_bytes(b'abc')
            self.assertEqual(module.check_asset(root, asset), 'verified')

    def test_manual_assets_are_explicit_and_unknown_component_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(module.check_asset(Path(directory), dict(local_path='MANO_RIGHT.pkl', delivery='manual')),
                             'manual_download_required')
        manifest = dict(assets=[dict(component='hamer')])
        with self.assertRaises(ValueError):
            module.selected_assets(manifest, ['typo'])


if __name__ == '__main__':
    unittest.main()
