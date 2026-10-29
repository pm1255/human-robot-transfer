import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('publish_model_assets', TOOLS / 'publish_model_assets.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class PublishAssetTests(unittest.TestCase):
    def test_remote_size_alone_is_not_integrity(self):
        asset = dict(bytes=4, sha256='expected')
        self.assertFalse(publisher.remote_matches(asset, None))
        self.assertFalse(publisher.remote_matches(asset, SimpleNamespace(size=4, lfs=None)))
        self.assertFalse(publisher.remote_matches(asset, SimpleNamespace(size=4, lfs=dict(sha256='wrong'))))
        self.assertTrue(publisher.remote_matches(asset, SimpleNamespace(size=4, lfs=dict(sha256='expected'))))
        self.assertTrue(publisher.remote_matches(asset, SimpleNamespace(size=4, lfs=SimpleNamespace(sha256='expected'))))

    def test_shared_sam_root_does_not_redirect_other_models(self):
        sam = dict(component='sam31', local_path='models/sam3.1/model.pt')
        other = dict(component='hamer', local_path='models/hamer/model.ckpt')
        self.assertEqual(publisher.source_path(sam, Path('/project'), Path('/shared')), Path('/shared/model.pt'))
        self.assertEqual(publisher.source_path(other, Path('/project'), Path('/shared')), Path('/project/models/hamer/model.ckpt'))

    def test_no_credentials_stops_before_api_creation(self):
        with patch.object(sys, 'argv', ['publish_model_assets.py', '--publish']), \
             patch('huggingface_hub.get_token', return_value=None), \
             patch('huggingface_hub.HfApi') as api:
            with self.assertRaisesRegex(RuntimeError, 'login is required'):
                publisher.main()
            api.assert_not_called()

    def test_verified_remote_asset_needs_no_local_copy_or_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / 'model_assets'
            model_dir.mkdir()
            asset = dict(component='hamer', local_path='models/missing.ckpt',
                         hub_path='hamer/model.ckpt', bytes=4, sha256='expected', delivery='mirror')
            (model_dir / 'manifest.json').write_text(json.dumps(dict(assets=[asset])))
            (model_dir / 'MODEL_CARD.md').write_text('# Model card\n')
            remote = SimpleNamespace(rfilename=asset['hub_path'], size=4,
                                     lfs=SimpleNamespace(sha256='expected'))
            info = SimpleNamespace(siblings=[remote], sha='before', private=True)
            with patch.object(sys, 'argv', ['publish_model_assets.py', '--publish', '--root', str(root)]), \
                 patch.object(publisher, 'REPO', root), \
                 patch('huggingface_hub.get_token', return_value='test-fixture'), \
                 patch('huggingface_hub.HfApi') as factory, \
                 patch.object(publisher, 'source_path') as source, \
                 contextlib.redirect_stdout(io.StringIO()):
                api = factory.return_value
                api.model_info.return_value = info
                api.create_commit.return_value = SimpleNamespace(oid='after')
                publisher.main()
                source.assert_not_called()
                uploaded = [operation.path_in_repo for operation in api.create_commit.call_args.kwargs['operations']]
                self.assertNotIn(asset['hub_path'], uploaded)
                self.assertIn('README.md', uploaded)
                self.assertIn('manifest.json', uploaded)
                api.update_repo_settings.assert_not_called()


if __name__ == '__main__':
    unittest.main()
