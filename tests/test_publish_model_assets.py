import importlib.util
from pathlib import Path
import sys
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


if __name__ == '__main__':
    unittest.main()
