import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "hunyuan3d"))

from PIL import Image
from pygltflib import GLTF2

from app import IMAGE_TEXTURE_MODE, TextureService, apply_cpu_image_texture_to_glb


class CpuTextureTests(unittest.TestCase):
    def test_cpu_glb_texture_preserves_skin_and_embeds_image(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "app" / "examples" / "geno.glb"

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "texture.png"
            output_path = Path(tmp) / "textured.glb"
            Image.new("RGBA", (8, 8), (220, 60, 40, 255)).save(image_path)

            apply_cpu_image_texture_to_glb(source, image_path, output_path)

            original = GLTF2().load_binary(str(source))
            textured = GLTF2().load_binary(str(output_path))

        self.assertEqual(len(textured.skins or []), len(original.skins or []))
        self.assertTrue(textured.images)
        self.assertTrue(textured.textures)
        self.assertTrue(textured.materials)
        primitive = textured.meshes[0].primitives[0]
        self.assertIsNotNone(primitive.material)
        material = textured.materials[primitive.material]
        self.assertIsNotNone(material.pbrMetallicRoughness.baseColorTexture)

    def test_texture_service_defaults_to_cpu_texture_without_cuda(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "app" / "examples" / "geno.glb"

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "texture.png"
            Image.new("RGBA", (8, 8), (40, 80, 220, 255)).save(image_path)

            result = TextureService(device="cpu").texture(
                source,
                image_path,
                preserve_rig=True,
                texture_mode=IMAGE_TEXTURE_MODE,
            )
            textured = GLTF2().load_binary(str(result.output_path))

        self.assertIn("CPU", result.status)
        self.assertTrue(textured.skins)
