"""
Embedding model wrappers.

This module abstracts away the details of loading and running different
facial embedding models.  By default it uses InsightFace (iresnet100 trained
on glint360k) but optionally supports AdaFace if installed.  The
:class:`Embedder` interface exposes a single method :meth:`extract_faces`
which takes a BGR image and returns a list of face records with bounding
boxes, detection scores, pose angles and embeddings.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Dict, Any, Optional


class Embedder:
    """Base class for all embedders."""

    def extract_faces(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and embed faces from a BGR image.

        Subclasses must implement this method and return a list of dicts
        containing at least ``embedding``, ``bbox`` and ``det_score``.
        """
        raise NotImplementedError


class InsightFaceEmbedder(Embedder):
    """Wrapper around InsightFace ``FaceAnalysis`` API.

    Parameters
    ----------
    model_name: str
        Name of the model package to load from InsightFace.  ``"iresnet100"``
        corresponds to ``buffalo_l`` (glint360k).  Other names supported by
        InsightFace can be specified here.
    min_face_size: int
        Minimum side length (in pixels) of detected faces.  Smaller faces
        are filtered out.
    use_gpu: bool
        Whether to use CUDA if available; falls back to CPU otherwise.
    """
    def __init__(self, model_name: str = "iresnet100", min_face_size: int = 80, use_gpu: bool = True) -> None:
        from insightface.app import FaceAnalysis
        import sys
        # Map our model name to InsightFace's model packages
        name_map = {
            "iresnet100": "buffalo_l",
        }
        pkg = name_map.get(model_name, model_name)
        # Prepare device list.  If CUDA is available and requested, use it.
        providers = []
        if use_gpu:
            try:
                import onnxruntime  # noqa: F401
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except Exception:
                providers = ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # Construct FaceAnalysis and prepare; fall back if some InsightFace packages
        # (e.g. antelopev2 on certain installs) fail to load bundled detection.
        try:
            self.app = FaceAnalysis(name=pkg, providers=providers)
            self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
        except AssertionError as e:
            # Known issue: some InsightFace packages may not include detection models
            # depending on version/provider availability. Fallback to buffalo_l which
            # bundles a compatible detector+recognizer so the pipeline can proceed.
            if str(pkg).lower() == "antelopev2":
                print(
                    "Warning: InsightFace package 'antelopev2' failed to load detection. "
                    "Falling back to 'buffalo_l' (iResNet100 glint360k).\n"
                    "To use antelopev2, ensure your InsightFace version is >= 0.7 and "
                    "that onnxruntime (GPU if available) is correctly installed.",
                    file=sys.stderr,
                )
                self.app = FaceAnalysis(name="buffalo_l", providers=providers)
                self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
            else:
                raise
        self.min_face_size = min_face_size

    def extract_faces(self, img: np.ndarray) -> List[Dict[str, Any]]:
        faces = self.app.get(img)
        results: List[Dict[str, Any]] = []
        for f in faces:
            bbox = f.bbox.astype(float)
            x1, y1, x2, y2 = bbox
            # Filter by face size
            if min(x2 - x1, y2 - y1) < self.min_face_size:
                continue
            embedding = f.embedding.astype(np.float32)
            # L2 normalise
            embedding /= np.linalg.norm(embedding) + 1e-9
            det_score = float(f.det_score) if hasattr(f, "det_score") else None
            pose = {
                "yaw": float(getattr(f.pose, "yaw", 0.0)),
                "pitch": float(getattr(f.pose, "pitch", 0.0)),
                "roll": float(getattr(f.pose, "roll", 0.0)),
            }
            results.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "embedding": embedding,
                "det_score": det_score,
                "pose": pose,
            })
        return results


class AdaFaceEmbedder(Embedder):
    """Wrapper around AdaFace model.

    This wrapper loads a PyTorch AdaFace model and performs face detection
    using InsightFace SCRFD.  AdaFace models focus on embedding quality by
    dynamically adjusting margins based on facial quality.  See
    https://github.com/mk-minchul/adaface for more details.
    """
    def __init__(self, weights_path: Optional[str] = None, min_face_size: int = 80, use_gpu: bool = True) -> None:
        # Detector from InsightFace
        from insightface.app import FaceAnalysis
        providers = []
        if use_gpu:
            try:
                import onnxruntime
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except Exception:
                providers = ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.detector = FaceAnalysis(name="scrfd_10g", providers=providers)
        self.detector.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
        self.min_face_size = min_face_size
        # Load AdaFace model
        try:
            import torch
            from adaface_pytorch.model import build_model
        except ImportError as e:
            raise RuntimeError("AdaFace is not installed.  Install optional dependency adaface-pytorch.") from e
        # weights_path can be None to use default pretrain.  Use build_model helper.
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.model = build_model(weights_path=weights_path)
        self.model.eval().to(self.device)

    def extract_faces(self, img: np.ndarray) -> List[Dict[str, Any]]:
        # First detect faces using SCRFD
        dets = self.detector.get(img)
        results: List[Dict[str, Any]] = []
        import torch
        for f in dets:
            x1, y1, x2, y2 = f.bbox.astype(float)
            if min(x2 - x1, y2 - y1) < self.min_face_size:
                continue
            # Align face using InsightFace's aligner (5 point landmarks)
            kps = f.kps
            # Compute similarity transform to 112x112
            M = self.detector.app.ins_get_transform(kps)
            aligned = cv2.warpAffine(img, M, (112, 112))
            aligned = aligned[:, :, ::-1]  # BGR->RGB
            # Preprocess for AdaFace: scale to [0,1], normalise
            face = aligned.astype(np.float32) / 255.0
            face = (face - 0.5) / 0.5
            face = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(face)
            embedding = feat.cpu().numpy().squeeze().astype(np.float32)
            embedding /= np.linalg.norm(embedding) + 1e-9
            pose = {
                "yaw": float(getattr(f.pose, "yaw", 0.0)),
                "pitch": float(getattr(f.pose, "pitch", 0.0)),
                "roll": float(getattr(f.pose, "roll", 0.0)),
            }
            results.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "embedding": embedding,
                "det_score": float(f.det_score) if hasattr(f, "det_score") else None,
                "pose": pose,
            })
        return results


def get_embedder(model_name: str, adaface_weights: Optional[str] = None,
                 min_face_size: int = 80, use_gpu: bool = True) -> Embedder:
    """Factory function returning an embedder instance given a model name."""
    model_name = model_name.lower()
    if model_name == "iresnet100":
        return InsightFaceEmbedder(model_name="iresnet100", min_face_size=min_face_size, use_gpu=use_gpu)
    elif model_name == "adaface":
        return AdaFaceEmbedder(weights_path=adaface_weights, min_face_size=min_face_size, use_gpu=use_gpu)
    else:
        # Attempt to load arbitrary InsightFace model names directly
        return InsightFaceEmbedder(model_name=model_name, min_face_size=min_face_size, use_gpu=use_gpu)
