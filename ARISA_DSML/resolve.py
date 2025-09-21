import numpy as np
from mtcnn.mtcnn import MTCNN


def convert_results(results):
    if isinstance(results, dict):
        return {k: convert_results(v) for k, v in results.items()}
    elif isinstance(results, list):
        return [convert_results(v) for v in results]
    elif isinstance(results, np.integer):
        return int(results)
    else:
        return results


def detect_faces(image):
    detector = MTCNN()
    # detect faces in the image
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.shape[-1] != 3:
        image = np.stack([image.squeeze()] * 3, axis=-1)
    results = detector.detect_faces(image)
    json_results = convert_results(results)
    return json_results
