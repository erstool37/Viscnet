# ---------- entropy helpers ----------
def _frame_entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def video_entropy_array(video_path: str) -> np.ndarray:
    """
    Compute per-frame entropy (bits) for a 50-frame grayscale video.
    Returns np.ndarray of shape (50,).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    vals = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vals.append(_frame_entropy(gray))
    cap.release()

    if len(vals) != 50:
        raise ValueError(f"Expected 50 frames, got {len(vals)} in {video_path}")
    return np.array(vals, dtype=np.float32)
