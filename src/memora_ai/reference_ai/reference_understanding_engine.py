from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(slots=True)
class SceneAnalysis:
    """Detailed analysis for a single scene in the reference reel."""

    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    frame_count: int
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    rotation: float = 0.0
    collage_layout: str = "single"
    photo_positions: list[dict[str, float]] = field(default_factory=list)
    text_positions: list[dict[str, float]] = field(default_factory=list)
    animation_duration: float = 0.0
    motion_timing: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class TransitionAnalysis:
    """Detailed analysis for a transition between scenes."""

    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    transition_type: str
    intensity: float


@dataclass(slots=True)
class ReferenceStyle:
    """Complete structural description of a reference reel."""

    scenes: list[SceneAnalysis] = field(default_factory=list)
    transitions: list[TransitionAnalysis] = field(default_factory=list)
    total_frames: int = 0
    fps: float = 0.0
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class ReferenceUnderstandingEngine:
    """Analyze a reference reel and build a rich structural description of its style."""

    def __init__(self) -> None:
        self._scene_threshold = 0.18
        self._transition_threshold = 0.14

    def analyze(self, video_path: str) -> ReferenceStyle:
        """Analyze a reference reel and return a comprehensive style description."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

            frames: list[np.ndarray] = []
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frames.append(frame)

            return self._build_reference_style(frames, fps, total_frames, width, height)
        finally:
            capture.release()

    def _build_reference_style(
        self,
        frames: list[np.ndarray],
        fps: float,
        total_frames: int,
        width: int,
        height: int,
    ) -> ReferenceStyle:
        if not frames:
            return ReferenceStyle(total_frames=total_frames, fps=fps, width=width, height=height)

        scene_boundaries = self._detect_scene_boundaries(frames)
        scenes = self._build_scenes(frames, scene_boundaries, fps)
        transitions = self._build_transitions(frames, scene_boundaries, fps)

        return ReferenceStyle(
            scenes=scenes,
            transitions=transitions,
            total_frames=total_frames,
            fps=fps,
            duration_seconds=total_frames / fps if fps > 0 else 0.0,
            width=width,
            height=height,
            metadata={
                "scene_count": len(scenes),
                "transition_count": len(transitions),
                "analysis_version": "1.0",
            },
        )

    def _detect_scene_boundaries(self, frames: list[np.ndarray]) -> list[int]:
        boundaries: list[int] = [0]
        prev_frame = self._prepare_frame(frames[0])

        for index in range(1, len(frames)):
            current_frame = self._prepare_frame(frames[index])
            score = self._frame_change_score(prev_frame, current_frame)
            if score >= self._scene_threshold:
                boundaries.append(index)
            prev_frame = current_frame

        if boundaries[-1] != len(frames) - 1:
            boundaries.append(len(frames) - 1)
        return boundaries

    def _build_scenes(
        self,
        frames: list[np.ndarray],
        boundaries: list[int],
        fps: float,
    ) -> list[SceneAnalysis]:
        scenes: list[SceneAnalysis] = []
        for idx, start_frame in enumerate(boundaries[:-1]):
            end_frame = boundaries[idx + 1]
            if end_frame <= start_frame:
                continue
            scene_frames = frames[start_frame:end_frame + 1]
            scene = self._analyze_scene(scene_frames, start_frame, end_frame, fps)
            scenes.append(scene)
        return scenes

    def _build_transitions(
        self,
        frames: list[np.ndarray],
        boundaries: list[int],
        fps: float,
    ) -> list[TransitionAnalysis]:
        transitions: list[TransitionAnalysis] = []
        for idx in range(len(boundaries) - 1):
            start_frame = boundaries[idx]
            end_frame = boundaries[idx + 1]
            if end_frame <= start_frame:
                continue
            transition_frames = frames[start_frame:end_frame + 1]
            transition_type = self._classify_transition(transition_frames)
            if transition_type == "none":
                continue
            start_time = start_frame / fps if fps > 0 else 0.0
            end_time = end_frame / fps if fps > 0 else 0.0
            transitions.append(
                TransitionAnalysis(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=round(start_time, 4),
                    end_time=round(end_time, 4),
                    duration=round(max(0.0, end_time - start_time), 4),
                    transition_type=transition_type,
                    intensity=self._estimate_transition_intensity(transition_frames),
                )
            )
        return transitions

    def _analyze_scene(
        self,
        frames: list[np.ndarray],
        start_frame: int,
        end_frame: int,
        fps: float,
    ) -> SceneAnalysis:
        frame0 = self._prepare_frame(frames[0])
        frame1 = self._prepare_frame(frames[-1])

        zoom = self._estimate_zoom(frame0, frame1)
        pan_x, pan_y = self._estimate_pan(frame0, frame1)
        rotation = self._estimate_rotation(frame0, frame1)
        collage_layout = self._estimate_collage_layout(frames)
        photo_positions = self._estimate_photo_positions(frames)
        text_positions = self._estimate_text_positions(frames)
        animation_duration = max(0.0, (end_frame - start_frame) / fps if fps > 0 else 0.0)

        return SceneAnalysis(
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=round(start_frame / fps if fps > 0 else 0.0, 4),
            end_time=round(end_frame / fps if fps > 0 else 0.0, 4),
            duration=round(max(0.0, (end_frame - start_frame) / fps) if fps > 0 else 0.0, 4),
            frame_count=max(1, end_frame - start_frame + 1),
            zoom=round(zoom, 4),
            pan_x=round(pan_x, 4),
            pan_y=round(pan_y, 4),
            rotation=round(rotation, 4),
            collage_layout=collage_layout,
            photo_positions=photo_positions,
            text_positions=text_positions,
            animation_duration=round(animation_duration, 4),
            motion_timing={
                "zoom": round(zoom, 4),
                "pan_x": round(pan_x, 4),
                "pan_y": round(pan_y, 4),
                "rotation": round(rotation, 4),
            },
        )

    @staticmethod
    def _prepare_frame(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(resized, (5, 5), 0)

    @staticmethod
    def _frame_change_score(prev_frame: np.ndarray, current_frame: np.ndarray) -> float:
        hist_change = ReferenceUnderstandingEngine._histogram_difference(prev_frame, current_frame)
        diff = cv2.absdiff(prev_frame, current_frame)
        mean_diff = float(np.mean(diff) / 255.0)
        return float(np.clip(0.6 * hist_change + 0.4 * mean_diff, 0.0, 1.0))

    @staticmethod
    def _histogram_difference(prev_frame: np.ndarray, current_frame: np.ndarray) -> float:
        hist_prev = cv2.calcHist([prev_frame], [0], None, [64], [0, 256])
        hist_curr = cv2.calcHist([current_frame], [0], None, [64], [0, 256])
        cv2.normalize(hist_prev, hist_prev)
        cv2.normalize(hist_curr, hist_curr)
        return float(cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_BHATTACHARYYA))

    @staticmethod
    def _estimate_zoom(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        diff = float(np.mean(cv2.absdiff(frame_a, frame_b)))
        return float(np.clip(1.0 + diff / 1000.0, 0.8, 1.6))

    @staticmethod
    def _estimate_pan(frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[float, float]:
        return (0.0, 0.0)

    @staticmethod
    def _estimate_rotation(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        return 0.0

    @staticmethod
    def _estimate_collage_layout(self, frame: np.ndarray) -> str:
    """
    Detect whether the current frame contains
    a single photo or multiple photos.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 60, 180)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    rectangles = 0

    h, w = gray.shape

    min_area = h * w * 0.02

    for c in contours:

        area = cv2.contourArea(c)

        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(c)

        ratio = bw / max(bh, 1)

        if 0.4 < ratio < 2.5:
            rectangles += 1

    if rectangles >= 4:
        return "grid"

    if rectangles >= 2:
        return "multiple"

    return "single"

    @staticmethod
    def _estimate_photo_positions(frames: list[np.ndarray]) -> list[dict[str, float]]:
        return []

    @staticmethod
    def _estimate_text_positions(frames: list[np.ndarray]) -> list[dict[str, float]]:
        return []

    def _classify_transition(self, frames: list[np.ndarray]) -> str:
        if len(frames) < 2:
            return "none"

        scores = [self._frame_change_score(self._prepare_frame(frames[idx]), self._prepare_frame(frames[idx + 1])) for idx in range(len(frames) - 1)]
        mean_score = float(np.mean(scores)) if scores else 0.0
        if mean_score >= self._transition_threshold:
            return "fade"
        return "none"

    @staticmethod
    def _estimate_transition_intensity(frames: list[np.ndarray]) -> float:
        if len(frames) < 2:
            return 0.0
        return float(np.clip(len(frames) / 120.0, 0.0, 1.0))
