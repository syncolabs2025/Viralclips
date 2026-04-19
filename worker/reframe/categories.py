from dataclasses import dataclass
from enum import Enum


class ContentCategory(str, Enum):
    SOLO_VLOG      = "solo_vlog"
    PODCAST        = "podcast"
    INTERVIEW      = "interview"
    MUSIC_VIDEO    = "music_video"
    GAMING         = "gaming"
    TRAVEL_SCENERY = "travel_scenery"
    COMEDY_SKIT    = "comedy_skit"
    FITNESS        = "fitness"
    SPORTS_ACTION  = "sports_action"
    COOKING_FOOD   = "cooking_food"
    PRODUCT_REVIEW = "product_review"
    KEYNOTE_TALK   = "keynote_talk"
    REACTION_VIDEO = "reaction_video"
    ASMR_LOFI      = "asmr_lofi"
    UNKNOWN        = "unknown"


@dataclass
class ReframeRule:
    subject_type:       str    # "face" | "body" | "object" | "scene" | "hands" | "screen"
    detection_fps:      float  # how often to run subject detection
    smoothing:          str    # "kalman" | "ema" | "beat_sync"
    ema_alpha:          float  # for ema smoothing (ignored when smoothing != "ema")
    kalman_process_var: float  # lower = smoother camera
    kalman_measure_var: float  # higher = trust measurements less
    headroom_ratio:     float  # shift crop up so subject sits below this fraction of frame
    active_slot_pct:    float  # fraction of 9:16 height given to the primary slot
    beat_sync_cuts:     bool   # snap shot boundaries to beat grid (music video)
    zoom_on_emphasis:   bool   # punch-in when voice energy spikes
    full_body:          bool   # expand bbox to include torso (fitness / music)
    gdino_prompt:       str    # GroundingDINO text prompt for non-face subjects


CATEGORY_RULES: dict[ContentCategory, ReframeRule] = {
    ContentCategory.SOLO_VLOG: ReframeRule(
        subject_type="face",      detection_fps=5.0,  smoothing="kalman",
        ema_alpha=0.15, kalman_process_var=5.0,  kalman_measure_var=200.0,
        headroom_ratio=0.20, active_slot_pct=1.0,
        beat_sync_cuts=False, zoom_on_emphasis=False, full_body=False,
        gdino_prompt="person . face",
    ),
    ContentCategory.PODCAST: ReframeRule(
        subject_type="face",      detection_fps=5.0,  smoothing="kalman",
        ema_alpha=0.10, kalman_process_var=3.0,  kalman_measure_var=300.0,
        headroom_ratio=0.15, active_slot_pct=0.60,
        beat_sync_cuts=False, zoom_on_emphasis=False, full_body=False,
        gdino_prompt="person . face . microphone",
    ),
    ContentCategory.INTERVIEW: ReframeRule(
        subject_type="face",      detection_fps=5.0,  smoothing="kalman",
        ema_alpha=0.10, kalman_process_var=3.0,  kalman_measure_var=300.0,
        headroom_ratio=0.15, active_slot_pct=0.65,
        beat_sync_cuts=False, zoom_on_emphasis=False, full_body=False,
        gdino_prompt="person . face . microphone . desk",
    ),
    ContentCategory.MUSIC_VIDEO: ReframeRule(
        subject_type="body",      detection_fps=10.0, smoothing="beat_sync",
        ema_alpha=0.20, kalman_process_var=15.0, kalman_measure_var=80.0,
        headroom_ratio=0.06, active_slot_pct=1.0,
        beat_sync_cuts=True,  zoom_on_emphasis=False, full_body=True,
        gdino_prompt="performer . singer . dancer . artist . musician",
    ),
    ContentCategory.GAMING: ReframeRule(
        subject_type="screen",    detection_fps=2.0,  smoothing="ema",
        ema_alpha=0.15, kalman_process_var=20.0, kalman_measure_var=50.0,
        headroom_ratio=0.0, active_slot_pct=0.65,
        beat_sync_cuts=False, zoom_on_emphasis=False, full_body=False,
        gdino_prompt="game screen . player character . health bar . minimap . UI",
    ),
    ContentCategory.TRAVEL_SCENERY: ReframeRule(
        subject_type="scene",     detection_fps=1.0,  smoothing="ema",
        ema_alpha=0.05, kalman_process_var=2.0,  kalman_measure_var=500.0,
        headroom_ratio=0.0, active_slot_pct=1.0,
        beat_sync_cuts=False, zoom_on_emphasis=False, full_body=False,
        gdino_prompt="landmark . mountain . building . person . horizon . architecture",
    ),
    ContentCategory.COMEDY_SKIT: ReframeRule(
        subject_type="face",      detection_fps=8.0,  smoothing="kalman",
        ema_alpha=0.15, kalman_process_var=8.0,  kalman_measure_var=150.0,
        headroom_ratio=0.18, active_slot_pct=1.0,
        beat_sync_cuts=False, zoom_on_emphasis=False, full_body=False,
        gdino_prompt="person . face",
    ),
    ContentCategory.FITNESS: ReframeRule(
        subject_type="body",      detection_fps=8.0,  smoothing="kalman",
        ema_alpha=0.15, kalman_process_var=10.0, kalman_measure_var=100.0,
        headroom_ratio=0.05, active_slot_pct=1.0,
        beat_sync_cuts=False, zoom_on_emphasis=False, full_body=True,
        gdino_prompt="person . athlete . exercise equipment . gym",
    ),
    ContentCategory.SPORTS_ACTION: ReframeRule(
        subject_type="object",    detection_fps=15.0, smoothing="kalman",
        ema_alpha=0.25, kalman_process_var=25.0, kalman_measure_var=80.0,
        headroom_ratio=0.0, active_slot_pct=1.0,
        beat_sync_cuts=False, zoom_on_emphasis=False, full_body=True,
        gdino_prompt="ball . player . athlete . goal . basket . court . puck",
    ),
    ContentCategory.COOKING_FOOD: ReframeRule(
        subject_type="hands",     detection_fps=4.0,  smoothing="ema",
        ema_alpha=0.20, kalman_process_var=8.0,  kalman_measure_var=150.0,
        headroom_ratio=0.0, active_slot_pct=1.0,
        beat_sync_cuts=False, zoom_on_emphasis=False, full_body=False,
        gdino_prompt="hand . knife . bowl . food . cutting board . pan . ingredient . plate",
    ),
    ContentCategory.PRODUCT_REVIEW: ReframeRule(
        subject_type="face",      detection_fps=5.0,  smoothing="kalman",
        ema_alpha=0.15, kalman_process_var=5.0,  kalman_measure_var=200.0,
        headroom_ratio=0.15, active_slot_pct=0.60,
        beat_sync_cuts=False, zoom_on_emphasis=False, full_body=False,
        gdino_prompt="product . package . device . bottle . box . hand . gadget",
    ),
    ContentCategory.KEYNOTE_TALK: ReframeRule(
        subject_type="face",      detection_fps=3.0,  smoothing="kalman",
        ema_alpha=0.10, kalman_process_var=3.0,  kalman_measure_var=400.0,
        headroom_ratio=0.15, active_slot_pct=0.60,
        beat_sync_cuts=False, zoom_on_emphasis=False, full_body=False,
        gdino_prompt="slide . screen . presenter . podium . stage . monitor",
    ),
    ContentCategory.REACTION_VIDEO: ReframeRule(
        subject_type="face",      detection_fps=5.0,  smoothing="kalman",
        ema_alpha=0.15, kalman_process_var=5.0,  kalman_measure_var=200.0,
        headroom_ratio=0.15, active_slot_pct=0.50,
        beat_sync_cuts=False, zoom_on_emphasis=False, full_body=False,
        gdino_prompt="person . face . phone screen . tablet . monitor . video",
    ),
    ContentCategory.ASMR_LOFI: ReframeRule(
        subject_type="object",    detection_fps=1.0,  smoothing="ema",
        ema_alpha=0.03, kalman_process_var=1.0,  kalman_measure_var=1000.0,
        headroom_ratio=0.0, active_slot_pct=1.0,
        beat_sync_cuts=False, zoom_on_emphasis=False, full_body=False,
        gdino_prompt="hand . object . texture . surface . book . pen",
    ),
    ContentCategory.UNKNOWN: ReframeRule(
        subject_type="face",      detection_fps=5.0,  smoothing="kalman",
        ema_alpha=0.15, kalman_process_var=5.0,  kalman_measure_var=200.0,
        headroom_ratio=0.20, active_slot_pct=1.0,
        beat_sync_cuts=False, zoom_on_emphasis=False, full_body=False,
        gdino_prompt="person . face",
    ),
}
