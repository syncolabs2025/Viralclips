"""
Cross-shot identity clustering via ArcFace cosine similarity.

All face embeddings extracted across every shot are clustered with DBSCAN
(cosine distance metric).  Faces above the similarity threshold are assigned
the same stable person_N ID, regardless of which shot they appear in.

identify_lead_performer() then finds the cluster with the highest combined
score of total screen time × centre-bias — the person who is most often in
the centre of frame for the longest is the lead.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("reframe.identity")


@dataclass
class PersonInfo:
    person_id:        str
    total_screen_time: float
    avg_center_dist:  float
    shot_indices:     list[int] = field(default_factory=list)
    role:             str = "unknown"      # "lead" | "secondary" | "background"


def cluster_identities(
    shot_embeddings: list[list[np.ndarray]],
    shot_durations:  list[float],
    threshold:       float = 0.40,
) -> dict[int, list[str]]:
    """
    Cluster face embeddings across all shots into stable person IDs.

    Args:
        shot_embeddings: outer list = shots; inner list = per-face embeddings.
        shot_durations:  duration in seconds for each shot.
        threshold:       ArcFace cosine similarity above which two faces are
                         the same person (0.40 is the standard ArcFace threshold).

    Returns:
        {shot_idx: [person_id_for_face_0, person_id_for_face_1, ...]}
    """
    all_embs: list[np.ndarray] = []
    meta: list[tuple[int, int]] = []    # (shot_idx, face_idx)

    for shot_idx, embs in enumerate(shot_embeddings):
        for face_idx, emb in enumerate(embs):
            if emb is not None and emb.size > 0:
                normed = emb / (np.linalg.norm(emb) + 1e-8)
                all_embs.append(normed)
                meta.append((shot_idx, face_idx))

    if not all_embs:
        return {}

    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        log.warning("scikit-learn not available — assigning unique person IDs per detection")
        return {
            shot_idx: [f"person_{shot_idx}_{fi}" for fi in range(len(shot_embeddings[shot_idx]))]
            for shot_idx in range(len(shot_embeddings))
        }

    matrix = np.stack(all_embs)
    eps     = 1.0 - threshold    # cosine distance equivalent
    labels  = DBSCAN(eps=eps, min_samples=1, metric="cosine", n_jobs=-1).fit_predict(matrix)

    unique   = sorted(set(labels))
    id_map   = {lbl: f"person_{i}" for i, lbl in enumerate(unique)}

    result: dict[int, list[str]] = {}
    for (shot_idx, face_idx), label in zip(meta, labels):
        bucket = result.setdefault(shot_idx, [])
        while len(bucket) <= face_idx:
            bucket.append("unknown")
        bucket[face_idx] = id_map[label]

    n_persons = len(unique)
    log.info("Identity clustering: %d unique persons across %d shots", n_persons, len(shot_embeddings))
    return result


def identify_lead_performer(
    shot_to_persons: dict[int, list[str]],
    shot_analyses:   list,          # list[ShotAnalysisResult]
    shot_durations:  list[float],
) -> Optional[str]:
    """
    Return the person_id of the lead performer.

    Scoring:
        lead_score = screen_time_seconds × (1 − avg_normalised_centre_distance)

    The person who is most present AND most centred wins.
    """
    screen_time:   dict[str, float]        = {}
    center_scores: dict[str, list[float]]  = {}

    for shot_idx, person_ids in shot_to_persons.items():
        duration = shot_durations[shot_idx] if shot_idx < len(shot_durations) else 1.0
        analysis = shot_analyses[shot_idx]  if shot_idx < len(shot_analyses)  else None

        for face_idx, pid in enumerate(person_ids):
            screen_time[pid] = screen_time.get(pid, 0.0) + duration
            if analysis and face_idx < len(analysis.faces):
                center_scores.setdefault(pid, []).append(analysis.faces[face_idx].center_dist)

    if not screen_time:
        return None

    scores: dict[str, float] = {
        pid: t * (1.0 - float(np.mean(center_scores.get(pid, [0.5]))))
        for pid, t in screen_time.items()
    }

    lead = max(scores, key=scores.__getitem__)
    log.info("Lead: %s  (score=%.2f  screen_time=%.1fs)", lead, scores[lead], screen_time[lead])
    return lead
