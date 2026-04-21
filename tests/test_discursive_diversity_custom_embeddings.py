from pathlib import Path
import importlib.util
import sys
import types

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DISC_DIVERSITY_PATH = REPO_ROOT / "src" / "team_comm_tools" / "features" / "discursive_diversity.py"
WITHIN_PERSON_PATH = REPO_ROOT / "src" / "team_comm_tools" / "features" / "within_person_discursive_range.py"


def load_discursive_diversity_modules(monkeypatch):
    fake_package = types.ModuleType("team_comm_tools")
    fake_package.__path__ = []
    fake_features_package = types.ModuleType("team_comm_tools.features")
    fake_features_package.__path__ = []

    monkeypatch.setitem(sys.modules, "team_comm_tools", fake_package)
    monkeypatch.setitem(sys.modules, "team_comm_tools.features", fake_features_package)

    disc_spec = importlib.util.spec_from_file_location(
        "team_comm_tools.features.discursive_diversity",
        DISC_DIVERSITY_PATH,
    )
    disc_module = importlib.util.module_from_spec(disc_spec)
    sys.modules[disc_spec.name] = disc_module
    disc_spec.loader.exec_module(disc_module)

    within_spec = importlib.util.spec_from_file_location(
        "team_comm_tools.features.within_person_discursive_range",
        WITHIN_PERSON_PATH,
    )
    within_module = importlib.util.module_from_spec(within_spec)
    sys.modules[within_spec.name] = within_module
    within_spec.loader.exec_module(within_module)

    return disc_module, within_module


def test_within_person_disc_range_handles_custom_embedding_dimensions(monkeypatch):
    _, within_module = load_discursive_diversity_modules(monkeypatch)

    vec_a = np.array([1.0] * 1536)
    vec_b = np.array([0.5] * 1536)

    chat_data = pd.DataFrame(
        {
            "conversation_num": ["conv1", "conv1", "conv1"],
            "speaker_nickname": ["alice", "alice", "bob"],
            "chunk_num": [0, 1, 0],
            "message_embedding": [vec_a, vec_b, vec_a],
        }
    )

    result = within_module.get_within_person_disc_range(
        chat_data,
        num_chunks=2,
        conversation_id_col="conversation_num",
        speaker_id_col="speaker_nickname",
    )

    assert list(result.index) == ["conv1"]
    assert np.isfinite(result["incongruent_modulation"].iloc[0])
    assert np.isfinite(result["within_person_disc_range"].iloc[0])
