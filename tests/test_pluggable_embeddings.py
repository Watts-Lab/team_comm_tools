from pathlib import Path
import ast
import importlib.util
import sys
import types

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_EMBEDDINGS_PATH = REPO_ROOT / "src" / "team_comm_tools" / "utils" / "check_embeddings.py"
FEATURE_BUILDER_PATH = REPO_ROOT / "src" / "team_comm_tools" / "feature_builder.py"


def load_check_embeddings_module(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = lambda *args, **kwargs: types.SimpleNamespace(
        encode=lambda texts: np.array([[float(len(text))] for text in texts])
    )
    fake_sentence_transformers.util = object()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: object()
    )
    fake_transformers.AutoModelForSequenceClassification = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: object()
    )
    fake_transformers.logging = types.SimpleNamespace(set_verbosity=lambda *args, **kwargs: None)

    fake_scipy = types.ModuleType("scipy")
    fake_scipy_special = types.ModuleType("scipy.special")
    fake_scipy_special.softmax = lambda values, axis=None: values
    fake_scipy.special = fake_scipy_special

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
    monkeypatch.setitem(sys.modules, "scipy.special", fake_scipy_special)

    spec = importlib.util.spec_from_file_location("check_embeddings_test_module", CHECK_EMBEDDINGS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_feature_builder_module(monkeypatch, check_embeddings_module):
    fake_package = types.ModuleType("team_comm_tools")
    fake_package.__path__ = []
    fake_utils_package = types.ModuleType("team_comm_tools.utils")
    fake_utils_package.__path__ = []

    fake_download_resources = types.ModuleType("team_comm_tools.utils.download_resources")
    fake_download_resources.download = lambda: None

    fake_chat_calc = types.ModuleType("team_comm_tools.utils.calculate_chat_level_features")
    fake_chat_calc.ChatLevelFeaturesCalculator = object

    fake_user_calc = types.ModuleType("team_comm_tools.utils.calculate_user_level_features")
    fake_user_calc.UserLevelFeaturesCalculator = object

    fake_conv_calc = types.ModuleType("team_comm_tools.utils.calculate_conversation_level_features")
    fake_conv_calc.ConversationLevelFeaturesCalculator = object

    fake_preprocess = types.ModuleType("team_comm_tools.utils.preprocess")
    fake_preprocess.preprocess_conversation_columns = (
        lambda df, column_names, grouping_keys, cumulative_grouping, within_task: df
    )
    fake_preprocess.remove_unhashable_cols = lambda df, column_names: df
    fake_preprocess.preprocess_text_lowercase_but_retain_punctuation = lambda text: str(text).lower()
    fake_preprocess.preprocess_text = lambda text: str(text).lower()
    fake_preprocess.preprocess_naive_turns = lambda df, column_names: df

    fake_check_embeddings = types.ModuleType("team_comm_tools.utils.check_embeddings")
    fake_check_embeddings.build_vector_cache_path = check_embeddings_module.build_vector_cache_path
    fake_check_embeddings.check_embeddings = lambda *args, **kwargs: None

    feature_names = [
        "Named Entity Recognition",
        "Sentiment (RoBERTa)",
        "Message Length",
        "Message Quantity",
        "Information Exchange",
        "LIWC and Other Lexicons",
        "Questions",
        "Conversational Repair",
        "Word Type-Token Ratio",
        "Proportion of First-Person Pronouns",
        "Function Word Accommodation",
        "Content Word Accommodation",
        "Hedge",
        "TextBlob Subjectivity",
        "TextBlob Polarity",
        "Positivity Z-Score",
        "Dale-Chall Score",
        "Time Difference",
        "Politeness Strategies",
        "Politeness / Receptiveness Markers",
        "Certainty",
        "Online Discussion Tags",
        "Turn-Taking Index",
        "Equal Participation",
        "Team Burstiness",
        "Conversation Level Aggregates",
        "User Level Aggregates",
        "Information Diversity",
    ]

    fake_feature_dict = types.ModuleType("team_comm_tools.feature_dict")
    fake_feature_dict.feature_dict = {}
    for name in feature_names:
        fake_feature_dict.feature_dict[name] = {
            "vect_data": name == "Information Diversity",
            "bert_sentiment_data": name == "Sentiment (RoBERTa)",
            "level": "Conversation" if name in {
                "Turn-Taking Index",
                "Equal Participation",
                "Team Burstiness",
                "Conversation Level Aggregates",
                "User Level Aggregates",
                "Information Diversity",
            } else "Chat",
            "function": f"func_{name}",
            "columns": [],
        }

    monkeypatch.setitem(sys.modules, "team_comm_tools", fake_package)
    monkeypatch.setitem(sys.modules, "team_comm_tools.utils", fake_utils_package)
    monkeypatch.setitem(sys.modules, "team_comm_tools.utils.download_resources", fake_download_resources)
    monkeypatch.setitem(sys.modules, "team_comm_tools.utils.calculate_chat_level_features", fake_chat_calc)
    monkeypatch.setitem(sys.modules, "team_comm_tools.utils.calculate_user_level_features", fake_user_calc)
    monkeypatch.setitem(sys.modules, "team_comm_tools.utils.calculate_conversation_level_features", fake_conv_calc)
    monkeypatch.setitem(sys.modules, "team_comm_tools.utils.preprocess", fake_preprocess)
    monkeypatch.setitem(sys.modules, "team_comm_tools.utils.check_embeddings", fake_check_embeddings)
    monkeypatch.setitem(sys.modules, "team_comm_tools.feature_dict", fake_feature_dict)

    spec = importlib.util.spec_from_file_location("feature_builder_test_module", FEATURE_BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.feature_dict = fake_feature_dict.feature_dict
    return module


def test_build_vector_cache_path_keeps_default_location(monkeypatch):
    check_embeddings_module = load_check_embeddings_module(monkeypatch)
    default_path = "vector_data/sentence/chats/output.csv"

    assert check_embeddings_module.build_vector_cache_path(default_path) == default_path


def test_build_vector_cache_path_changes_when_backend_changes(monkeypatch):
    check_embeddings_module = load_check_embeddings_module(monkeypatch)

    def fake_encoder(texts):
        return np.array([[float(len(text)), 1.0, 2.0] for text in texts])

    path_one = check_embeddings_module.build_vector_cache_path(
        "vector_data/sentence/chats/output.csv",
        embedding_fn=fake_encoder,
        embedding_backend_id="openai-text-embedding-3-small",
        embedding_dim=3,
    )
    path_two = check_embeddings_module.build_vector_cache_path(
        "vector_data/sentence/chats/output.csv",
        embedding_fn=fake_encoder,
        embedding_backend_id="openai-text-embedding-3-large",
        embedding_dim=3,
    )

    assert path_one != path_two
    assert path_one.endswith(".csv")
    assert path_two.endswith(".csv")


def test_generate_vect_uses_custom_embedding_fn(tmp_path, monkeypatch):
    check_embeddings_module = load_check_embeddings_module(monkeypatch)
    calls = []

    def fake_encoder(texts):
        calls.append(list(texts))
        return np.array([[float(len(text)), float(len(text)) + 1.0] for text in texts])

    chat_data = pd.DataFrame({"message": ["hello", " ", "world"]})
    output_path = tmp_path / "vectors.csv"

    check_embeddings_module.generate_vect(
        chat_data,
        str(output_path),
        "message",
        batch_size=8,
        embedding_fn=fake_encoder,
        embedding_dim=2,
    )

    output_df = pd.read_csv(output_path)
    first_vector = np.array(ast.literal_eval(output_df.iloc[0]["message_embedding"]))
    empty_vector = np.array(ast.literal_eval(output_df.iloc[1]["message_embedding"]))
    third_vector = np.array(ast.literal_eval(output_df.iloc[2]["message_embedding"]))

    assert calls == [["hello", "world"]]
    assert np.array_equal(first_vector, np.array([5.0, 6.0]))
    assert np.array_equal(empty_vector, np.array([0.0, 0.0]))
    assert np.array_equal(third_vector, np.array([5.0, 6.0]))


def test_feature_builder_passes_custom_embedding_config_into_cache_and_generation(tmp_path, monkeypatch):
    check_embeddings_module = load_check_embeddings_module(monkeypatch)
    feature_builder_module = load_feature_builder_module(monkeypatch, check_embeddings_module)
    captured = {}

    def fake_check_embeddings(chat_data, vect_path, bert_path, need_sentence, need_sentiment,
                              regenerate_vectors, message_col="message", embedding_fn=None, embedding_dim=None):
        captured["vect_path"] = vect_path
        captured["bert_path"] = bert_path
        captured["embedding_fn"] = embedding_fn
        captured["embedding_dim"] = embedding_dim

        Path(vect_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "message": chat_data[message_col],
                "message_embedding": ["[0.0, 0.0, 0.0]"] * len(chat_data),
            }
        ).to_csv(vect_path, index=False)

        Path(bert_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "positive_bert": [0.0] * len(chat_data),
                "negative_bert": [0.0] * len(chat_data),
                "neutral_bert": [1.0] * len(chat_data),
            }
        ).to_csv(bert_path, index=False)

    feature_builder_module.check_embeddings = fake_check_embeddings

    def fake_encoder(texts):
        return np.array([[1.0, 2.0, 3.0] for _ in texts])

    input_df = pd.DataFrame(
        {
            "conversation_num": [1, 1],
            "speaker_nickname": ["a", "b"],
            "message": ["hello", "world"],
            "timestamp": [1, 2],
        }
    )

    builder = feature_builder_module.FeatureBuilder(
        input_df=input_df,
        vector_directory=f"{tmp_path}/",
        output_file_path_chat_level=str(tmp_path / "chat" / "custom_vectors.csv"),
        output_file_path_conv_level=str(tmp_path / "conv" / "custom_vectors.csv"),
        output_file_path_user_level=str(tmp_path / "user" / "custom_vectors.csv"),
        embedding_fn=fake_encoder,
        embedding_backend_id="openai-text-embedding-3-small",
        embedding_dim=3,
    )

    assert captured["embedding_fn"] is fake_encoder
    assert captured["embedding_dim"] == 3
    assert captured["bert_path"] == f"{tmp_path}/sentiment/chats/custom_vectors.csv"
    assert captured["vect_path"] != f"{tmp_path}/sentence/chats/custom_vectors.csv"
    assert builder.vect_path == captured["vect_path"]
