import numpy as np
import torch

from bankstudy.model import MLP, load_checkpoint, predict, train_mlp


def test_mlp_produces_one_logit_per_row_including_singleton():
    model = MLP(5, [8, 4], 0.1)
    assert model(torch.zeros(7, 5)).shape == (7,)
    assert model(torch.zeros(1, 5)).shape == (1,)


def test_training_checkpoint_roundtrip_and_seed_reproducibility(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(40, 5)).astype(np.float32)
    y = np.tile([0, 0, 0, 1], 10)
    config = {"hidden": [8], "dropout": 0.1, "learning_rate": 0.01, "weight_decay": 0.001,
              "batch_size": 8, "max_epochs": 4, "patience": 2, "min_delta": 0.0001}
    model, history = train_mlp(x[:24], y[:24], x[24:], y[24:], config, "balanced", 42, tmp_path / "a.pt")
    loaded, saved = load_checkpoint(tmp_path / "a.pt")
    np.testing.assert_array_equal(predict(model, x), predict(loaded, x))
    assert saved["epoch"] == history["best_epoch"]
    assert saved["pos_weight"] == 3.0
    assert saved["architecture"]["input_dim"] == 5
    second, _ = train_mlp(x[:24], y[:24], x[24:], y[24:], config, "balanced", 42, tmp_path / "b.pt")
    np.testing.assert_array_equal(predict(model, x), predict(second, x))
    assert len(history["history"]) <= 4
