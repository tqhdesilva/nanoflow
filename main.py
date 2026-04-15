"""NanoFlow — unified entry point."""

import hydra
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchtnt.framework import train as tnt_train, predict as tnt_predict

import config as _config  # noqa: F401 — registers structured config schema
from config import Config
from unit import load_ema
from viz import plot_samples, plot_image_samples


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: Config) -> None:
    trained_model = None
    ema_params = None
    train_data = None

    # --- Training ---
    if cfg.train_unit is not None:
        train_unit = hydra.utils.instantiate(cfg.train_unit)
        tnt_train(
            train_unit, train_unit.train_loader, max_epochs=train_unit.tcfg.epochs
        )
        train_unit.cleanup()

        if train_unit.rank == 0:
            trained_model = train_unit._unwrap()
            if train_unit.ema_ready():
                ema_params = train_unit.ema_params
            train_data = train_unit.train_ds

    # --- Inference ---
    if cfg.inference is not None:
        icfg = cfg.inference
        if trained_model is not None:
            model = trained_model
            model.eval()
            if ema_params is not None:
                load_ema(model, ema_params)
            infer_unit = hydra.utils.instantiate(icfg.infer_unit, model=model)
        else:
            infer_unit = hydra.utils.instantiate(icfg.infer_unit)

        # TODO: for conditional generation, replace with actual conditioning
        # tensors (class labels, embeddings, etc.) of shape (n_samples, ...).
        # predict_step infers n_samples from data.shape[0].
        predict_loader = DataLoader(
            TensorDataset(torch.zeros(icfg.n_samples)), batch_size=icfg.n_samples
        )
        tnt_predict(infer_unit, predict_loader)
        generated = torch.cat(infer_unit.results, dim=0)

        if icfg.save_path:
            title = f"{infer_unit.num_steps} Euler steps, {icfg.n_samples} samples"
            if infer_unit.latent_shape is None:
                real = train_data.data if train_data is not None else None
                if real is not None:
                    plot_samples(real, generated, title, icfg.save_path)
                else:
                    print(
                        f"Generated {generated.shape[0]} samples"
                        " (no real data for comparison)"
                    )
            else:
                plot_image_samples(generated, title, icfg.save_path)
        else:
            print(
                f"Generated {generated.shape[0]} samples (set save_path to write plot)"
            )


if __name__ == "__main__":
    main()
