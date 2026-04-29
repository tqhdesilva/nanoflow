"""NanoFlow — unified entry point."""

import os
import signal
import sys

import hydra
import torch
import torch.distributed as dist
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from torchtnt.framework import fit as tnt_fit
from torchtnt.framework import predict as tnt_predict
from torchtnt.framework.callbacks import LearningRateMonitor
from torchtnt.utils import init_from_env

import config as _config  # noqa: F401 — registers structured config schema
from callbacks import (
    CheckpointCallback,
    EpochSummaryCallback,
    RunDirCallback,
    SampleLoggerCallback,
    StepLossCallback,
    make_run_dir,
)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg) -> None:
    if cfg.device == "mps":
        device = torch.device("mps")
    else:
        device = init_from_env(device_type=cfg.device)

    trained_model = None
    train_data = None

    # --- Training ---
    if cfg.train_unit is not None:
        train_unit = hydra.utils.instantiate(cfg.train_unit, device=device)
        train_loader = hydra.utils.instantiate(cfg.train_loader)
        val_loader = hydra.utils.instantiate(cfg.val_loader)

        run_dir_cb = RunDirCallback(
            runs_dir=cfg.runs_dir,
            run_prefix=cfg.training.run_prefix,
            cfg=cfg,
        )
        ckpt_cb = CheckpointCallback(
            ckpt_dir=run_dir_cb.ckpt_dir,
            save_every=cfg.training.save_every,
            resume=cfg.training.resume,
        )
        epoch_summary = EpochSummaryCallback(
            tb_logger=run_dir_cb.tb_logger,
            total_epochs=cfg.training.epochs,
            batch_size=cfg.training.batch_size,
        )
        step_loss = StepLossCallback(
            tb_logger=run_dir_cb.tb_logger,
            log_every=cfg.training.log_every,
        )
        callbacks = [
            run_dir_cb,
            ckpt_cb,
            epoch_summary,
            step_loss,
            LearningRateMonitor(loggers=run_dir_cb.tb_logger),
        ]
        if cfg.get("sample_logger") is not None:
            scfg = cfg.sample_logger
            callbacks.append(
                SampleLoggerCallback(
                    tb_logger=run_dir_cb.tb_logger,
                    save_every=cfg.training.save_every,
                    latent_shape=list(scfg.latent_shape),
                    n_samples=scfg.n_samples,
                    num_steps=scfg.num_steps,
                    guidance_scale=getattr(scfg, "guidance_scale", 1.0),
                    p_uncond=cfg.training.p_uncond,
                )
            )

        # SIGTERM: write a `preempted.pt` checkpoint, then exit.
        def _handler(sig, frame):
            print(f"\nSIGTERM caught — saving preempted checkpoint to {ckpt_cb.save_path('preempted')}")
            ckpt_cb.save(train_unit, "preempted")
            sys.exit(0)

        signal.signal(signal.SIGTERM, _handler)

        tnt_fit(
            train_unit,
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
            max_epochs=cfg.training.epochs,
            evaluate_every_n_epochs=cfg.training.save_every,
            callbacks=callbacks,
        )

        trained_model = (
            train_unit.swa_model.module
            if getattr(train_unit, "swa_model", None) is not None
            else train_unit._raw_module
        )
        train_data = getattr(train_loader, "dataset", None)

    # --- Inference (rank 0 only) ---
    if cfg.inference is not None and int(os.environ.get("RANK", 0)) == 0:
        from viz import plot_image_samples, plot_samples

        icfg = cfg.inference
        cs = OmegaConf.select(icfg, "class_sampler", default=None)
        if trained_model is not None:
            model = trained_model
            model.eval()
            infer_unit = hydra.utils.instantiate(
                icfg.infer_unit, model=model, class_sampler=cs
            )
        else:
            infer_unit = hydra.utils.instantiate(icfg.infer_unit, class_sampler=cs)

        if cs is not None:
            if cs.probs is not None:
                probs = torch.tensor(cs.probs)
                labels = torch.multinomial(probs, icfg.n_samples, replacement=True)
            else:
                labels = torch.randint(0, cs.num_classes, (icfg.n_samples,))
            predict_loader = DataLoader(
                TensorDataset(torch.zeros(icfg.n_samples), labels),
                batch_size=icfg.n_samples,
            )
        else:
            predict_loader = DataLoader(
                TensorDataset(torch.zeros(icfg.n_samples)),
                batch_size=icfg.n_samples,
            )
        tnt_predict(infer_unit, predict_loader)
        generated = torch.cat(infer_unit.results, dim=0).cpu()

        metrics_cfg = OmegaConf.select(icfg, "metrics", default=None)
        if metrics_cfg:
            if "run_dir_cb" in locals():
                run_dir = run_dir_cb.run_dir
            else:
                run_dir = make_run_dir(cfg.runs_dir, cfg.training.run_prefix)
                print(f"Run dir: {run_dir}")
            results = []
            for m_cfg in metrics_cfg:
                metric = hydra.utils.instantiate(m_cfg)
                results.append(metric(generated))
            metrics_path = run_dir / "metrics.yaml"
            with open(metrics_path, "w") as f:
                yaml.dump(results, f, sort_keys=False)
            print(f"Wrote metrics → {metrics_path}", flush=True)

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

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
