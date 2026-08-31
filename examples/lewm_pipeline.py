from pathlib import Path
import torch
from aster.core import ArtifactStore, atomic_json
from aster.data.actions import ActionSpec, ActionNormalizer
from aster.data.lewm import fit_lewm_actions
from aster.data.datasets import StatefulSampler
from aster.data.tensors import TensorTreeDataset
from aster.models import load_model
from aster.models.vit import ViTConfig
from aster.models.lewm import LeWMConfig, LeWorldModel
from aster.methods.lewm import LeWMMethod, LeWMObjective
from aster.planning.lewm import LeWMCEM, LeWMCEMConfig, LeWMMPC
from aster.training import Trainer


def render(position):

    grid = torch.linspace(-1, 1, 16, device=position.device)
    x = position[..., None, None]
    dot = torch.exp(-((grid[None, :] - x).square() + grid[:, None].square()) / 0.07)
    return torch.stack((x.expand_as(dot), dot * 2 - 1, grid[None, :].expand_as(dot)), dim=-3)


class VisualPoint:
    def __init__(self, position, goal):
        self.position, self.goal = position.clone(), goal.clone()

    def observation(self):
        return render(self.position)[:, None]

    def goal_observation(self):
        return render(self.goal)[:, None]

    def step(self, delta):

        self.position = (self.position + delta[:, 0]).clamp(-1, 1)
        return -(self.position - self.goal).abs()


def run_demo(output_dir, *, seed=743, steps=600):
    if type(steps) is not int or steps < 1:
        raise ValueError("Positive training steps required")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(seed + 1)
    positions = torch.rand(2048, generator=rng) * 2 - 1
    successor = torch.rand(2048, generator=rng) * 2 - 1

    controls = (successor - positions)[:, None]
    spec = ActionSpec(
        ("x_delta",), ("normalized_world_distance",), "visual_point_1d", "delta", 5.0, 1
    )
    normalizer = fit_lewm_actions(controls[:1792], spec=spec)
    pixels = torch.stack((render(positions), render(successor)), 1)
    actions = normalizer.normalize(controls)[:, None]
    c = LeWMConfig(
        encoder=ViTConfig(
            hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32
        ),
        embed_dim=8,
        action_dim=1,
        history_size=1,
        predictor_hidden_dim=16,
        predictor_depth=1,
        predictor_heads=2,
        predictor_head_dim=8,
        predictor_mlp_dim=32,
        projector_hidden_dim=32,
    )
    model = LeWorldModel(c)
    engine = Trainer(model, lr=0.001, max_grad_norm=1.0)
    method = LeWMMethod(engine, objective=LeWMObjective(num_proj=64), seed=seed + 2)
    torch.save(dict(pixels=pixels[:1792], actions=actions[:1792]), output / "train.pt")
    dataset = TensorTreeDataset(
        output / "train.pt",
        preprocessing=dict(
            pixel_contract="visual_point_rgb_torch_v1",
            normalizer=normalizer.to_dict(),
            split="train_1792",
        ),
    )
    sampler = StatefulSampler(dataset, seed=seed + 4)
    engine.register_state("data_sampler", sampler)
    losses = []
    for _ in range(steps):
        rows = sampler.take(64)
        if not rows:
            sampler.next_epoch()
            rows = sampler.take(64)
        data = {key: torch.stack([row[key] for row in rows]) for key in ("pixels", "actions")}
        losses.append(method.update([data]).loss)

    checkpoint = engine.save_checkpoint(output / "checkpoint")
    model.eval()
    with torch.no_grad():
        heldout = model(pixels[1792:], actions[1792:])
        heldout_mse = (heldout.predictions - heldout.embeddings[:, 1:]).square().mean().item()
        latent_std = heldout.embeddings.flatten(0, 1).std(0).mean().item()
    exported = output / "model"
    model.save_pretrained(exported)
    atomic_json(exported / "action_normalization.json", normalizer.to_dict())
    atomic_json(
        exported / "pixel_contract.json",
        dict(type="visual_point_rgb_torch_v1", size=16, range=[-1, 1]),
    )
    artifact = ArtifactStore(output / "artifacts").publish(
        exported,
        kind="world_model",
        metadata=dict(
            architecture="lewm",
            model_source="lucas-maes/le-wm@8edfeb336732b5f3ce7b8b210d0ba370a09e2cac",
            training_data_fingerprint=dataset.fingerprint,
            validation="local_1d_pixel_control_not_public_pusht",
        ),
    )
    restored = load_model(artifact.path).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            restored(pixels[1792:], actions[1792:]).predictions, heldout.predictions, atol=0, rtol=0
        )
    planner = LeWMCEM(
        restored,
        LeWMCEMConfig(horizon=1, num_samples=64, topk=8, n_steps=6, batch_size=4),
        seed=seed + 3,
    )
    policy = LeWMMPC(planner, normalizer=normalizer)
    env = VisualPoint(
        torch.tensor([-0.75, 0.75, -0.45, 0.45]), torch.tensor([0.65, -0.65, 0.4, -0.4])
    )
    start_error = (env.position - env.goal).abs()
    returns = torch.zeros(4)
    for _ in range(8):
        returns += env.step(policy.act(env.observation(), env.goal_observation()))
    error = (env.position - env.goal).abs()
    report = dict(
        seed=seed,
        steps=steps,
        train_loss_first=losses[0],
        train_loss_last=losses[-1],
        heldout_latent_mse=heldout_mse,
        heldout_latent_std=latent_std,
        initial_goal_error=start_error.mean().item(),
        final_goal_error=error.mean().item(),
        goal_errors=error.tolist(),
        success_rate=(error < 0.15).float().mean().item(),
        mean_return=returns.mean().item(),
        zero_action_mean_return=(-8 * start_error.mean()).item(),
        evaluation_episodes=len(error),
        success_threshold=0.15,
        artifact_id=artifact.id,
        artifact_path=str(artifact.path),
        checkpoint_path=str(checkpoint),
        benchmark="local_1d_pixel_control_not_public_pusht",
    )
    atomic_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output_dir, steps=args.steps), ensure_ascii=False, indent=2))
