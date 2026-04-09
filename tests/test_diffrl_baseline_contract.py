"""Guard core Ant SHAC defaults against accidental drift from DiffRL baseline."""

from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET

import yaml

from msk_warp import PACKAGE_ROOT
from msk_warp.envs.ant import AntEnv


def _load_ant_cfg() -> dict:
    cfg_path = PACKAGE_ROOT / "configs" / "ant_shac.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _mean_motor_gear(model_xml_path) -> float:
    root = ET.parse(model_xml_path).getroot()
    gears = []
    for motor in root.findall("./actuator/motor"):
        gears.append(float(motor.attrib.get("gear", "1").split()[0]))
    assert gears, "ant.xml must define motor actuators"
    return float(sum(gears) / len(gears))


def test_ant_shac_hyperparams_match_diffrl_baseline_contract():
    cfg = _load_ant_cfg()
    conf = cfg["params"]["config"]
    env = cfg["params"]["env"]
    net = cfg["params"]["network"]

    assert conf["gamma"] == 0.99
    assert conf["steps_num"] == 32
    assert abs(float(conf["actor_learning_rate"]) - 2e-3) < 1e-12
    assert abs(float(conf["critic_learning_rate"]) - 2e-3) < 1e-12
    assert conf["lr_schedule"] == "linear"
    assert conf["target_critic_alpha"] == 0.2
    assert conf["obs_rms"] is True
    assert conf["ret_rms"] is False
    assert conf["critic_iterations"] == 16
    assert conf["critic_method"] == "td-lambda"
    assert conf["lambda"] == 0.95
    assert conf["num_batch"] == 4
    assert conf["truncate_grads"] is True
    assert conf["grad_norm"] == 1.0
    assert conf["betas"] == [0.7, 0.95]

    assert env["episode_length"] == 1000
    assert env["num_actors"] == 64
    assert env["stochastic_init"] is True
    assert env["substeps"] == 16
    assert env["model_path"] == "assets/ant.xml"

    assert net["actor"] == "ActorStochasticMLP"
    assert net["actor_mlp"]["units"] == [128, 64, 32]
    assert net["actor_mlp"]["activation"] == "elu"
    assert net["critic"] == "CriticMLP"
    assert net["critic_mlp"]["units"] == [64, 64]
    assert net["critic_mlp"]["activation"] == "elu"


def test_ant_action_scale_matches_diffrl_effective_scale():
    cfg = _load_ant_cfg()
    env = cfg["params"]["env"]
    ant_xml = PACKAGE_ROOT / "assets" / "ant.xml"

    mean_gear = _mean_motor_gear(ant_xml)
    effective_scale = float(env["action_strength"]) * mean_gear

    # DiffRL ant applies action_strength=200.0 directly in env.step()
    # (its XML actuator gears are not used by dflex parse_mjcf).
    assert abs(effective_scale - 200.0) < 1e-6


def test_ant_reward_defaults_match_diffrl_formula():
    sig = inspect.signature(AntEnv.__init__)
    params = sig.parameters

    assert params["forward_vel_weight"].default == 1.0
    assert params["heading_weight"].default == 1.0
    assert params["up_weight"].default == 0.1
    assert params["height_weight"].default == 1.0
    assert params["joint_vel_penalty"].default == 0.0
    assert params["push_reward_weight"].default == 0.0
