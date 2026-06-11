"""PPO (rsl_rl) on a MolmoSpaces drawer in Isaac Lab: a Franka arm learns to open a
drawer extracted from the converted iTHOR FloorPlan1 kitchen (assets_rl/molmo_drawer.usda).

Direct-workflow env adapted surgically from isaaclab_tasks.direct.franka_cabinet —
same MDP structure/rewards/PPO hyperparameters, MolmoSpaces asset instead of Sektion.

Usage: python -u train_molmo_drawer.py --num_envs 2048 --max_iterations 250
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=250)
parser.add_argument("--smoke", action="store_true", help="tiny run to validate mechanics")
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------------
import torch
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform

MOLMO_DRAWER_USD = os.path.expanduser("~/assets_rl/molmo_drawer.usda")
SLIDER_LINK = "drawer_372d9ee41d70550432c30a66a6e5b331_1_1_0"
DRAWER_JOINT = "drawer_372d9ee41d70550432c30a66a6e5b331_1_1_0_joint_0"
DRAWER_TRAVEL = 0.319  # m, prismatic upper limit
# from extract_drawer_asset.py: front-face center in slider-local frame (+4cm standoff)
# top-hook grasp for the handle-less iTHOR drawer: fingertips dip over the rim just
# behind the front panel (slider-local: y=up-ish world Z, z=open dir world +X after spawn rot)
GRASP_LOCAL = (0.0, 0.01, -0.035)
INWARD_AXIS = (0.0, -1.0, 0.0)  # gripper forward -> world -Z (point down into the box)
UP_AXIS = (0.0, 0.0, 1.0)       # gripper finger axis -> world +X (straddle the panel)


@configclass
class MolmoDrawerEnvCfg(DirectRLEnvCfg):
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 9
    observation_space = 23
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
    )

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=5.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(0.62, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"], effort_limit_sim=87.0, stiffness=80.0, damping=4.0
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"], effort_limit_sim=12.0, stiffness=80.0, damping=4.0
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"], effort_limit_sim=200.0, stiffness=2e3, damping=1e2
            ),
        },
    )

    # MolmoSpaces drawer (from iTHOR FloorPlan1), rotated -90deg about Z so it opens toward +X
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Drawer",
        spawn=sim_utils.UsdFileCfg(
            usd_path=MOLMO_DRAWER_USD,
            activate_contact_sensors=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.7071068, 0.0, 0.0, -0.7071068),
            joint_pos={DRAWER_JOINT: 0.0},
        ),
        actuators={
            "drawer": ImplicitActuatorCfg(
                joint_names_expr=[".*joint_0"], effort_limit_sim=87.0, stiffness=0.0, damping=2.0
            ),
        },
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 0.0  # z-straddle penalty assumes a frontal handle; not our geometry


class MolmoDrawerEnv(DirectRLEnv):
    cfg: MolmoDrawerEnvCfg

    def __init__(self, cfg: MolmoDrawerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos, xformable, device):
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()
            px, py, pz = (world_pos[i] - env_pos[i] for i in range(3))
            qx, qy, qz = world_quat.imaginary
            return torch.tensor([px, py, pz, world_quat.real, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )
        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])
        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        drawer_local_grasp_pose = torch.tensor(list(GRASP_LOCAL) + [1.0, 0.0, 0.0, 0.0], device=self.device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor(INWARD_AXIS, device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_up_axis = torch.tensor(UP_AXIS, device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.drawer_link_idx = self._cabinet.find_bodies(SLIDER_LINK)[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["cabinet"] = self._cabinet

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self):
        terminated = self._cabinet.data.joint_pos[:, 0] > 0.28
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self):
        self._compute_intermediate_values()
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        d = torch.norm(self.robot_grasp_pos - self.drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        axis1 = tf_vector(self.robot_grasp_rot, self.gripper_forward_axis)
        axis2 = tf_vector(self.drawer_grasp_rot, self.drawer_inward_axis)
        axis3 = tf_vector(self.robot_grasp_rot, self.gripper_up_axis)
        axis4 = tf_vector(self.drawer_grasp_rot, self.drawer_up_axis)
        dot1 = torch.bmm(axis1.view(self.num_envs, 1, 3), axis2.view(self.num_envs, 3, 1)).squeeze()
        dot2 = torch.bmm(axis3.view(self.num_envs, 1, 3), axis4.view(self.num_envs, 3, 1)).squeeze()
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        action_penalty = torch.sum(self.actions**2, dim=-1)
        open_reward = self._cabinet.data.joint_pos[:, 0]

        lfinger_dist = robot_left_finger_pos[:, 2] - self.drawer_grasp_pos[:, 2]
        rfinger_dist = self.drawer_grasp_pos[:, 2] - robot_right_finger_pos[:, 2]
        finger_dist_penalty = torch.zeros_like(lfinger_dist)
        finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        rewards = (
            self.cfg.dist_reward_scale * dist_reward
            + self.cfg.rot_reward_scale * rot_reward
            + self.cfg.open_reward_scale * open_reward
            + self.cfg.finger_reward_scale * finger_dist_penalty
            - self.cfg.action_penalty_scale * action_penalty
        )
        jp = self._cabinet.data.joint_pos[:, 0]
        rewards = torch.where(jp > 0.01, rewards + 0.25, rewards)
        rewards = torch.where(jp > 0.12, rewards + 0.25, rewards)
        rewards = torch.where(jp > 0.25, rewards + 0.25, rewards)

        self.extras["log"] = {
            "dist_reward": (self.cfg.dist_reward_scale * dist_reward).mean(),
            "rot_reward": (self.cfg.rot_reward_scale * rot_reward).mean(),
            "open_reward": (self.cfg.open_reward_scale * open_reward).mean(),
            "drawer_open_m": jp.mean(),
            "drawer_open_frac": (jp / DRAWER_TRAVEL).mean(),
        }
        return rewards

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125, 0.125, (len(env_ids), self._robot.num_joints), self.device
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)
        self._compute_intermediate_values(env_ids)

    def _get_observations(self):
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        to_target = self.drawer_grasp_pos - self.robot_grasp_pos
        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
                self._cabinet.data.joint_pos[:, 0].unsqueeze(-1),
                self._cabinet.data.joint_vel[:, 0].unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _compute_intermediate_values(self, env_ids=None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        drawer_pos = self._cabinet.data.body_pos_w[env_ids, self.drawer_link_idx]
        drawer_rot = self._cabinet.data.body_quat_w[env_ids, self.drawer_link_idx]
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
        ) = tf_combine(hand_rot, hand_pos, self.robot_local_grasp_rot[env_ids], self.robot_local_grasp_pos[env_ids])
        (
            self.drawer_grasp_rot[env_ids],
            self.drawer_grasp_pos[env_ids],
        ) = tf_combine(
            drawer_rot, drawer_pos, self.drawer_local_grasp_rot[env_ids], self.drawer_local_grasp_pos[env_ids]
        )


# ------------------------------- training -------------------------------
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaaclab_tasks.direct.franka_cabinet.agents.rsl_rl_ppo_cfg import FrankaCabinetPPORunnerCfg

env_cfg = MolmoDrawerEnvCfg()
env_cfg.scene.num_envs = args.num_envs
if args.smoke:
    env_cfg.scene.num_envs = 8

env = MolmoDrawerEnv(cfg=env_cfg)
print(f">>> ENV READY: {env.num_envs} envs, device={env.device}", flush=True)
print(f">>> drawer joints: {env._cabinet.joint_names}", flush=True)
print(f">>> drawer bodies: {env._cabinet.body_names}", flush=True)

env = RslRlVecEnvWrapper(env)
agent_cfg = FrankaCabinetPPORunnerCfg()
agent_cfg.experiment_name = "molmo_drawer"
agent_cfg.max_iterations = 2 if args.smoke else args.max_iterations

log_dir = os.path.expanduser("~/rl_logs/molmo_drawer")
os.makedirs(log_dir, exist_ok=True)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
runner.save(os.path.join(log_dir, "final_checkpoint.pt"))
print(">>> TRAINING DONE", flush=True)
env.close()
simulation_app.close()
os._exit(0)
