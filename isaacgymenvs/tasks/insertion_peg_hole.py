# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import *

import theseus

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from gym import spaces

import torch

from isaacgymenvs.tasks.base.vec_task import VecTask


class InsertionPegHole(VecTask):
    """
    Generic class for insertion tasks with a hole and a peg.
    """

    def __init__(
            self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render,
    ):
        assert cfg["env"]["recomputePrePhysicsStep"], "InsertionBox2D requires recomputePrePhysicsStep to be True"

        self.cfg = cfg

        self.n_taskpace_pos = self.cfg["env"]["n_taskpace_pos"]  # 2 prismatic joints
        self.n_taskpace_rot = self.cfg["env"]["n_taskpace_rot"]  # 1 revolute joint
        self.n_taskpace_dims = self.n_taskpace_pos + self.n_taskpace_rot
        self.num_dof = self.n_taskpace_dims

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        ############################
        # Hole and peg assets
        self.asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        if "asset" in self.cfg["env"]:
            self.asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", self.asset_root))
            self.asset_file_hole = self.cfg["env"]["asset"]["assetFileNameHole"]
            self.asset_file_peg = self.cfg["env"]["asset"]["assetFileNamePeg"]
        else:
            raise KeyError

        ############################
        # learning parameters
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.learn_rotations = self.cfg["env"]["learnRotations"]
        self.learn_stiffness = self.cfg["env"]["learnStiffness"]
        self.just_learn_stiffness = self.cfg["env"].get("justLearnStiffness", False)
        if self.just_learn_stiffness:
            self.enable_nominal_policy = True
            self.learn_stiffness = True
        if self.learn_stiffness:
            self._K_pos_min = self.cfg["env"].get("K_pos_min", 10.)
            self._K_pos_max = self.cfg["env"].get("K_pos_max", 100.)
            self._K_rot_min = self.cfg["env"].get("K_rot_min", 5.)
            self._K_rot_max = self.cfg["env"].get("K_rot_max", 50.)
            self._delta_K_pos = .5 * (self._K_pos_max - self._K_pos_min)
            self._central_K_pos = .5 * (self._K_pos_max + self._K_pos_min)
            self._delta_K_rot = .5 * (self._K_rot_max - self._K_rot_min)
            self._central_K_rot = .5 * (self._K_rot_max + self._K_rot_min)

        self.enable_sparse_reward = self.cfg["env"]["enableSparseReward"]

        ############################
        # observations (and state)
        # All relative to the world frame

        # 0:2 - box position
        # 3:11 - box orientation
        # 12:14 - box linear velocity
        # 15:17 - box angular velocity
        self.cfg["env"]["numObservations"] = 3 + 9 + 3 + 3

        ############################
        # actions
        # action is the desired EE velocity in the base frame (and the stiffness parameter)
        # on the 2d space (2 prismatic (x and y) + 1 revolute (rotation around-axis))
        self.action_pos_delta_idxs = None
        self.action_rot_delta_idxs = None
        self.action_K_idxs = None
        if self.just_learn_stiffness:
            self.cfg["env"]["numActions"] = self.n_taskpace_pos + self.n_taskpace_rot  # 2 K_pos + 1 K_rot
            self.action_K_idxs = np.arange(0, self.n_taskpace_pos + self.n_taskpace_rot)
        else:
            self.action_pos_delta_idxs = np.arange(0, self.n_taskpace_pos)
            self.cfg["env"]["numActions"] = self.n_taskpace_pos
            extra_actions = 0
            if not self.learn_rotations:
                if self.learn_stiffness:
                    extra_actions += self.n_taskpace_pos  # parameters of K_pos
                    self.action_K_idxs = np.arange(self.n_taskpace_pos, self.n_taskpace_pos + self.n_taskpace_pos)
            else:
                self.action_rot_delta_idxs = np.arange(self.n_taskpace_pos, self.n_taskpace_pos + self.n_taskpace_rot)
                extra_actions += self.n_taskpace_rot  # orientation
                if self.learn_stiffness:
                    extra_actions += self.n_taskpace_pos + self.n_taskpace_rot  # parameters of K_pos and K_rot
                    self.action_K_idxs = np.arange(self.n_taskpace_pos, self.n_taskpace_pos + self.n_taskpace_pos + self.n_taskpace_rot)
            self.cfg["env"]["numActions"] += extra_actions

        #######################################################################################################
        # Initialize vectorized task
        super().__init__(
            config=self.cfg,
            rl_device=rl_device, sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless, virtual_screen_capture=virtual_screen_capture,
            force_render=force_render)

        self._num_envs_array = np.arange(self.num_envs)

        ############################
        # buffers for actions
        self.actions_pos_delta = torch.zeros((self.num_envs, 3), device=self.device)
        self.actions_rot_delta = torch.zeros((self.num_envs, 3), device=self.device)

        ############################
        # low-level controller
        self.enable_nominal_policy = self.cfg["env"].get("enableNominalPolicy", False)
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "ik"}, "Invalid control type specified. Must be one of: {osc, ik}"
        self.control_vel = self.cfg["env"]["controlVelocity"]

        self.delta_translation_min = self.cfg["env"].get("delta_translation_min", -0.05)
        self.delta_translation_max = self.cfg["env"].get("delta_translation_max", 0.05)
        self.delta_rotation_max = self.cfg["env"].get("delta_rotation_max", 0.1)

        self.linear_velocity_min = self.cfg["env"].get("linear_velocity_min", -0.05)
        self.linear_velocity_max = self.cfg["env"].get("linear_velocity_max", 0.05)
        self.angular_velocity_min = self.cfg["env"].get("angular_velocity_min", -0.1)
        self.angular_velocity_max = self.cfg["env"].get("angular_velocity_max", 0.1)

        # goal pose of the peg
        self.peg_pos_goal = torch.zeros((self.num_envs, 3), device=self.device)
        self.peg_rot_goal = theseus.SO3(
            quaternion=torch.tensor([1., 0., 0., 0.], device=self.device).unsqueeze(0).repeat(self.num_envs, 1))

        # target pose of the peg for the low-level controller
        self.peg_pos_target = torch.zeros((self.num_envs, 3), device=self.device)
        self.peg_rot_target = theseus.SO3(
            quaternion=torch.tensor([1., 0., 0., 0.], device=self.device).unsqueeze(0).repeat(self.num_envs, 1))
        self.peg_lin_vel_target = torch.zeros((self.num_envs, 3), device=self.device)
        self.peg_ang_vel_target = torch.zeros((self.num_envs, 3), device=self.device)

        ############################
        # buffer for dof state
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # buffer for rigid body state
        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state = gymtorch.wrap_tensor(rb_state_tensor)

        # buffer for jacobian of peg
        # geometric jacobian in the world frame
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "peg")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        self._jacobian_ee = jacobian[:, self.peg_ee_index-1, :, :]

        # buffer for mass matrix of peg
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "peg")
        self._mm = gymtorch.wrap_tensor(_massmatrix)

        # # buffer for force sensor
        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # sensors_per_env = 1
        # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)
        #
        # _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        # self.contact_sensor_tensor = gymtorch.wrap_tensor(_net_cf)
        #
        # #dof_sensor_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        # #self.dof_forces = gymtorch.wrap_tensor(dof_sensor_tensor).view(self.num_envs, 3)
        # #self.first = True

        ############################
        # Impedance control stiffness and damping
        if self.control_type == 'osc':
            self.kp_pos_factor = 1000.
            self.kp_rot_factor = 100.
        elif self.control_type == 'ik':
            self.kp_pos_factor = 100.
            self.kp_rot_factor = 50.

        self.kv_pos_factor = 2. * np.sqrt(self.kp_pos_factor)
        self.kv_rot_factor = 2. * np.sqrt(self.kp_rot_factor)

        self.kp = torch.eye(6, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.kp[:, :3, :3] *= self.kp_pos_factor
        self.kp[:, 3:, 3:] *= self.kp_rot_factor

        self.kv = torch.eye(6, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.kv[:, :3, :3] *= self.kv_pos_factor
        self.kv[:, 3:, 3:] *= self.kv_rot_factor

        self.kp_reduced = torch.zeros((self.num_envs, 3, 3), device=self.device)
        self.kv_reduced = torch.zeros((self.num_envs, 3, 3), device=self.device)

        ############################
        # Initial states
        self.initial_position_bounds = torch.tensor(
            self.cfg["env"].get(
                "initialPositionBounds", [[-1]*self.n_taskpace_pos, [1]*self.n_taskpace_pos]),
            device=self.device
        )

        self.initial_rotation_bounds = torch.tensor(
            self.cfg["env"].get(
                "initialRotationBounds", [[-math.pi/2+1e-3]*self.n_taskpace_rot, [math.pi/2-1e-3]*self.n_taskpace_rot]),
            device=self.device
        )

        self.use_initial_states = self.cfg["env"].get('useInitialStates', 'False')
        if self.use_initial_states:
            # construct a grid of initial states
            top_x = torch.linspace(-0.55, 0.55, steps=50, device=self.device)
            top_y = torch.linspace(0.35, 0.35, steps=50, device=self.device)
            top = torch.stack((top_x, top_y), dim=1)

            middle_left_x = torch.linspace(-0.35, -0.2, steps=25, device=self.device)
            middle_left_y = torch.linspace(0., 0., steps=25, device=self.device)
            middle_left = torch.stack((middle_left_x, middle_left_y), dim=1)

            middle_right_x = torch.linspace(0.2, 0.35, steps=25, device=self.device)
            middle_right_y = torch.linspace(0., 0., steps=25, device=self.device)
            middle_right = torch.stack((middle_right_x, middle_right_y), dim=1)

            bottom_x = torch.linspace(-0.95, 0.95, steps=50, device=self.device)
            bottom_y = torch.linspace(-0.55, -0.95, steps=50, device=self.device)
            bottom = torch.stack((bottom_x, bottom_y), dim=1)

            self.initial_states = torch.cat((top, middle_left, middle_right, bottom))

        ############################
        # visualization for better debugging
        self.axes_geom = gymutil.AxesGeometry(0.3)
        self.target_pose_geom = gymutil.WireframeSphereGeometry(
            0.05, 8, 8, gymapi.Transform(), color=(255/255., 165/255., 0))

    def create_sim(self):
        self.dt = self.sim_params.dt

        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        plane_params.distance = 0.3
        plane_params.dynamic_friction = 0.
        plane_params.static_friction = 0.
        plane_params.restitution = 0.
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(-spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # load asset hole
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.armature = 0.01
        asset_options.disable_gravity = True
        asset_hole = self.gym.load_asset(self.sim, self.asset_root, self.asset_file_hole, asset_options)

        # load asset peg
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.armature = 0.01
        asset_options.disable_gravity = True
        asset_peg = self.gym.load_asset(self.sim, self.asset_root, self.asset_file_peg, asset_options)

        self.peg_ee_index = self.gym.get_asset_rigid_body_dict(asset_peg)["ee"]

        # default pose
        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p = gymapi.Vec3(0., 0., 0.)
            pose.r = gymapi.Quat(0., 0., 0., 1.)
        else:
            pose.p = gymapi.Vec3(0., 0., 0.)
            pose.r = gymapi.Quat(0., 0., 0., 1.)

        self.envs = []
        self.rb_peg_idxs = []
        self.actor_peg_idxs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)

            # Hole
            handle_hole = self.gym.create_actor(env_ptr, asset_hole, pose, "hole", i, 0, 0)
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle_hole)
            for prop in props:
                # https://www.researchgate.net/publication/330003074_Wear_and_coefficient_of_friction_of_PLA_-_Graphite_composite_in_3D_printing_technology
                prop.friction = 0.4
            self.gym.set_actor_rigid_shape_properties(env_ptr, handle_hole, props)

            # Peg
            handle_peg = self.gym.create_actor(env_ptr, asset_peg, pose, "peg", i, 0, 0)
            peg_idx = self.gym.get_actor_index(env_ptr, handle_peg, gymapi.DOMAIN_SIM)
            self.actor_peg_idxs.append(peg_idx)

            # Set properties for the 2 prismatic and 1 revolute joint
            dof_props = self.gym.get_actor_dof_properties(env_ptr, handle_peg)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][2] = gymapi.DOF_MODE_EFFORT
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0

            self.gym.set_actor_dof_properties(env_ptr, handle_peg, dof_props)

            props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle_peg)
            for prop in props:
                # https://www.researchgate.net/publication/330003074_Wear_and_coefficient_of_friction_of_PLA_-_Graphite_composite_in_3D_printing_technology
                prop.friction = 0.4
            self.gym.set_actor_rigid_shape_properties(env_ptr, handle_peg, props)

            # index of peg end-effector rigid body
            rb_idx = self.gym.find_actor_rigid_body_index(env_ptr, handle_peg, "ee", gymapi.DOMAIN_SIM)
            self.rb_peg_idxs.append(rb_idx)

        self.actor_peg_idxs = to_torch(self.actor_peg_idxs, dtype=torch.long, device=self.device)

    def reset_idx(self, env_ids):
        if self.use_initial_states:
            positions = self.initial_states[torch.randint(self.initial_states.shape[0], (len(env_ids),)), :]
            positions = torch.hstack((positions, torch.zeros((len(env_ids), 1), device=self.device)))
        else:
            positions = torch.zeros((len(env_ids), self.num_dof), device=self.device)
            initial_position_min, initial_position_max = self.initial_position_bounds
            positions[..., :self.n_taskpace_pos] = (
                    initial_position_min[None] + (initial_position_max - initial_position_min)[None]
                    * torch.rand((len(env_ids), self.n_taskpace_pos), device=self.device)
            )

            rot_min, rot_max = self.initial_rotation_bounds
            positions[:, self.n_taskpace_pos:] = (
                    rot_min[None] + (rot_max - rot_min)[None]
                    * torch.rand((len(env_ids), self.n_taskpace_rot), device=self.device)
            )

        velocities = torch.zeros_like(positions)

        self.dof_pos[env_ids, :] = positions
        self.dof_vel[env_ids, :] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        actor_peg_idxs = self.actor_peg_idxs[env_ids].to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(actor_peg_idxs), len(env_ids_int32)
        )

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # TODO remove? - see https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/issues/32
        self.gym.simulate(self.sim)

    def reset(self):
        """
        Overwrite since superclass reset method does reset to (0,0) and this is called initially.
        We do not want to start in (0,0).
        """
        self.post_physics_step()
        return super().reset()

    def compute_observations(self, env_ids=None):
        self._refresh()

        if env_ids is None:
            env_ids = self._num_envs_array

        # 3-dimensional data
        peg_positions = self.rb_state[self.rb_peg_idxs][:, 0:3]
        peg_rotations = self.rb_state[self.rb_peg_idxs][:, 3:7]
        peg_linear_velocity = self.rb_state[self.rb_peg_idxs][:, 7:10]
        peg_angluar_velocity = self.rb_state[self.rb_peg_idxs][:, 10:13]

        # the observations are in the 3-dimensional space
        # the policy and value function can use another representation internally
        self.obs_buf[env_ids, 0:3] = peg_positions
        # rotations as rotation matrix
        # the policy and value function can internally use another representation
        peg_rotations_so3 = theseus.SO3(quaternion=quat_xyzw_to_wxyz(peg_rotations))
        self.obs_buf[env_ids, 3:3+9] = peg_rotations_so3.to_matrix().view(-1, 9)
        start_idx = 3+9
        self.obs_buf[env_ids, start_idx:start_idx+3] = peg_linear_velocity
        self.obs_buf[env_ids, start_idx+3:start_idx+3+3] = peg_angluar_velocity

        # external force at EE
        #self.gym.refresh_force_sensor_tensor(self.sim)
        #vec_force = torch.zeros_like(self.obs_buf[..., -2:])
        #vec_force[..., 0:2] = self.vec_sensor_tensor[..., 0:2]
        #self.obs_buf[env_ids, -2:] = vec_force[env_ids]
        # self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.obs_buf[env_ids, -2:] = self.contact_sensor_tensor[self.rb_peg_idxs][env_ids, 0:2] / 50.

        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_insertion_reward(
            self.peg_pos_goal, self.peg_rot_goal,
            self.rb_state, torch.tensor(self.rb_peg_idxs, device=self.device),
            self.reset_buf, self.progress_buf, self.max_episode_length,
            enable_sparse_reward=self.enable_sparse_reward,
        )

    def _update_stiffness_and_damping_matrices(self, actions_K):
        """
        actions_K should be [K_pos_x, K_pos_y, K_pos_z, K_rot_x, K_rot_y, K_rot_z]
        """
        # scale values from actions to defined range and compute stiffness and damping matrices
        # position
        a_pos = torch.tanh(actions_K[:, :self.n_taskpace_pos])
        a_pos_scaled = a_pos * self._delta_K_pos + self._central_K_pos
        self.kp[:, :self.n_taskpace_pos, :self.n_taskpace_pos] = torch.diag_embed(a_pos_scaled)
        # rotation
        if self.learn_rotations:
            a_rot = torch.tanh(actions_K[:, self.n_taskpace_pos:])
            a_rot_scaled = a_rot * self._delta_K_rot + self._central_K_rot
            if self.n_taskpace_rot == 1:
                # 2d case
                self.kp[:, -1, -1] = a_rot_scaled.squeeze()
            else:
                self.kp[:, self.n_taskpace_pos:, self.n_taskpace_pos:] = torch.diag_embed(a_rot_scaled)

        # update critial damping
        self.kv = 2. * torch.sqrt(self.kp)

    def _refresh(self):
        # refresh tensors to get the latest state
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

    def clip_rotation(self, rot):
        # clip the rotation in the tangent space, such that the angle of rotation is capped
        rot_aa = rot.log_map()
        rot_aa_norm = torch.linalg.vector_norm(rot_aa + 1e-6, dim=-1, ord=np.inf)
        scale_ratio = torch.clip(rot_aa_norm, 0, self.delta_rotation_max) / rot_aa_norm
        rot_aa = scale_ratio.view(-1, 1) * rot_aa
        return theseus.SO3.exp_map(rot_aa)

    def compute_forces(self, error_taskspace, vel_taskspace):
        kp = self.kp
        kv = self.kv

        if self.n_taskpace_pos == 2 and self.n_taskpace_rot == 1:
            # adapt errors to 2D
            error_taskspace = torch.cat((error_taskspace[:, :2], error_taskspace[:, -1][:, None]), dim=-1)
            vel_taskspace = torch.cat((vel_taskspace[:, :2], vel_taskspace[:, -1][:, None]), dim=-1)
            # adapt the stiffness and damping matrices to 2D
            self.kp_reduced[:, :2, :2] = kp[:, :2, :2]
            self.kp_reduced[:, -1, -1] = kp[:, -1, -1]
            self.kv_reduced[:, :2, :2] = kv[:, :2, :2]
            self.kv_reduced[:, -1, -1] = kv[:, -1, -1]
            kp = self.kp_reduced
            kv = self.kv_reduced
            # reduce the jacobian to 2D
            J = torch.cat((self._jacobian_ee[:, 0:2, :], self._jacobian_ee[:, -1, :][:, None]), dim=-2)
        else:
            J = self._jacobian_ee

        if self.control_type == 'osc':
            # Operational Space Control
            mm_inv = torch.inverse(self._mm)
            m_eef_inv = J @ mm_inv @ J.transpose(-2, -1)
            m_eef = torch.inverse(m_eef_inv)
            forces = J.transpose(-2, -1) @ m_eef @ (
                    kp @ error_taskspace[..., None, ...] - kv @ vel_taskspace[..., None, ...])
        else:
            # Differentiable Inverse Kinematics
            J_pinv = torch.linalg.pinv(J)
            error_joints = J_pinv @ error_taskspace.unsqueeze(dim=-1)
            error_joints = error_joints.squeeze()
            joint_vel = self.dof_vel
            # PD controller at the joint level
            forces = kp @ error_joints[..., None, ...] - kv @ joint_vel[..., None, ...]

        return forces

    def pre_physics_step(self, actions, step_controller=0):
        self._refresh()

        # Get current rigid body peg poses and velocities
        peg_pos_cur = self.rb_state[self.rb_peg_idxs, :3]
        peg_rot_cur = theseus.SO3(quaternion=quat_xyzw_to_wxyz(self.rb_state[self.rb_peg_idxs, 3:7]))
        peg_lin_vel_cur = self.rb_state[self.rb_peg_idxs, 7:10]
        peg_ang_vel_cur = self.rb_state[self.rb_peg_idxs, 10:13]

        # decompose actions, and possibly expand to 3D
        self.actions_pos_delta[:, :self.n_taskpace_pos] = actions[:, :self.n_taskpace_pos]

        if self.learn_rotations:
            if self.n_taskpace_rot == 1:
                self.actions_rot_delta[:, -1] = actions[:, self.n_taskpace_pos]
            else:
                self.actions_rot_delta = actions[:, self.n_taskpace_pos:self.n_taskpace_pos+self.n_taskpace_rot]

        # -----------------------------------------------------
        # Position
        if self.control_vel:
            # Velocity control
            raise NotImplementedError
        else:
            # Position control
            if step_controller == 0:
                # update and fix the target at the beginning of the low-level control loop
                actions_pos_nominal = torch.zeros_like(self.actions_pos_delta)
                if self.enable_nominal_policy:
                    actions_pos_nominal_ = self.peg_pos_goal - peg_pos_cur  # vector pointing towards the goal
                    # clip the nominal signal, so that the actions are not too small in comparison to the nominal signal
                    # actions_pos_nominal = torch.clip(actions_pos_nominal_, self.delta_translation_min, self.delta_translation_max)
                    actions_pos_nominal = actions_pos_nominal_
                # combine the nominal and the learned policy
                actions_pos_ = actions_pos_nominal + self.actions_pos_delta
                # clip the delta in translation to make sure the target is within the bounds
                actions_pos = torch.clip(actions_pos_, self.delta_translation_min, self.delta_translation_max)
                self.peg_pos_target = (peg_pos_cur + actions_pos).clone()
            # update the error and clip it
            error_pos_ = self.peg_pos_target - peg_pos_cur
            error_pos = torch.clip(error_pos_, self.delta_translation_min, self.delta_translation_max)

        # -----------------------------------------------------
        # Orientation
        if self.control_vel:
            # Velocity control
            raise NotImplementedError
        else:
            # Position control
            if step_controller == 0:
                # update and fix the target at the beginning of the low-level control loop
                actions_rot_delta = theseus.SO3.exp_map(self.actions_rot_delta)
                if self.enable_nominal_policy or not self.learn_rotations:
                    # distance from current to the goal orientation
                    rot_nominal = self.peg_rot_goal.compose(peg_rot_cur.inverse())
                    # clip the nominal signal, so that the actions are not too small in comparison to the nominal signal
                    # rot_nominal = self.clip_rotation(rot_nominal)
                    # combine the nominal and the learned policy
                    actions_rot_delta = actions_rot_delta.compose(rot_nominal)
                # clip the delta in rotation to make sure the target is within the bounds
                self.peg_rot_target = self.clip_rotation(actions_rot_delta.compose(peg_rot_cur))
            # update the error and clip it
            actions_rot_ = self.peg_rot_target.compose(peg_rot_cur.inverse())
            actions_rot = self.clip_rotation(actions_rot_)
            # https://studywolf.wordpress.com/2018/12/03/force-control-of-task-space-orientation/ - method 3
            error_rot = 2. * actions_rot.log_map()

        # taskspace errors
        error_taskspace = torch.cat([error_pos, error_rot], dim=-1)

        # current velocities
        vel_taskspace = torch.cat([peg_lin_vel_cur, peg_ang_vel_cur], dim=-1)

        # Update the (possibly learned) stiffness and dampig matrices
        if self.learn_stiffness:
            self._update_stiffness_and_damping_matrices(actions[:, self.action_K_idxs])

        # compute the force/torques with the OSC or IK controller
        forces_ = self.compute_forces(error_taskspace, vel_taskspace)

        # apply forces/torques to the joints
        forces = gymtorch.unwrap_tensor(forces_.squeeze())
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

        # visualize the temporary target poses
        self.visualize()

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

        self.visualize()

    def visualize(self):
        # vis
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self._refresh()
            rb_peg_states = self.rb_state[self.rb_peg_idxs]
            peg_rot_target_quat = quat_wxyz_to_xyzw(self.peg_rot_target.to_quaternion())
            for i in range(self.num_envs):
                env = self.envs[i]

                # draw peg current state
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(rb_peg_states[i][0], rb_peg_states[i][1], rb_peg_states[i][2])
                pose.r = gymapi.Quat(rb_peg_states[i][3], rb_peg_states[i][4], rb_peg_states[i][5], rb_peg_states[i][6])

                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, pose)

                # draw peg target state
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(self.peg_pos_target[i][0], self.peg_pos_target[i][1], self.peg_pos_target[i][2])
                pose.r = gymapi.Quat(peg_rot_target_quat[i][0], peg_rot_target_quat[i][1], peg_rot_target_quat[i][2], peg_rot_target_quat[i][3])
                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, pose)
                gymutil.draw_lines(self.target_pose_geom, self.gym, self.viewer, env, pose)


#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def quat_wxyz_to_xyzw(quat_wxyz):
    quat_xyzw = torch.zeros_like(quat_wxyz)
    quat_xyzw[..., 3] = quat_wxyz[..., 0].clone()
    quat_xyzw[..., 0:3] = quat_wxyz[..., 1:4].clone()
    return quat_xyzw


@torch.jit.script
def quat_xyzw_to_wxyz(quat_xyzw):
    quat_wxyz = torch.zeros_like(quat_xyzw)
    quat_wxyz[..., 0] = quat_xyzw[..., 3].clone()
    quat_wxyz[..., 1:4] = quat_xyzw[..., 0:3].clone()
    return quat_wxyz


@torch.jit.script
def error_orientation(desired, current):
    """
    https://studywolf.wordpress.com/2018/12/03/force-control-of-task-space-orientation/
    """
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def compute_insertion_reward(
        peg_pos_goal, peg_rot_goal,
        rb_state, rb_peg_idxs, reset_buf, progress_buf, max_episode_length,
        enable_sparse_reward=False,
        weight_position=1., weight_orientation=5.,
        terminate_pos_error=0.005, terminate_angle_error=5.*math.pi/180.
):
    position_peg = rb_state[rb_peg_idxs][:, 0:3]
    orientation_peg = rb_state[rb_peg_idxs][:, 3:7]
    box_linear_velocity = rb_state[rb_peg_idxs][:, 7:10]
    box_angluar_velocity = rb_state[rb_peg_idxs][:, 10:13]

    # position
    peg_pos_dist = (peg_pos_goal - position_peg).pow(2)
    reward = -peg_pos_dist.sum(dim=-1) * weight_position

    # rotation
    peg_rot_delta = peg_rot_goal.compose(theseus.SO3(quaternion=quat_xyzw_to_wxyz(orientation_peg)).inverse())
    peg_rot_delta_aa = peg_rot_delta.log_map()
    peg_rot_delta_aa_angle = torch.linalg.vector_norm(peg_rot_delta_aa, dim=-1)
    peg_rot_dist = peg_rot_delta_aa.pow(2).sum(dim=-1)
    reward -= peg_rot_dist * weight_orientation

    # terminate condition
    condition = torch.logical_or(
        progress_buf >= max_episode_length - 1,  # max steps reached
        torch.logical_and(
            torch.all(peg_pos_dist < terminate_pos_error, dim=-1),  # position error
            peg_rot_delta_aa_angle < terminate_angle_error  # angle error
        )
    )

    reset = torch.where(condition, torch.ones_like(reset_buf), reset_buf)

    if enable_sparse_reward:
        reward = reward * 0 - 1

    #add_final_reward = torch.where(condition, torch.ones_like(reward)*10., torch.zeros_like(reward))
    #reward = reward + add_final_reward

    return reward, reset
