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

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from gym import spaces

import torch

from isaacgymenvs.tasks.base.vec_task import VecTask


class InsertionBox2D(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        assert cfg["env"]["recomputePrePhysicsStep"], "InsertionBox2D requires recomputePrePhysicsStep to be True"

        self.num_dof = 3  # 2 prismatic joints + 1 revolute joint

        self.cfg = cfg

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
        self.learn_orientations = self.cfg["env"]["learnOrientations"]
        self.learn_stiffness = self.cfg["env"]["learnStiffness"]
        self.just_learn_stiffness = self.cfg["env"].get("justLearnStiffness", False)
        if self.just_learn_stiffness:
            self.enable_nominal_policy = True
            self.learn_stiffness = True
        if self.learn_stiffness:
            self._K_pos_min = self.cfg["env"]["K_pos_min"]
            self._K_pos_max = self.cfg["env"]["K_pos_max"]
            self._K_orn_min = self.cfg["env"]["K_orn_min"]
            self._K_orn_max = self.cfg["env"]["K_orn_max"]
            self._delta_K_pos = .5 * (self._K_pos_max - self._K_pos_min)
            self._central_K_pos = .5 * (self._K_pos_max + self._K_pos_min)
            self._delta_K_orn = .5 * (self._K_orn_max - self._K_orn_min)
            self._central_K_orn = .5 * (self._K_orn_max + self._K_orn_min)

        self.enable_sparse_reward = self.cfg["env"]["enableSparseReward"]

        ############################
        # observations (and state)
        # All relative to the world frame
        self.observe_orientations = self.cfg["env"].get("observeOrientations", True)
        self.enable_orientations = self.cfg["env"]["enableOrientations"]
        self.observe_velocities = self.cfg['env']['enableVelocityState']
        self.observe_force = self.cfg["env"].get("observeForce", False)

        if not self.observe_orientations:
            # Without rotations
            # 0:3 - box position
            # 3:6 - box linear velocity
            if not self.observe_velocities:
                self.cfg["env"]["numObservations"] = 2
            else:
                self.cfg["env"]["numObservations"] = 2 + 2
        else:
            # With rotations
            # 0:3 - box position
            # 3:7 - box orientation
            # 7:10 - box linear velocity
            # 10:13 - box angular velocity
            if not self.observe_velocities:
                self.cfg["env"]["numObservations"] = 2 + 2
            else:
                self.cfg["env"]["numObservations"] = 2 + 2 + 2 + 2
        if self.observe_force:
            self.cfg["env"]["numObservations"] += 2

        ############################
        # actions
        # action is the desired velocity on the 2d space (2 prismatic (x and y) + 1 revolute (rotation around-axis))
        self.cfg["env"]["numActions"] = 2
        extra_actions = 0
        if not self.learn_orientations:
            if self.learn_stiffness:
                extra_actions += 2  # parameters of Kpos
        else:
            extra_actions += 1  # orientation
            if self.learn_stiffness:
                extra_actions += 2 + 1  # parameters of Kpos and Korn
        self.cfg["env"]["numActions"] += extra_actions

        if self.just_learn_stiffness:
            self.cfg["env"]["numActions"] = 3

        ############################
        # low-level controller
        self.enable_nominal_policy = self.cfg["env"].get("enableNonimalPolicy", False)
        self.use_osc = self.cfg["env"].get("useOSC", False)  # if False, use differentiable inverse kinematics
        self.control_vel = self.cfg["env"]["controlVelocity"]

        self.linear_velocity_norm_min = self.cfg["env"].get("linear_velocity_norm_min", 0.)
        self.linear_velocity_norm_max = self.cfg["env"].get("linear_velocity_norm_max", 1.)
        self.angular_velocity_norm_min = self.cfg["env"].get("angular_velocity_norm_min", 0.)
        self.angular_velocity_norm_max = self.cfg["env"].get("angular_velocity_norm_max", 1.)

        #######################################################################################################
        # Initialize vectorized task
        super().__init__(
            config=self.cfg,
            rl_device=rl_device, sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless, virtual_screen_capture=virtual_screen_capture,
            force_render=force_render)

        self._num_envs_array = np.arange(self.num_envs)

        # buffer for dof state
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # buffer for rigid body state
        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state = gymtorch.wrap_tensor(rb_state_tensor)

        # buffer for jacobian of peg
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "peg")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        self._jacobian_ee_tensor = jacobian[:, self.peg_ee_index-1, :, :]

        # In 2D, we only have 3 dofs, corresponding to 2 prismatic joints (x and y) and 1 revolute joint (around z)
        # prealocate the jacobian for OSC computations
        self._jacobian_ee = torch.zeros(size=(self.num_envs, 3, 3), device=self.device)

        # buffer for mass matrix of peg
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "peg")
        self._mm = gymtorch.wrap_tensor(_massmatrix)

        if self.observe_force:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            sensors_per_env = 1
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

            _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
            self.contact_sensor_tensor = gymtorch.wrap_tensor(_net_cf)

            #dof_sensor_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            #self.dof_forces = gymtorch.wrap_tensor(dof_sensor_tensor).view(self.num_envs, 3)
            #self.first = True

        ############################
        # Default impedance control stiffness and damping
        # These values should be adapted if used in OSC or differentiable IK
        # 2 prismatic joints + 1 revolute joint
        self.kp = torch.zeros((self.num_envs, 3, 3), device=self.device)
        self.kp_pos_factor = 50.
        self.kp[:, :3, :3] = self.kp_pos_factor * torch.eye(3).reshape((1, 3, 3)).repeat(self.num_envs, 1, 1)
        if self.enable_orientations:
            self.kp_orn_factor = 30.
            self.kp[:, 2, 2] = self.kp_orn_factor
        else:
            # if no orientation, 0 gain for revolute joint
            self.kp[:, 2, 2] = 0

        self.kv = torch.zeros((self.num_envs, 3, 3), device=self.device)
        self.kv_pos_factor = 2 * np.sqrt(self.kp_pos_factor)
        self.kv[:, :3, :3] = self.kv_pos_factor * torch.eye(3).reshape((1, 3, 3)).repeat(self.num_envs, 1, 1)
        if self.enable_orientations:
            self.kv_orn_factor = 2 * np.sqrt(self.kp_orn_factor)
            self.kv[:, 2, 2] = self.kv_orn_factor
        else:
            # if no orientation, 0 gain for revolute joint
            self.kv[:, 2, 2] = 0

        ############################
        # Initial states
        self.initial_position_bounds = torch.tensor(
            self.cfg["env"].get("initialPositionBounds", [[-1, -1], [1, 1]]),
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
            if not self.enable_orientations:  # joint lower and upper limits are zero
                dof_props['lower'][2] = 0.0
                dof_props['upper'][2] = 0.0

            self.gym.set_actor_dof_properties(env_ptr, handle_peg, dof_props)

            props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle_peg)
            for prop in props:
                # https://www.researchgate.net/publication/330003074_Wear_and_coefficient_of_friction_of_PLA_-_Graphite_composite_in_3D_printing_technology
                prop.friction = 0.4
            self.gym.set_actor_rigid_shape_properties(env_ptr, handle_peg, props)

            # index of peg end-effector rigid body
            rb_idx = self.gym.find_actor_rigid_body_index(env_ptr, handle_peg, "ee", gymapi.DOMAIN_SIM)
            self.rb_peg_idxs.append(rb_idx)

        self.peg_ee_index = self.gym.get_asset_rigid_body_dict(asset_peg)["ee"]
        self.actor_peg_idxs = to_torch(self.actor_peg_idxs, dtype=torch.long, device=self.device)

    def reset_idx(self, env_ids):
        if self.use_initial_states:
            positions = self.initial_states[torch.randint(self.initial_states.shape[0], (len(env_ids),)), :]
            positions = torch.hstack((positions, torch.zeros((len(env_ids), 1), device=self.device)))
        else:
            positions = torch.zeros((len(env_ids), self.num_dof), device=self.device)
            initial_position_min, initial_position_max = self.initial_position_bounds
            positions[..., :2] = (initial_position_min[None] +
                                  (initial_position_max - initial_position_min)[None]
                                  * torch.rand((len(env_ids), 2), device=self.device))

        if self.enable_orientations:
            rot_lim = np.pi/2
            rot_min = -rot_lim
            rot_max = rot_lim
            positions[:, 2] = rot_min + (rot_max - rot_min) * torch.rand(len(env_ids), device=self.device)

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
        box_positions = self.rb_state[self.rb_peg_idxs][:, 0:3]
        box_orientation = self.rb_state[self.rb_peg_idxs][:, 3:7]
        box_linear_velocity = self.rb_state[self.rb_peg_idxs][:, 7:10]
        box_angluar_velocity = self.rb_state[self.rb_peg_idxs][:, 10:13]

        # convert to 2-dimensional observations
        self.obs_buf[env_ids, 0:2] = box_positions[:, 0:2]
        if not self.observe_orientations:
            if self.observe_velocities:
                self.obs_buf[env_ids, 2:4] = box_linear_velocity[:, 0:2]
        else:
            # get the angle around z-axis, which is yaw
            roll, pitch, yaw = get_euler_xyz(box_orientation)

            # convert angle to polar coordinates
            cos_theta = torch.cos(yaw)
            sin_theta = torch.sin(yaw)
            theta = torch.cat((cos_theta.unsqueeze(1), sin_theta.unsqueeze(1)), dim=1)

            self.obs_buf[env_ids, 2:4] = theta[env_ids, :]  # box z-axis orientation in polar coordinates
            if self.observe_velocities:
                self.obs_buf[env_ids, 4:6] = box_linear_velocity[:, 0:2]  # box linear velocity
                self.obs_buf[env_ids, 6:8] = box_angluar_velocity[:, 0:2]  # box angular velocity

        if self.observe_force:
            #self.gym.refresh_force_sensor_tensor(self.sim)
            #vec_force = torch.zeros_like(self.obs_buf[..., -2:])
            #vec_force[..., 0:2] = self.vec_sensor_tensor[..., 0:2]
            #self.obs_buf[env_ids, -2:] = vec_force[env_ids]

            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.obs_buf[env_ids, -2:] = self.contact_sensor_tensor[self.rb_peg_idxs][env_ids, 0:2] / 50.

        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_insertion_reward_2d(
            self.rb_state, torch.tensor(self.rb_peg_idxs, device=self.device),
            self.reset_buf, self.progress_buf, self.max_episode_length,
            enable_velocities_states=self.observe_velocities,
            enable_orientations=self.enable_orientations,
            enable_sparse_reward=self.enable_sparse_reward,
            reward_orientations=self.observe_orientations
        )

    def _create_stiffness_and_damping_matrices(self, actions_K):
        """
        actions_K should be [K_pos_x, K_pos_y, K_orn]
        """
        # scale values from actions to defined range
        a_pos = torch.tanh(actions_K[..., 0:2])
        a_pos_scaled = a_pos * self._delta_K_pos + self._central_K_pos
        a_orn = torch.tanh(actions_K[..., 2])
        a_orn_scaled = a_orn * self._delta_K_orn + self._central_K_orn

        #print(a_pos_scaled[0], torch.sqrt(a_pos_scaled)[0])
        #print(a_orn_scaled[0], torch.sqrt(a_orn_scaled)[0])

        # calculate stiffness and damping
        K_pos_x = a_pos_scaled[..., 0]
        K_pos_y = a_pos_scaled[..., 1]
        K_orn = a_orn_scaled

        #print(K_pos_x[0], K_pos_y[0], K_orn[0])

        kp = torch.eye(3, device=self.device).reshape((1, 3, 3)).repeat(self.num_envs, 1, 1)
        kp[..., 0, 0] = K_pos_x
        kp[..., 1, 1] = K_pos_y
        kp[..., 2, 2] = K_orn

        kv = torch.eye(3, device=self.device).reshape((1, 3, 3)).repeat(self.num_envs, 1, 1)
        if self.use_osc:
            kv[..., 0, 0] = 2 * torch.sqrt(K_pos_x)
            kv[..., 1, 1] = 2 * torch.sqrt(K_pos_y)
            kv[..., 2, 2] = 2 * torch.sqrt(K_orn)  # mass is canceled in osc
        else:
            # the factors come from the mass matrix and need to be applied to get a stable system
            kv[..., 0, 0] = 2 * torch.sqrt(K_pos_x * 1)
            kv[..., 1, 1] = 2 * torch.sqrt(K_pos_y * 1)
            kv[..., 2, 2] = 2 * torch.sqrt(K_orn * 0.01)

        return kp, kv

    def _refresh(self):
        # refresh tensors to get the latest state
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

    def pre_physics_step(self, actions, step=0):
        self._refresh()

        # Set (possibly learned) stiffness and damping matrices
        if self.learn_stiffness:
            if self.just_learn_stiffness:
                kp, kv = self._create_stiffness_and_damping_matrices(actions.clone())
            else:
                kp, kv = self._create_stiffness_and_damping_matrices(actions[..., 3:6].clone())
        else:
            kp = self.kp
            kv = self.kv

        # Get current box poses and velocities
        box_pos_cur = self.rb_state[self.rb_peg_idxs, :3]
        box_orn_cur = self.rb_state[self.rb_peg_idxs, 3:7]
        box_lin_vel_cur = self.rb_state[self.rb_peg_idxs, 7:10]
        box_ang_vel_cur = self.rb_state[self.rb_peg_idxs, 10:13]

        # positions / linear velocities
        if not self.control_vel:
            # position control
            vector_to_goal_pos = -box_pos_cur[..., :2]
            # clip the velocity vector
            velocity_norm = torch.linalg.vector_norm(vector_to_goal_pos + 1e-6, dim=1, ord=np.inf)
            scale_ratio = torch.clip(velocity_norm, 0., self.linear_velocity_norm_max) / velocity_norm
            # add policy to nominal PD signal
            pos_err = scale_ratio.view(-1, 1) * vector_to_goal_pos
            # if not self.just_learn_stiffness:
            #     pos_err += actions[..., :2]

            # clip linear velocity by norm
            velocity_norm = torch.linalg.vector_norm(pos_err[..., :2] + 1e-6, dim=1, ord=np.inf)
            scale_ratio = torch.clip(
                velocity_norm, self.linear_velocity_norm_min, self.linear_velocity_norm_max) / velocity_norm
            pos_err[:, :2] = scale_ratio.view(-1, 1) * pos_err[:, :2]
        else:
            # velocity control
            if self.enable_nominal_policy:
                signal_to_goal_pos = -box_pos_cur[..., :2]
                #clip first the PD signal
                velocity_norm = torch.linalg.vector_norm(signal_to_goal_pos + 1e-6, dim=1, ord=np.inf)
                scale_ratio = torch.clip(velocity_norm, 0., self.linear_velocity_norm_max) / velocity_norm
                # add PD to signal from policy
                if self.just_learn_stiffness:
                    pos_err = scale_ratio.view(-1, 1) * signal_to_goal_pos
                else:
                    pos_err = actions[:, :2] + scale_ratio.view(-1,1) * signal_to_goal_pos

                # clip linear velocity by norm
                velocity_norm = torch.linalg.vector_norm(pos_err[:, :2] + 1e-6, dim=1, ord=np.inf)
                scale_ratio = torch.clip(velocity_norm, self.linear_velocity_norm_min,
                                         self.linear_velocity_norm_max) / velocity_norm
                pos_err[:, :2] = scale_ratio.view(-1, 1) * pos_err[:, :2]
            else:
                # clip linear velocity by norm
                velocity_norm = torch.linalg.vector_norm(actions[:, :2] + 1e-6, dim=1, ord=np.inf)
                scale_ratio = torch.clip(velocity_norm, self.linear_velocity_norm_min, self.linear_velocity_norm_max) / velocity_norm
                actions[:, :2] = scale_ratio.view(-1, 1) * actions[:, :2]

                pos_err = actions[:, :2]


        # orientations / angular velocities
        if not self.enable_orientations:
            # position control
            raise NotImplementedError
        else:
            # velocity control
            if not self.learn_orientations:
                # Orientation is not part of the policy output. The desired orientation is set to the final one.
                # control the angle
                orn_err = torch.zeros_like(box_ang_vel_cur)

                box_orn_des = torch.zeros_like(box_orn_cur)
                box_orn_des[..., 3] = 1.  # no rotation wrt the base

                # clip signal from PD
                signal_to_goal_orn = error_orientation(box_orn_des, box_orn_cur)
                velocity_norm = torch.abs(signal_to_goal_orn[:, 2] + 1e-6)
                scale_ratio = torch.clip(velocity_norm, 0.,
                                         self.angular_velocity_norm_max) / velocity_norm
                orn_err[:, 2] = scale_ratio * signal_to_goal_orn[:, 2]
            else:
                if not self.control_vel:
                    raise NotImplementedError
                else:
                    orn_err = torch.zeros_like(box_ang_vel_cur)

                    if self.enable_PD_to_goal:
                        box_orn_des = torch.zeros_like(box_orn_cur)
                        box_orn_des[..., 3] = 1.  # no rotation wrt the base

                        # clip signal from PD
                        signal_to_goal_orn = error_orientation(box_orn_des, box_orn_cur)
                        velocity_norm = torch.abs(signal_to_goal_orn[:, 2] + 1e-6)
                        scale_ratio = torch.clip(velocity_norm, 0.,
                                                 self.angular_velocity_norm_max) / velocity_norm
                        if self.just_learn_stiffness:
                            orn_err[:, 2] = scale_ratio * signal_to_goal_orn[:, 2]
                        else:
                            orn_err[:, 2] = actions[:, 2] + scale_ratio * signal_to_goal_orn[:, 2]

                        # clip angular velocity by norm
                        velocity_norm = torch.abs(orn_err[:, 2] + 1e-6)
                        scale_ratio = torch.clip(velocity_norm, self.angular_velocity_norm_min,
                                                 self.angular_velocity_norm_max) / velocity_norm
                        orn_err[:, 2] = scale_ratio * orn_err[:, 2]
                    else:
                        # clip angular velocity by norm
                        velocity_norm = torch.abs(actions[:, 2] + 1e-6)
                        scale_ratio = torch.clip(velocity_norm, self.angular_velocity_norm_min,
                                                 self.angular_velocity_norm_max) / velocity_norm
                        actions[:, 2] = scale_ratio * actions[:, 2]

                        orn_err[..., 2] = actions[:, 2] # angular velocity around z-axis

            #error_norm = torch.abs(orn_err[:, 2] + 1e-6)
            #scale_ratio = torch.clip(error_norm, 0., 0.1) / error_norm
            #orn_err[:, 2] = scale_ratio * orn_err[:, 2]

        ### concat error ###
        error_taskspace = torch.cat([pos_err[:, :2], orn_err[:, -1].view(-1, 1)], dim=1)

        ### get current velocities ###
        task_vel = torch.cat([box_lin_vel_cur[:, :2], box_ang_vel_cur[:, -1].view(-1, 1)], dim=1)

        ### Jacobian ###
        self._jacobian_ee[:, 0:2, :] = self._jacobian_ee_tensor[..., 0:2, :]
        self._jacobian_ee[:, 2, :] = self._jacobian_ee_tensor[..., -1, :]
        J = self._jacobian_ee

        if self.use_osc:
            # OSC
            mm_inv = torch.inverse(self._mm)
            m_eef_inv = J @ mm_inv @ J.transpose(-2, -1)
            m_eef = torch.inverse(m_eef_inv)

            actions_dof_tensor = J.transpose(-2, -1) @ m_eef @ (
                    kp @ error_taskspace[..., None, ...] - kv @ task_vel[..., None, ...])
        else:
            # Differentiable Inverse Kinematics
            J_pinv = torch.linalg.pinv(J)
            error_joints = J_pinv @ error_taskspace.unsqueeze(dim=-1)
            error_joints = error_joints.squeeze()

            joint_vel = self.dof_vel
            # PD controller at the joint level
            actions_dof_tensor = kp @ error_joints[..., None, ...] - kv @ joint_vel[..., None, ...]

        #actions_dof_tensor *= 0
        #if self.first:
        #    actions_dof_tensor[..., 2,0] = -1.
        #    self.first = False

        forces = gymtorch.unwrap_tensor(actions_dof_tensor.squeeze())
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

        #self.gym.refresh_force_sensor_tensor(self.sim)
        #forces = self.vec_sensor_tensor.clone()[0]
        #forces[torch.logical_and(forces >= 0, forces <= 1e-6)] = 0.
        #forces[torch.logical_and(forces < 0, forces >= -1e-6)] = 0.
        #print("force", forces* 100)

        #self.gym.refresh_dof_force_tensor(self.sim)
        #dof_forces = self.vec_sensor_tensor.clone()[0]
        #dof_forces[torch.logical_and(dof_forces >= 0, dof_forces <= 1e-6)] = 0.
        #dof_forces[torch.logical_and(dof_forces < 0, dof_forces >= -1e-6)] = 0.
        #print("dof_force", dof_forces * 100)

        #print("----")

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

        # vis
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)

            self._refresh()
            box_rb_states = self.rb_state[self.rb_peg_idxs]

            for i in range(self.num_envs):
                env = self.envs[i]

                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(box_rb_states[i][0], box_rb_states[i][1], box_rb_states[i][2])
                pose.r = gymapi.Quat(box_rb_states[i][3], box_rb_states[i][4], box_rb_states[i][5], box_rb_states[i][6])

                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, pose)


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def error_orientation(desired, current):
    """
    https://studywolf.wordpress.com/2018/12/03/force-control-of-task-space-orientation/
    """
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


@torch.jit.script
def compute_insertion_reward_2d(
        rb_state, rb_peg_idxs, reset_buf, progress_buf, max_episode_length,
        enable_velocities_states=False, enable_orientations=False, enable_sparse_reward=False, reward_orientations=False,
        weight_position=1., weight_orientation=5.
):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, bool, bool, bool, bool, float, float) -> Tuple[torch.Tensor, torch.Tensor]

    position_peg = rb_state[rb_peg_idxs][:, 0:3]
    orientation_peg = rb_state[rb_peg_idxs][:, 3:7]
    box_linear_velocity = rb_state[rb_peg_idxs][:, 7:10]
    box_angluar_velocity = rb_state[rb_peg_idxs][:, 10:13]

    # 2D distance
    peg_dist_pos = torch.sqrt(position_peg[..., 0]**2 + position_peg[..., 1]**2)
    reward = -peg_dist_pos * weight_position

    peg_dist_orn = torch.zeros_like(peg_dist_pos)
    if enable_orientations:
        box_orn_des = torch.zeros_like(orientation_peg)
        box_orn_des[..., 3] = 1.  # no rotation wrt the base. quaternion (0, 0, 0, 1)
        err_orn = error_orientation(box_orn_des, orientation_peg)
        peg_dist_orn = torch.sqrt(err_orn[..., 2]**2)
    if reward_orientations:
        reward -= peg_dist_orn * weight_orientation

    condition = torch.logical_or(progress_buf >= max_episode_length - 1, peg_dist_pos < 0.05)
    if enable_orientations:
        condition = torch.logical_or(
            progress_buf >= max_episode_length - 1, torch.logical_and(peg_dist_pos < 0.05, peg_dist_orn < 0.1))

    reset = torch.where(condition, torch.ones_like(reset_buf), reset_buf)

    if enable_sparse_reward:
        reward = reward * 0 - 1

    #add_final_reward = torch.where(condition, torch.ones_like(reward)*10., torch.zeros_like(reward))
    #reward = reward + add_final_reward

    return reward, reset
