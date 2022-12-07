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
from tasks.base.vec_task import VecTask
from gym import spaces
import pytorch_kinematics as pk

import torch


@torch.jit.script
def orientation_error(desired, current):
    """
    https://studywolf.wordpress.com/2018/12/03/force-control-of-task-space-orientation/
    """
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)



class Box2DInsertion(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.use_osc = False
        self.cfg = cfg

        self.enable_PD_to_goal = self.cfg["env"].get("enable_PD_to_goal", False)
        self.observe_force = self.cfg["env"].get("observeForce", False)

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.controller_freq = self.cfg["env"].get("controller_freq", None)
        self.recompute_prephysics_step = self.cfg["env"].get("recomputePrePhysicsStep", False)

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        #self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.minimum_linear_velocity_norm = self.cfg["env"].get("minimum_linear_velocity_norm", 0.)
        self.maximum_linear_velocity_norm = self.cfg["env"].get("maximum_linear_velocity_norm", 10.)
        self.minimum_angular_velocity_norm = self.cfg["env"].get("minimum_angular_velocity_norm", 0.)
        self.maximum_angular_velocity_norm = self.cfg["env"].get("maximum_angular_velocity_norm", 1.)

        self.observe_velocities = self.cfg['env']['enableVelocityState']
        self.observe_orientations = self.cfg["env"]["learnOrientations"]

        self.enable_orientations = self.cfg["env"]["enableOrientations"]
        self.enable_ic = self.cfg["env"]["enableIC"]

        self.control_vel = self.cfg["env"]["controlVelocity"]
        if not self.control_vel:
            raise NotImplementedError

        self.learn_stiffness = self.cfg["env"]["learnStiffness"]
        if self.learn_stiffness:
            self._K_pos_min = self.cfg["env"]["K_pos_min"]
            self._K_pos_max = self.cfg["env"]["K_pos_max"]
            self._K_orn_min = self.cfg["env"]["K_orn_min"]
            self._K_orn_max = self.cfg["env"]["K_orn_max"]
            self._delta_K_pos = .5 * (self._K_pos_max - self._K_pos_min)
            self._central_K_pos = .5 * (self._K_pos_max + self._K_pos_min)
            self._delta_K_orn = .5 * (self._K_orn_max - self._K_orn_min)
            self._central_K_orn = .5 * (self._K_orn_max + self._K_orn_min)

        # Observations
        # All relative to the world
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

        # Action is the desired velocity on the 3 joints representing the dofs (2 prismatic + 1 revolute)
        extra_actions = 0
        if not self.observe_orientations:
            if self.learn_stiffness:
                extra_actions += 2  # parameters of Kpos
            self.cfg["env"]["numActions"] = 2 + extra_actions
        else:
            extra_actions += 1
            if self.learn_stiffness:
                extra_actions += 2 + 1  # parameters of Kpos and Korn
            self.cfg["env"]["numActions"] = 2 + extra_actions

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state = gymtorch.wrap_tensor(rb_state_tensor)

        if self.observe_force:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            sensors_per_env = 1
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

            #dof_sensor_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            #self.dof_forces = gymtorch.wrap_tensor(dof_sensor_tensor).view(self.num_envs, 3)
            #self.first = True

        # vis
        self.axes_geom = gymutil.AxesGeometry(0.3)

        # Default IC stiffness and damping
        # 2 prismatic joints + 1 revolute joint
        self.kp = torch.zeros((self.num_envs, 3, 3), device=self.device)
        self.kp_pos_factor = 100.
        self.kp[:, :3, :3] = self.kp_pos_factor * torch.eye(3).reshape((1, 3, 3)).repeat(self.num_envs, 1, 1)
        if self.enable_orientations:
            self.kp_orn_factor = 25.
            self.kp[:, 2, 2] = self.kp_orn_factor
        else:
            # if no orientation, 0 gain for revolute joint
            self.kp[:, 2, 2] = 0

        self.enable_damping_term = self.cfg["env"]["enableDampingTerm"]
        if self.enable_damping_term:
            self.kv = torch.zeros((self.num_envs, 3, 3), device=self.device)
            self.kv_pos_factor = 2 * np.sqrt(self.kp_pos_factor)
            self.kv[:, :3, :3] = self.kv_pos_factor * torch.eye(3).reshape((1, 3, 3)).repeat(self.num_envs, 1, 1)
            if self.enable_orientations:
                self.kv_orn_factor = 2 * np.sqrt(self.kp_orn_factor*0.01)
                self.kv[:, 2, 2] = self.kv_orn_factor
            else:
                # if no orientation, 0 gain for revolute joint
                self.kv[:, 2, 2] = 0
        else:
            self.kv = torch.zeros_like(self.kp)


        self.enable_sparse_reward = self.cfg["env"]["enableSparseReward"]
        self.initial_position_bounds = self.cfg["env"].get("initialPositionBounds", [[-1, -1], [1, 1]])

        self.use_init_states = self.cfg["env"].get('useInitStates', 'False')
        if self.use_init_states:
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


    def create_sim(self):
        self.dt = self.sim_params.dt

        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        self._create_robot_model()

        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "box")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :3, :3]

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        plane_params.distance = 0.01
        plane_params.dynamic_friction = 0.
        plane_params.static_friction = 0.
        plane_params.restitution = 0.
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(-spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Platform asset
        if "asset" in self.cfg["env"]:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            asset_root = project_root + self.cfg["env"]["asset"]["assetRoot"]
            asset_file = self.cfg["env"]["asset"]["assetFileName"]
        else:
            raise KeyError

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.armature = 0.01
        asset_options.disable_gravity = True
        box2d_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(box2d_asset)

        # default pose
        box_size = 0.05
        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p = gymapi.Vec3(0., 0., box_size / 2)
            pose.r = gymapi.Quat(0, 0, 0, 1)
        else:
            pose.p = gymapi.Vec3(0., box_size / 2, 0.)
            pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

        if self.observe_force:
            # create force sensors attached to the EEF
            eef_index = self.gym.find_asset_rigid_body_index(box2d_asset, "box")
            sensor_pose = gymapi.Transform()
            self.gym.create_asset_force_sensor(box2d_asset, eef_index, sensor_pose)

        self.envs = []
        self.box2d_handles = []
        self.box_rb_idxs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            self.envs.append(env_ptr)

            box2d_handle = self.gym.create_actor(env_ptr, box2d_asset, pose, "box", i, 0, 0)
            #self.gym.enable_actor_dof_force_sensors(env_ptr, box2d_handle)

            # Set properties for the 2 prismatic and 1 revolute joint
            dof_props = self.gym.get_actor_dof_properties(env_ptr, box2d_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][2] = gymapi.DOF_MODE_EFFORT
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            if not self.enable_orientations:
                dof_props['lower'][2] = 0.0
                dof_props['upper'][2] = 0.0

            self.gym.set_actor_dof_properties(env_ptr, box2d_handle, dof_props)
            self.box2d_handles.append(box2d_handle)

            props = self.gym.get_actor_rigid_shape_properties(env_ptr, box2d_handle)
            for prop in props:
                prop.friction = 0.4  # see https://www.researchgate.net/publication/330003074_Wear_and_coefficient_of_friction_of_PLA_-_Graphite_composite_in_3D_printing_technology
            self.gym.set_actor_rigid_shape_properties(env_ptr, box2d_handle, props)#

            # index of box rigid body
            rb_idx = self.gym.find_actor_rigid_body_index(env_ptr, box2d_handle, 'box', gymapi.DOMAIN_SIM)
            self.box_rb_idxs.append(rb_idx)

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 3-dimensional data
        box_positions = self.rb_state[self.box_rb_idxs][:, 0:3]
        box_orientation = self.rb_state[self.box_rb_idxs][:, 3:7]
        box_linear_velocity = self.rb_state[self.box_rb_idxs][:, 7:10]
        box_angluar_velocity = self.rb_state[self.box_rb_idxs][:, 10:13]

        # convert to 2-dimensional observations
        self.obs_buf[env_ids, 0:2] = box_positions[:, 0:2]
        if not self.observe_orientations:
            if self.observe_velocities:
                self.obs_buf[env_ids, 2:4] = box_linear_velocity[:, 0:2]
        else:
            # get angle around z-axis which is yaw
            roll, pitch, yaw = get_euler_xyz(box_orientation)

            # convert angle to polar coordinates
            cos_theta = torch.cos(yaw)
            sin_theta = torch.sin(yaw)
            theta = torch.cat((cos_theta.unsqueeze(1), sin_theta.unsqueeze(1)), dim=1)

            self.obs_buf[env_ids, 2:4] = theta[env_ids, :]  # box z-axis orientation in polar coordinates
            if self.observe_velocities:
                self.obs_buf[env_ids, 4:6] = box_linear_velocity[:, 0:2] # box linear velocity
                self.obs_buf[env_ids, 6:8] = box_angluar_velocity[:, 0:2]  # box angular velocity

        if self.observe_force:
            self.gym.refresh_force_sensor_tensor(self.sim)
            vec_force = torch.zeros_like(self.obs_buf[..., -2:])
            vec_force[..., 0:2] = self.vec_sensor_tensor[..., 0:2]
            self.obs_buf[env_ids, -2:] = vec_force[env_ids]
            #print(self.vec_sensor_tensor[0])

        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_box2d_insertion_reward(
            self.rb_state, torch.tensor(self.box_rb_idxs, device=self.device),
            self.reset_buf, self.progress_buf, self.max_episode_length,
            enable_velocities_states=self.observe_velocities,
            enable_orientations=self.enable_orientations, enable_sparse_reward=self.enable_sparse_reward, reward_orientations=self.observe_orientations
        )

    def reset_idx(self, env_ids):
        if self.use_init_states:
            positions = self.initial_states[torch.randint(self.initial_states.shape[0],(len(env_ids),)),:]
            positions = torch.hstack((positions, torch.zeros((len(env_ids), 1), device=self.device)))
        else:
            positions = torch.zeros((len(env_ids), self.num_dof), device=self.device)
            min_initial_position, max_initial_position = self.initial_position_bounds

            positions[:, 0] = (min_initial_position[0] - max_initial_position[0]) * torch.rand(len(env_ids), device=self.device) + max_initial_position[0]
            positions[:, 1] = (min_initial_position[1] - max_initial_position[1]) * torch.rand(len(env_ids), device=self.device) + max_initial_position[1]

            bad_locations_x = torch.where(torch.less(torch.abs(positions[:, 0]), 0.16), True, False)
            bad_locations_y = torch.where(torch.logical_and(torch.greater(positions[:, 1], -0.11), torch.less(positions[:, 1], 0.06)), True, False)
            bad_locations = torch.where(torch.logical_and(bad_locations_x, bad_locations_y), True, False)
            while bad_locations.any():
                num_samples = torch.count_nonzero(bad_locations).item()
                new_samples_x = (min_initial_position[0] - max_initial_position[0]) * torch.rand(num_samples,
                                                                                                   device=self.device) + \
                                  max_initial_position[0]
                new_samples_y = (min_initial_position[1] - max_initial_position[1]) * torch.rand(num_samples,
                                                                                                   device=self.device) + \
                                  max_initial_position[1]

                positions[:,0:2][torch.where(bad_locations)] = torch.cat((new_samples_x.unsqueeze(1), new_samples_y.unsqueeze(1)), dim=1)
                bad_locations_x = torch.where(torch.less(torch.abs(positions[:, 0]), 0.16), True, False)
                bad_locations_y = torch.where(
                    torch.logical_and(torch.greater(positions[:, 1], -0.11), torch.less(positions[:, 1], 0.06)), True,
                    False)
                bad_locations = torch.where(torch.logical_and(bad_locations_x, bad_locations_y), True, False)

        if self.enable_orientations:
            rot_lim = np.pi/2
            positions[:, 2] = (-rot_lim - rot_lim) * torch.rand(len(env_ids), device=self.device) + rot_lim

        velocities = torch.zeros((len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids, :] = positions
        self.dof_vel[env_ids, :] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        #TODO remove? see https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/issues/32
        self.gym.simulate(self.sim)

    def reset(self):
        """
        Overwrite since super class reset method does reset to (0,0) and this is called initally. We do not want to start in (0,0)
        """
        env_ids = torch.tensor(list(range(self.num_envs)), device=self.device)
        self.post_physics_step()

        return super().reset()

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

    def _create_robot_model(self):
        if "asset" in self.cfg["env"]:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            asset_root = project_root + self.cfg["env"]["asset"]["assetRoot"]
            asset_file = self.cfg["env"]["asset"]["assetFileName"]
        else:
            raise KeyError
        ee_name = "box"

        # load robot description from URDF and specify end effector link
        self.robot = pk.build_serial_chain_from_urdf(open(asset_root+"/"+asset_file).read(), ee_name, root_link_name="base")
        self.robot.to(device=self.device)

    def pre_physics_step(self, actions, step=0):
        actions_dof_tensor = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float)

        if not self.enable_ic:
            # Direct control
            if not self.observe_orientations:
                actions_dof_tensor[:, :2] = actions.to(self.device).squeeze()
            else:
                actions_dof_tensor = actions.to(self.device).squeeze()
        else:
            # refresh tensors
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_mass_matrix_tensors(self.sim)

            # Set (possibly learned) stiffness and damping matrices

            if self.learn_stiffness:
                kp, kv = self._create_stiffness_and_damping_matrices(actions[..., 3:6].clone())
            else:
                kp = self.kp
                kv = self.kv

            # Get current box poses and velocities
            box_pos_cur = self.rb_state[self.box_rb_idxs, :3]
            box_orn_cur = self.rb_state[self.box_rb_idxs, 3:7]
            box_lin_vel_cur = self.rb_state[self.box_rb_idxs, 7:10]
            box_ang_vel_cur = self.rb_state[self.box_rb_idxs, 10:13]

            # positions / linear velocities
            if not self.control_vel:
                raise NotImplementedError
            else:
                if self.enable_PD_to_goal:
                    signal_to_goal_pos = -box_pos_cur[:, :2]
                    #clip first the PD signal
                    velocity_norm = torch.linalg.vector_norm(signal_to_goal_pos + 1e-6, dim=1, ord=np.inf)
                    scale_ratio = torch.clip(velocity_norm, 0., 10.) / velocity_norm
                    # add PD to signal from policy
                    #pos_err = actions[:, :2] + scale_ratio.view(-1,1) * signal_to_goal_pos
                    pos_err = scale_ratio.view(-1,1) * signal_to_goal_pos

                    # clip linear velocity by norm
                    velocity_norm = torch.linalg.vector_norm(pos_err[:, :2] + 1e-6, dim=1, ord=np.inf)
                    scale_ratio = torch.clip(velocity_norm, self.minimum_linear_velocity_norm,
                                             self.maximum_linear_velocity_norm) / velocity_norm
                    pos_err[:, :2] = scale_ratio.view(-1, 1) * pos_err[:, :2]
                else:
                    # clip linear velocity by norm
                    velocity_norm = torch.linalg.vector_norm(actions[:, :2] + 1e-6, dim=1, ord=np.inf)
                    scale_ratio = torch.clip(velocity_norm, self.minimum_linear_velocity_norm, self.maximum_linear_velocity_norm) / velocity_norm
                    actions[:, :2] = scale_ratio.view(-1, 1) * actions[:, :2]

                    pos_err = actions[:, :2]

            # orientations / angular velocities
            if not self.enable_orientations:
                raise NotImplementedError
            else:
                if not self.observe_orientations:
                    # Orientation is not part of the policy output. The desired orientation is set to the final one.
                    # control the angle
                    box_orn_des = torch.zeros_like(box_orn_cur)
                    box_orn_des[..., 3] = 1.  # no rotation wrt the base
                    orn_err = orientation_error(box_orn_des, box_orn_cur)
                else:
                    if not self.control_vel:
                        raise NotImplementedError
                    else:
                        orn_err = torch.zeros_like(box_ang_vel_cur)

                        if self.enable_PD_to_goal:
                            box_orn_des = torch.zeros_like(box_orn_cur)
                            box_orn_des[..., 3] = 1.  # no rotation wrt the base

                            # clip signal from PD
                            signal_to_goal_orn = orientation_error(box_orn_des, box_orn_cur)
                            velocity_norm = torch.abs(signal_to_goal_orn[:, 2] + 1e-6)
                            scale_ratio = torch.clip(velocity_norm, 0.,
                                                     10.) / velocity_norm
                            #orn_err[:, 2] = actions[:, 2] + scale_ratio * signal_to_goal_orn[:, 2]
                            orn_err[:, 2] = scale_ratio * signal_to_goal_orn[:, 2]

                            # clip angular velocity by norm
                            velocity_norm = torch.abs(orn_err[:, 2] + 1e-6)
                            scale_ratio = torch.clip(velocity_norm, self.minimum_angular_velocity_norm,
                                                     self.maximum_angular_velocity_norm) / velocity_norm
                            orn_err[:, 2] = scale_ratio * orn_err[:, 2]
                        else:
                            # clip angular velocity by norm
                            velocity_norm = torch.abs(actions[:, 2] + 1e-6)
                            scale_ratio = torch.clip(velocity_norm, self.minimum_angular_velocity_norm,
                                                     self.maximum_angular_velocity_norm) / velocity_norm
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
            J_6dof = self.robot.jacobian(self.dof_pos)
            # we only need 3x3 jacobian since we have just rotation around z (4th and 5th column are 0 anyways)
            J = torch.zeros(size=(self.num_envs, 3, 3), device=self.device)
            J[..., 0:2, 0:3] = J_6dof[..., 0:2, 0:3]
            J[..., 2, 0:3] = J_6dof[..., -1, 0:3]

            if not self.use_osc:
                # Differentiable Inverse Kinematics
                J_pinv = torch.linalg.pinv(J)
                error_joints = J_pinv @ error_taskspace.unsqueeze(dim=-1)
                error_joints = error_joints.squeeze()

                joint_vel = self.dof_vel
                # PD controller at the joint level
                actions_dof_tensor = kp @ error_joints[..., None, ...] - kv @ joint_vel[..., None, ...]
            else:
                # OSC
                mm_inv = torch.inverse(self._mm)
                m_eef_inv = J @ mm_inv @ torch.transpose(J, 1, 2)
                m_eef = torch.inverse(m_eef_inv)

                actions_dof_tensor = J.transpose(2, 1) @ m_eef @ (
                            kp @ error_taskspace[..., None, ...] - kv @ task_vel[..., None, ...])

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

            self.gym.refresh_rigid_body_state_tensor(self.sim)

            box_rb_states = self.rb_state[self.box_rb_idxs]

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
def distance_orientation(desired, current):
    return 1. - (desired*current).sum(-1)**2

@torch.jit.script
def compute_box2d_insertion_reward(
        rb_state, box_rb_idxs, reset_buf, progress_buf, max_episode_length,
        enable_velocities_states=False, enable_orientations=False, enable_sparse_reward=False, reward_orientations=False
):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, bool, bool, bool, bool) -> Tuple[torch.Tensor, torch.Tensor]

    box_positions = rb_state[box_rb_idxs][:, 0:3]
    box_orientation = rb_state[box_rb_idxs][:, 3:7]
    box_linear_velocity = rb_state[box_rb_idxs][:, 7:10]
    box_angluar_velocity = rb_state[box_rb_idxs][:, 10:13]

    box_pos_dist = torch.sqrt(
        box_positions[..., 0] * box_positions[..., 0] +
        box_positions[..., 1] * box_positions[..., 1]
    )
    reward = -box_pos_dist

    box_orn_dist = torch.zeros_like(box_pos_dist)
    if enable_orientations:
        # get angle around z-axis which is yaw
        box_orn_des = torch.zeros_like(box_orientation)
        box_orn_des[..., 3] = 1.  # no rotation wrt the base
        orn_err = orientation_error(box_orn_des, box_orientation)
        box_orn_dist = torch.sqrt(
            orn_err[..., 2] * orn_err[..., 2]
        )
    if reward_orientations:
        reward -= box_orn_dist * 5

    condition = torch.logical_or(progress_buf >= max_episode_length - 1, box_pos_dist < 0.05)
    if enable_orientations:
        condition = torch.logical_or(progress_buf >= max_episode_length - 1, torch.logical_and(box_pos_dist < 0.05, box_orn_dist < 0.1))

    reset = torch.where(condition, torch.ones_like(reset_buf), reset_buf)

    if enable_sparse_reward:
        reward = reward * 0 - 1

    #add_final_reward = torch.where(condition, torch.ones_like(reward)*10., torch.zeros_like(reward))
    #reward = reward + add_final_reward


    return reward, reset
