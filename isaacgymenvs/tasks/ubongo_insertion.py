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

import pytorch_kinematics as pk
import theseus as th

import torch


@torch.jit.script
def orientation_error(desired, current):
    """
    https://studywolf.wordpress.com/2018/12/03/force-control-of-task-space-orientation/
    """
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

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


class UbongoInsertion(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.use_osc = False

        self.cfg = cfg

        self.observe_force = self.cfg["env"].get("observeForce", False)

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.enable_PD_to_goal = self.cfg["env"].get("enable_PD_to_goal", False)

        self.controller_freq = self.cfg["env"].get("controller_freq", None)
        self.recompute_prephysics_step = self.cfg["env"].get("recomputePrePhysicsStep", False)

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        #self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.minimum_linear_velocity_norm = self.cfg["env"].get("minimum_linear_velocity_norm", 0.)
        self.maximum_linear_velocity_norm = self.cfg["env"].get("maximum_linear_velocity_norm", 10.)
        self.minimum_angular_velocity_norm = self.cfg["env"].get("minimum_angular_velocity_norm", 0.)
        self.maximum_angular_velocity_norm = self.cfg["env"].get("maximum_angular_velocity_norm", 1.)

        self.observe_velocities = self.cfg['env']['enableVelocityState']
        self.learn_orientations = self.cfg["env"]["learnOrientations"]
        self.observe_orientations = self.cfg["env"].get("observeOrientations", True)

        if self.observe_orientations:
            self.observe_joint_angles = self.cfg["env"]["observeJointAngles"]
            self.observe_rotation_matrix = self.cfg["env"]["observeRotationMatrix"]
            self.observe_quaternion = self.cfg["env"]["observeQuaternion"]
            if self.observe_joint_angles + self.observe_rotation_matrix + self.observe_quaternion > 1:
                raise KeyError("Just a single observation type allowed")
            if self.observe_joint_angles + self.observe_rotation_matrix + self.observe_quaternion == 0:
                self.observe_joint_angles = True

        self.enable_orientations = self.cfg["env"]["enableOrientations"]
        self.enable_ic = self.cfg["env"]["enableIC"]

        self.control_vel = self.cfg["env"]["controlVelocity"]
        self.learn_stiffness = self.cfg["env"]["learnStiffness"]

        self.just_learn_stiffness = self.cfg["env"].get("justLearnStiffness", False)
        if self.just_learn_stiffness:
            self.enable_PD_to_goal = True
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

        # Observations
        # All relative to the world
        if not self.observe_orientations:
            # Without rotations
            # 0:3 - box position
            # 3:6 - box linear velocity
            if not self.observe_velocities:
                self.cfg["env"]["numObservations"] = 3
            else:
                self.cfg["env"]["numObservations"] = 3 + 3
        else:
            # With rotations
            # 0:3 - box position
            # 3:7 - box orientation
            # 7:10 - box linear velocity
            # 10:13 - box angular velocity
            if not self.observe_velocities:
                if self.observe_joint_angles:
                    self.cfg["env"]["numObservations"] = 3 + 3
                if self.observe_rotation_matrix:
                    self.cfg["env"]["numObservations"] = 3 + 9
                if self.observe_quaternion:
                    self.cfg["env"]["numObservations"] = 3 + 4
            else:
                raise NotImplementedError
        if self.observe_force:
            self.cfg["env"]["numObservations"] += 3

        # Action is the desired velocity on the 3 joints representing the dofs (2 prismatic + 1 revolute)
        extra_actions = 0
        if not self.learn_orientations:
            if self.learn_stiffness:
                extra_actions += 3  # parameters of Kpos
            self.cfg["env"]["numActions"] = 3 + extra_actions
        else:
            if self.learn_stiffness:
                extra_actions += 3+3  # parameters of Kpos and Korn
            self.cfg["env"]["numActions"] = 6 + extra_actions

        if self.just_learn_stiffness:
            self.cfg["env"]["numActions"] = 6

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

            _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
            self.contact_sensor_tensor = gymtorch.wrap_tensor(_net_cf)
            #dof_sensor_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            #self.dof_forces = gymtorch.wrap_tensor(dof_sensor_tensor).view(self.num_envs, 6)

        # vis
        self.axes_geom = gymutil.AxesGeometry(0.3)

        # Default IC stiffness and damping
        # 3 prismatic joints + 1 revolute joint
        self.kp = torch.zeros((self.num_envs, 6 if self.enable_orientations else 3, 6 if self.enable_orientations else 3), device=self.device)
        self.kp_pos_factor = 100.
        self.kp[:, :3, :3] = self.kp_pos_factor * torch.eye(3).reshape((1, 3, 3)).repeat(self.num_envs, 1, 1)
        if self.enable_orientations:
            self.kp_orn_factor = 25.
            self.kp[:, 3:6, 3:6] = self.kp_orn_factor * torch.eye(3).reshape((1, 3, 3)).repeat(self.num_envs, 1, 1)

        self.enable_damping_term = self.cfg["env"]["enableDampingTerm"]
        if self.enable_damping_term:
            #standard_kv = 2 * torch.sqrt(self.kp)
            self.kv = torch.zeros((self.num_envs, 6 if self.enable_orientations else 3, 6 if self.enable_orientations else 3), device=self.device)
            self.kv_pos_factor = 2 * np.sqrt(self.kp_pos_factor)
            self.kv[:, :3, :3] = self.kv_pos_factor * torch.eye(3).reshape((1, 3, 3)).repeat(self.num_envs, 1, 1)
            if self.enable_orientations:
                self.kv_orn_factor = 2 * np.sqrt(self.kp_orn_factor * 0.01)
                self.kv[:, 3:6, 3:6] = self.kv_orn_factor * torch.eye(3).reshape((1, 3, 3)).repeat(self.num_envs, 1, 1)
        else:
            self.kv = torch.zeros_like(self.kp)


        self.enable_sparse_reward = self.cfg["env"]["enableSparseReward"]
        self.initial_position_bounds = self.cfg["env"].get("initialPositionBounds", [[-1, -1, -1], [1, 1, 1]])

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

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        plane_params.distance = 0.1
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
        #asset_options.armature = 0.01
        asset_options.disable_gravity = True
        asset_options.enable_gyroscopic_forces = False
        box3d_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(box3d_asset)

        # default pose
        box_size = 0.05
        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p = gymapi.Vec3(0., 0., box_size / 2)
            pose.r = gymapi.Quat(0, 0, 0, 1)
        else:
            pose.p = gymapi.Vec3(0., box_size / 2, 0.)
            pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

        # create force sensors attached to the EEF
        if self.observe_force:
            eef_index = self.gym.find_asset_rigid_body_index(box3d_asset, "box")
            sensor_pose = gymapi.Transform()

            sensor_props = gymapi.ForceSensorProperties()
            sensor_props.enable_forward_dynamics_forces = True # default True
            sensor_props.enable_constraint_solver_forces = True # default True

            self.gym.create_asset_force_sensor(box3d_asset, eef_index, sensor_pose, sensor_props)

        self.envs = []
        self.box3d_handles = []
        self.box_rb_idxs = []

        self.actor_handles = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            self.envs.append(env_ptr)

            box3d_handle = self.gym.create_actor(env_ptr, box3d_asset, pose, "box3d", i, 0, 0)
            self.actor_handles.append(box3d_handle)
            #self.gym.enable_actor_dof_force_sensors(env_ptr, box3d_handle)

            # Set properties for the 2 prismatic and 1 revolute joint
            dof_props = self.gym.get_actor_dof_properties(env_ptr, box3d_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][2] = gymapi.DOF_MODE_EFFORT

            dof_props['stiffness'].fill(0.0)
            dof_props['damping'].fill(0.0)
            dof_props['friction'].fill(0.0)
            if self.enable_orientations:
                dof_props['driveMode'][3] = gymapi.DOF_MODE_EFFORT
                dof_props['driveMode'][4] = gymapi.DOF_MODE_EFFORT
                dof_props['driveMode'][5] = gymapi.DOF_MODE_EFFORT
                #dof_props['lower'][3:5] = 0.0
                #dof_props['upper'][3:5] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, box3d_handle, dof_props)

            self.box3d_handles.append(box3d_handle)

            props = self.gym.get_actor_rigid_shape_properties(env_ptr, box3d_handle)
            for prop in props:
                prop.friction = 0.4  # see https://www.researchgate.net/publication/330003074_Wear_and_coefficient_of_friction_of_PLA_-_Graphite_composite_in_3D_printing_technology
            self.gym.set_actor_rigid_shape_properties(env_ptr, box3d_handle, props)#

            # index of box rigid body
            rb_idx = self.gym.find_actor_rigid_body_index(env_ptr, box3d_handle, 'box', gymapi.DOMAIN_SIM)
            self.box_rb_idxs.append(rb_idx)

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
        self.obs_buf[env_ids, 0:3] = box_positions[:, 0:3]
        if not self.observe_orientations:
            if self.observe_velocities:
                self.obs_buf[env_ids, 3:6] = box_linear_velocity[:, 0:3]
        else:
            # get angle around z-axis which is yaw
            if self.observe_joint_angles:
                self.obs_buf[env_ids, 3:6] = self.dof_pos[env_ids, 3:6]
            if self.observe_rotation_matrix:
                box_orientation_wxyz = quat_xyzw_to_wxyz(box_orientation)
                rot_mat = th.SO3(quaternion=box_orientation_wxyz).to_matrix()
                self.obs_buf[env_ids, 3:12] = rot_mat[env_ids, :].view(-1,9)
            if self.observe_quaternion:
                box_orientation_wxyz = quat_xyzw_to_wxyz(box_orientation)
                self.obs_buf[env_ids, 3:7] = box_orientation_wxyz
            if self.observe_velocities:
                self.obs_buf[env_ids, 12:15] = box_linear_velocity[:, 0:3]  # box linear velocity
                self.obs_buf[env_ids, 15:18] = box_angluar_velocity[:, 0:3]  # box angular velocity

        if self.observe_force:
            #self.gym.refresh_force_sensor_tensor(self.sim)
            #self.obs_buf[env_ids, -3:] = self.vec_sensor_tensor[env_ids, 0:3]

            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.obs_buf[env_ids, -3:] = self.contact_sensor_tensor[self.box_rb_idxs][env_ids, 0:3]

        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_box3d_insertion_reward(
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
            positions[:, 2] = (min_initial_position[2] - max_initial_position[2]) * torch.rand(len(env_ids), device=self.device) + max_initial_position[2]

        if self.enable_orientations:
            rot_lim = np.pi/4
            positions[:, 3] = (-rot_lim - rot_lim) * torch.rand(len(env_ids), device=self.device) + rot_lim
            positions[:, 4] = (-rot_lim - rot_lim) * torch.rand(len(env_ids), device=self.device) + rot_lim
            positions[:, 5] = (-rot_lim - rot_lim) * torch.rand(len(env_ids), device=self.device) + rot_lim

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
        a_pos = torch.tanh(actions_K[..., 0:3])
        a_pos_scaled = a_pos * self._delta_K_pos + self._central_K_pos
        a_orn = torch.tanh(actions_K[..., 3:6])
        a_orn_scaled = a_orn * self._delta_K_orn + self._central_K_orn

        # print(a_pos_scaled[0], torch.sqrt(a_pos_scaled)[0])
        # print(a_orn_scaled[0], torch.sqrt(a_orn_scaled)[0])

        # calculate stiffness and damping
        K_pos_x = a_pos_scaled[..., 0]
        K_pos_y = a_pos_scaled[..., 1]
        K_pos_z = a_pos_scaled[..., 1]
        K_orn_x = a_orn_scaled[..., 0]
        K_orn_y = a_orn_scaled[..., 1]
        K_orn_z = a_orn_scaled[..., 2]

        # print(K_pos_x[0], K_pos_y[0], K_orn[0])

        kp = torch.eye(6, device=self.device).reshape((1, 6, 6)).repeat(self.num_envs, 1, 1)
        kp[..., 0, 0] = K_pos_x
        kp[..., 1, 1] = K_pos_y
        kp[..., 2, 2] = K_pos_z
        kp[..., 3, 3] = K_orn_x
        kp[..., 4, 4] = K_orn_y
        kp[..., 5, 5] = K_orn_z


        kv = torch.eye(6, device=self.device).reshape((1, 6, 6)).repeat(self.num_envs, 1, 1)
        if self.use_osc:
            kv[..., 0, 0] = 2 * torch.sqrt(K_pos_x)
            kv[..., 1, 1] = 2 * torch.sqrt(K_pos_y)
            kv[..., 2, 2] = 2 * torch.sqrt(K_pos_z)  # mass is canceled in osc
            kv[..., 3, 3] = 2 * torch.sqrt(K_orn_x)
            kv[..., 4, 4] = 2 * torch.sqrt(K_orn_y)
            kv[..., 5, 5] = 2 * torch.sqrt(K_orn_z)
        else:
            # the factors come from the mass matrix and need to be applied to get a stable system
            kv[..., 0, 0] = 2 * torch.sqrt(K_pos_x*1)
            kv[..., 1, 1] = 2 * torch.sqrt(K_pos_y*1)
            kv[..., 2, 2] = 2 * torch.sqrt(K_pos_z*1)
            kv[..., 3, 3] = 2 * torch.sqrt(K_orn_x*0.01)
            kv[..., 4, 4] = 2 * torch.sqrt(K_orn_y*0.01)
            kv[..., 5, 5] = 2 * torch.sqrt(K_orn_z*0.01)

        return kp, kv

    def pre_physics_step(self, actions, step=0):
        actions_dof_tensor = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float)

        if not self.enable_ic:
            # Direct control
            if not self.learn_orientations:
                actions_dof_tensor[:, :2] = actions.to(self.device).squeeze()
            else:
                actions_dof_tensor = actions.to(self.device).squeeze()
        else:
            # Impedance Control
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Set (possibly learned) stiffness and damping matrices
            if self.learn_stiffness:
                if self.just_learn_stiffness:
                    kp, kv = self._create_stiffness_and_damping_matrices(actions.clone())
                else:
                    kp, kv = self._create_stiffness_and_damping_matrices(actions[..., 6:12].clone())
            else:
                kp = self.kp
                kv = self.kv

            # Get current box poses and velocities
            box_pos_cur = self.rb_state[self.box_rb_idxs, :3]
            box_orn_cur = self.rb_state[self.box_rb_idxs, 3:7]
            box_lin_vel_cur = self.rb_state[self.box_rb_idxs, 7:10]
            box_ang_vel_cur = self.rb_state[self.box_rb_idxs, 10:13]

            if step > 0:
                raise NotImplementedError

            # calculate desired linear velocities
            if self.enable_PD_to_goal:
                signal_to_goal_pos = -box_pos_cur[:, :3]

                # clip first the PD signal
                velocity_norm = torch.linalg.vector_norm(signal_to_goal_pos + 1e-6, dim=1, ord=np.inf)
                scale_ratio = torch.clip(velocity_norm, 0., self.maximum_linear_velocity_norm) / velocity_norm
                # add PD to signal from policy
                if self.just_learn_stiffness:
                    pos_err = scale_ratio.view(-1, 1) * signal_to_goal_pos
                else:
                    pos_err = actions[:, :3] + scale_ratio.view(-1, 1) * signal_to_goal_pos

                # clip linear velocity by norm
                velocity_norm = torch.linalg.vector_norm(pos_err[:, :3] + 1e-6, dim=1, ord=np.inf)
                scale_ratio = torch.clip(velocity_norm, self.minimum_linear_velocity_norm,
                                         self.maximum_linear_velocity_norm) / velocity_norm
                pos_err[:, :3] = scale_ratio.view(-1, 1) * pos_err[:, :3]
            else:
                # clip translating actions by norm
                linear_velocity_norm = torch.linalg.vector_norm(actions[:, :3] + 1e-6, dim=1, ord=np.inf)
                linear_scale_ratio = torch.clip(linear_velocity_norm, self.minimum_linear_velocity_norm,
                                         self.maximum_linear_velocity_norm) / linear_velocity_norm
                actions[:, :3] = linear_scale_ratio.view(-1, 1) * actions[:, :3]

                pos_err = actions[:, :3]  # position error is equal to the velocities that should be applied

            # orientations / angular velocities
            if self.enable_orientations:
                if not self.learn_orientations:
                    # Orientation is not part of the policy output. The desired oritentation is set to the final one.
                    # control the angle
                    box_orn_des = torch.zeros_like(box_orn_cur)
                    box_orn_des[..., 3] = 1.  # no rotation wrt the base
                    orn_err = orientation_error(box_orn_des, box_orn_cur)
                else:
                    orn_err = torch.zeros_like(box_ang_vel_cur)

                    if self.enable_PD_to_goal:
                        box_orn_des = torch.zeros_like(box_orn_cur)
                        box_orn_des[..., 3] = 1.  # no rotation wrt the base

                        # clip signal from PD
                        signal_to_goal_orn = orientation_error(box_orn_des, box_orn_cur)
                        velocity_norm = torch.linalg.vector_norm(signal_to_goal_orn + 1e-6, dim=1, ord=np.inf)
                        scale_ratio = torch.clip(velocity_norm, 0.,
                                                 self.maximum_angular_velocity_norm) / velocity_norm
                        if self.just_learn_stiffness:
                            orn_err[:, 2] = scale_ratio * signal_to_goal_orn[:, 2]
                        else:
                            orn_err = actions[..., 3:6] + scale_ratio.view(-1,1) * signal_to_goal_orn

                        # clip angular velocity by norm
                        velocity_norm = torch.linalg.vector_norm(orn_err + 1e-6, dim=1, ord=np.inf)
                        scale_ratio = torch.clip(velocity_norm, self.minimum_angular_velocity_norm,
                                                 self.maximum_angular_velocity_norm) / velocity_norm
                        orn_err = scale_ratio.view(-1,1) * orn_err
                    else:
                        # clip rotating actions by norm
                        angular_velocity_norm = torch.linalg.vector_norm(actions[..., 3:6] + 1e-6, dim=1, ord=np.inf)
                        angular_scale_ratio = torch.clip(angular_velocity_norm, self.minimum_angular_velocity_norm,
                                                 self.maximum_angular_velocity_norm) / angular_velocity_norm
                        actions[..., 3:6] = angular_scale_ratio.view(-1, 1) * actions[..., 3:6]

                        orn_err = actions[..., 3:6]

                error = torch.cat([pos_err, orn_err], -1)
            else:
                error = pos_err

            # velocities
            if self.enable_orientations:
                box_vel_cur = torch.cat([box_lin_vel_cur, box_ang_vel_cur], -1)
            else:
                box_vel_cur = box_lin_vel_cur

            # apply jacobian to map back to joint space error
            J = self.robot.jacobian(self.dof_pos)
            J_pinv = torch.linalg.pinv(J)
            dpose = J_pinv @ error.unsqueeze(dim=-1)
            dpose = dpose.squeeze()

            # torques
            #TODO box_Vel cur should be in joints! not box velocity, mul jacobbian also with this
            actions_dof_tensor = kp @ dpose[..., None, ...] - kv @ box_vel_cur[..., None, ...] #TODO clean make box_vel_cur -> joint_vels!

            self.gym.refresh_dof_state_tensor(self.sim)

        #print(actions_dof_tensor[0])
        forces = gymtorch.unwrap_tensor(actions_dof_tensor.squeeze())
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

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
def compute_box3d_insertion_reward(
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
        box_positions[..., 1] * box_positions[..., 1] +
        (box_positions[..., 2]-0.25)**2
    )
    reward = -box_pos_dist

    box_orn_dist = torch.zeros_like(box_pos_dist)
    if enable_orientations:
        box_orn_des = torch.zeros_like(box_orientation)
        box_orn_des[..., 3] = 1.  # no rotation wrt the base
        orn_error = orientation_error(box_orn_des, box_orientation)
        box_orn_dist = torch.sqrt(
            orn_error[..., 0] * orn_error[..., 0] +
            orn_error[..., 1] * orn_error[..., 1] +
            orn_error[..., 2] * orn_error[..., 2]
        )
    if reward_orientations:
        reward -= (box_orn_dist / np.pi) *5

    condition = torch.logical_or(progress_buf >= max_episode_length - 1, box_pos_dist < 0.1)
    if enable_orientations:
        condition = torch.logical_or(progress_buf >= max_episode_length - 1, torch.logical_and(box_pos_dist < 0.1, box_orn_dist < 0.05))

    reset = torch.where(condition, torch.ones_like(reset_buf), reset_buf)

    if enable_sparse_reward:
        reward = reward * 0 - 1

    return reward, reset



