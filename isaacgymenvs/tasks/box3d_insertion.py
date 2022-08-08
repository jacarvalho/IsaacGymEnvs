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
import torch
import xml.etree.ElementTree as ET
from typing import *

from pytorch3d.transforms import quaternion_multiply, axis_angle_to_quaternion

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from tasks.base.vec_task import VecTask


@torch.jit.script
def orientation_error(desired, current):
    """
    https://studywolf.wordpress.com/2018/12/03/force-control-of-task-space-orientation/
    """
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


class Box3DInsertion(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.max_push_effort = self.cfg["env"]["maxEffort"]

        self.enable_velocities_states = self.cfg['env']['enableVelocityState']
        self.enable_orientations = self.cfg["env"]["enableOrientations"]

        self.enable_ic = self.cfg["env"]["enableIC"]
        self.control_vel = self.cfg["env"]["controlVelocity"]
        self.learn_orientations = self.cfg["env"]["learnOrientations"]
        self.learn_stiffness = self.cfg["env"]["learnStiffness"]
        self.learn_damping = self.cfg["env"]["learnDamping"]
        if self.learn_damping:
            assert self.learn_stiffness

        # Observations
        # All relative to the world
        if not self.enable_orientations:
            # Without rotations
            # 0:3 - box position
            # 3:6 - box linear velocity
            if not self.enable_velocities_states:
                self.cfg["env"]["numObservations"] = 3
            else:
                self.cfg["env"]["numObservations"] = 3 + 3
        else:
            # With rotations
            # 0:3 - box position
            # 3:7 - box orientation
            # 7:10 - box linear velocity
            # 10:13 - box angular velocity
            if not self.enable_velocities_states:
                self.cfg["env"]["numObservations"] = 3 + 4
            else:
                self.cfg["env"]["numObservations"] = 3 + 4 + 3 + 3

        # Action is the desired velocity on the 3 joints representing the dofs (2 prismatic + 1 revolute)
        extra_actions = 0
        if not self.learn_orientations:
            if self.learn_stiffness:
                extra_actions += 4  # parameters of Kpos
            if self.learn_damping:
                extra_actions += 4  # parameters of Dpos
            self.cfg["env"]["numActions"] = 3 + extra_actions
        else:
            if self.learn_stiffness:
                extra_actions += 4 + 1  # parameters of Kpos and Korn
            if self.learn_damping:
                extra_actions += 4 + 1  # parameters of Dpos and Dorn
            self.cfg["env"]["numActions"] = 3 + extra_actions # TODO how many actions?

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)


        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state = gymtorch.wrap_tensor(rb_state_tensor)

        # vis
        self.axes_geom = gymutil.AxesGeometry(0.1)

        # Default IC stiffness and damping
        # 3 prismatic joints + 1 revolute joint
        self.kp = torch.zeros((self.num_envs, 3, 6 if self.enable_orientations else 3), device=self.device)
        self.kp_pos_factor = 5. if self.control_vel else 50.
        self.kp[:, :3, :3] = self.kp_pos_factor * torch.eye(3).reshape((1, 3, 3)).repeat(self.num_envs, 1, 1)
        if self.enable_orientations:
            self.kp_orn_factor = 10. if self.control_vel and self.learn_orientations else 50.
            self.kp[:, :3, 3:6] = self.kp_orn_factor * torch.eye(3).reshape((1, 3, 3)).repeat(self.num_envs, 1, 1)
        self.enable_damping_term = self.cfg["env"]["enableDampingTerm"]
        if self.enable_damping_term:
            self.kv = torch.zeros_like(self.kp)
        else:
            self.kv = 2 * torch.sqrt(self.kp)
        self.kv_pos_factor = 5. if self.control_vel else 50.
        self.kv_orn_factor = 10. if self.control_vel and self.learn_orientations else 50.

        self.enable_sparse_reward = self.cfg["env"]["enableSparseReward"]
        self.initial_position_bounds = self.cfg["env"].get("initialPositionBounds", [[-1, -1, -1], [1, 1, 1]]) #TODO make 3d

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

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        plane_params.distance = 0.0
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
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"]["assetRoot"])
            asset_file = self.cfg["env"]["asset"]["assetFileName"]
        else:
            raise KeyError

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.armature = 0.01
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

        self.envs = []
        self.box3d_handles = []
        self.box_rb_idxs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            self.envs.append(env_ptr)

            box3d_handle = self.gym.create_actor(env_ptr, box3d_asset, pose, "box3d", i, 0, 0)

            # Set properties for the 2 prismatic and 1 revolute joint
            dof_props = self.gym.get_actor_dof_properties(env_ptr, box3d_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][2] = gymapi.DOF_MODE_EFFORT
            #dof_props['driveMode'][3] = gymapi.DOF_MODE_EFFORT
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
           # if not self.enable_orientations:
                #dof_props['lower'][3] = 0.0
                #dof_props['upper'][3] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, box3d_handle, dof_props)

            self.box3d_handles.append(box3d_handle)

            # index of box rigid body
            rb_idx = self.gym.find_actor_rigid_body_index(env_ptr, box3d_handle, 'box', gymapi.DOMAIN_SIM)
            self.box_rb_idxs.append(rb_idx)

    def compute_observations(self, env_ids=None):
        #TODO what does mSVF need as input?
        #TODO also add -q for value etc @Joao?
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.obs_buf[env_ids, 0:3] = self.rb_state[self.box_rb_idxs][:, 0:3]  # box positions
        if not self.enable_orientations:
            if self.enable_velocities_states:
                self.obs_buf[env_ids, 3:6] = self.rb_state[self.box_rb_idxs][:, 7:10]  # box linear velocity
        else:
            self.obs_buf[env_ids, 3:7] = self.rb_state[self.box_rb_idxs][:, 3:7]  # box orientation
            if self.enable_velocities_states:
                self.obs_buf[env_ids, 10:13] = self.rb_state[self.box_rb_idxs][:, 10:13]  # box angular velocity

        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_box3d_insertion_reward(
            self.obs_buf,
            self.reset_buf, self.progress_buf, self.max_episode_length,
            enable_velocities_states=self.enable_velocities_states,
            enable_orientations=self.enable_orientations, enable_sparse_reward=self.enable_sparse_reward
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

            """
            NOT NEEDED IN 3D?
            
            #TODO make this rejection better? used to not sample inside the robot
            #TODO still correct for Stick?
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
        """
        if self.enable_orientations:
            positions[:, 3] = (-np.pi/2 - np.pi/2) * torch.rand(len(env_ids), device=self.device) + np.pi/2

        velocities = torch.zeros((len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def reset(self):
        """
        Overwrite since super class reset method does reset to (0,0) and this is called initally. We do not want to start in (0,0)
        """
        env_ids = torch.tensor(list(range(self.num_envs)), device=self.device)
        self.reset_idx(env_ids)
        self.post_physics_step()

        return super().reset()


    def pre_physics_step(self, actions):
        actions_dof_tensor = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float)

        if not self.enable_ic:
            # Direct control
            if not self.learn_orientations:
                actions_dof_tensor[:, :3] = actions.to(self.device).squeeze()
            else:
                actions_dof_tensor = actions.to(self.device).squeeze()
        else:
            # Impedance Control
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Set (possibly learned) stiffness and damping matrices
            kp = self.kp
            idx = 0
            if self.learn_stiffness:
                # TODO: Make sure Kpos_orn is positive definite
                idx = 3 if not self.learn_orientations else 4  # TODO should be 5 since rotate in 2 directions or even 6 for Roll pitch yaw
                action_kp_pos = actions[:, idx:idx+4]
                action_kp_pos_matrix = action_kp_pos.reshape(-1, 2, 2)
                kp[:, :2, :2] = self.kp_pos_factor * torch.bmm(action_kp_pos_matrix, action_kp_pos_matrix.transpose(-2, -1))
                if self.learn_orientations:
                    idx += 4
                    action_kp_orn = actions[:, idx:idx+1]
                    action_kp_orn_matrix = action_kp_orn.reshape(-1, 1, 1)
                    kp[:, 2, -1, None, None] = self.kp_orn_factor * torch.bmm(action_kp_orn_matrix, action_kp_orn_matrix.transpose(-2, -1))

            kv = self.kv
            if self.learn_damping:
                # TODO: Make sure Dpos_orn is positive definite
                idx += 1 if self.learn_orientations else 4
                action_kv_pos = actions[:, idx:idx+4]
                action_kv_pos_matrix = action_kv_pos.reshape(-1, 2, 2)
                kv[:, :2, :2] = self.kv_pos_factor * torch.bmm(action_kv_pos_matrix, action_kv_pos_matrix.transpose(-2, -1))
                if self.learn_orientations:
                    idx += 4
                    action_kv_orn = actions[:, idx:idx+1]
                    action_kv_orn_matrix = action_kv_orn.reshape(-1, 1, 1)
                    kp[:, 2, -1, None, None] = self.kv_orn_factor * torch.bmm(action_kv_orn_matrix, action_kv_orn_matrix.transpose(-2, -1))

            # Get current box poses and velocities
            box_pos_cur = self.rb_state[self.box_rb_idxs, :3]
            box_orn_cur = self.rb_state[self.box_rb_idxs, 3:7]
            box_lin_vel_cur = self.rb_state[self.box_rb_idxs, 7:10]
            box_ang_vel_cur = self.rb_state[self.box_rb_idxs, 10:13]

            # positions / linear velocities
            if not self.control_vel:
                # control the position
                box_pos_des = box_pos_cur.clone()
                box_pos_des[:, :3] = actions[:, :3]  # Modify only the x-y-z positions
                pos_err = box_pos_des * self.dt  # x_des - x = x + dx_des * dt - x = dx_des * dt
            else:
                # control the velocity
                box_lin_vel_des = box_lin_vel_cur.clone()
                box_lin_vel_des[:, :3] = actions[:, :3] # Modify only the x-y-z positions
                pos_err = box_lin_vel_des - box_lin_vel_cur

            # orientations / angular velocities
            if self.enable_orientations:
                if not self.learn_orientations:
                    # Orientation is not part of the policy output. The desired oritentation is set to the final one.
                    # control the angle
                    box_orn_des = torch.zeros_like(box_orn_cur)
                    box_orn_des[..., 3] = 1.  # no rotation wrt the base
                    orn_err = orientation_error(box_orn_des, box_orn_cur)
                else:
                    # TODO what to control here
                    if not self.control_vel:
                        # control the angle
                        box_orn_aa_des = torch.zeros(box_orn_cur.shape[0], 3, device=self.device)
                        box_orn_aa_des[..., 2] = actions[:, 2] * self.dt
                        box_orn_des = quaternion_multiply(box_orn_cur, axis_angle_to_quaternion(box_orn_aa_des))
                        orn_err = orientation_error(box_orn_des, box_orn_cur)
                    else:
                        # control the angular velocity
                        box_ang_vel_des = box_ang_vel_cur.clone()
                        box_ang_vel_des[..., 2] = actions[:, 2]  # angular velocity around z-axis
                        orn_err = box_ang_vel_des - box_ang_vel_cur
                dpose = torch.cat([pos_err, orn_err], -1)
            else:
                dpose = pos_err

            # velocities
            if self.enable_orientations:
                box_vel_cur = torch.cat([box_lin_vel_cur, box_ang_vel_cur], -1)
            else:
                box_vel_cur = box_lin_vel_cur

            # torques
            actions_dof_tensor = kp @ dpose[..., None, ...] - kv @ box_vel_cur[..., None, ...]

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
        box_state_buf, reset_buf, progress_buf, max_episode_length,
        enable_velocities_states=False, enable_orientations=False, enable_sparse_reward=False
):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, int, bool, bool, bool) -> Tuple[torch.Tensor, torch.Tensor]

    box_positions = box_state_buf[..., 0:3]
    box_positions[..., 2] -= 0.027  # TODO why need to do this? because z min is this since box cannot go below ground?
    box_pos_dist = torch.sqrt(
        box_positions[..., 0] * box_positions[..., 0] +
        box_positions[..., 1] * box_positions[..., 1] +
        box_positions[..., 2] * box_positions[..., 2]

    )
    reward = -box_pos_dist

    box_orn_dist = torch.zeros_like(box_pos_dist)
    if enable_orientations:
        box_orientations = box_state_buf[..., 3:7]
        desired = torch.zeros_like(box_orientations)
        desired[..., 3] = 1.  # no rotation wrt base
        box_orn_dist = distance_orientation(desired, box_orientations)
        reward -= box_orn_dist

    condition = torch.logical_or(progress_buf >= max_episode_length - 1, box_pos_dist < 0.005)

    if enable_orientations:
        condition = torch.logical_and(condition, box_orn_dist < 0.1)

    reset = torch.where(condition, torch.ones_like(reset_buf), reset_buf)

    if enable_sparse_reward:
        reward = reward * 0 - 1

    return reward, reset
