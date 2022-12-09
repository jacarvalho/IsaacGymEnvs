# Copyright (c) 2021-2022, NVIDIA Corporation
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

import numpy as np
import os

from isaacgym import gymtorch, gymutil
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
import theseus as th

import torch


"""
different goal representations to test

rotate 180° around y
    Rotation Matrix
        [[-1., 0., 0.], 
         [0., 1., 0.], 
         [0., 0., -1.]]
    Axis Angle
        [0, -3.1415927, 0]
    Quaternion xyzw
        [0, 1, 0, 0]) 
        
rotate 180° around y then rotate 90° around z
    Rotation Matrix
        [[0., 1., 0.], 
         [1., 0., 0.], 
         [0., 0., -1.]]
    Axis Angle
        [2.2214415, 2.2214415, 0]
    Quaternion xyzw
        [0.7071068, 0.7071068, 0, 0]

rotate 180° around x
    Rotation Matrix
        [[1., 0., 0.], 
         [0., -1., 0.], 
         [0., 0., -1.]]
    Axis Angle
        [3.1415927, 0, 0]
    Quaternion xyzw
        [1, 0, 0, 0]

rotate 180° around x 90° around z
    Rotation Matrix
        [[0., -1., 0.], 
         [-1., 0., 0.], 
         [0., 0., -1.]]
    Axis Angle
        [-2.2214415, 2.2214415, 0]
    Quaternion xyzw
        [-0.7071068, 0.7071068, 0, 0]
"""

class FrankaBox3DInsertion(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.observe_force = self.cfg["env"].get("observeForce", False)
        self.enable_PD_to_goal = self.cfg["env"].get("enable_PD_to_goal", False)

        self.goal_position = [0., 0., 0.]
        #self.goal_orientation = [0.7071068, 0.7071068, 0, 0] #xyzw
        self.goal_orientation = [1, 0, 0, 0] #xyzw

        self.controller_freq = self.cfg["env"].get("controller_freq", None)
        self.enable_sparse_reward = self.cfg["env"]["enableSparseReward"]  # if set to True, reward is for each action -1
        self.learn_orientations = self.cfg["env"]["learnOrientations"]  # if set to True, orientation is controlled by agent, if False by PD controller
        self.observe_orientations = self.cfg["env"].get("observeOrientations", True)


        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.max_delta_pos = self.cfg["env"].get("maxDeltaPosition", np.Inf)
        self.max_delta_orn = self.cfg["env"].get("maxDeltaOrientation", np.Inf)

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

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "inv_diff_kinematics"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # Observation type
        self.observation_type = self.cfg["env"]["observationType"]
        assert self.observation_type in {"pos_and_quat", "pos_and_rotMat"},\
            "Invalid control type specified. Must be one of: {pos_and_quat, pos_and_rotMat}"

        # Dimensions
        if self.observe_orientations:
            # Observations include: EEF_pose 7 = 3 position + 4 quaternion of rotation OR 12 = 3 position + 3*3 Rotation matrix
            self.cfg["env"]["numObservations"] = 7 if self.observation_type == "pos_and_quat" else 12
            # Actions include: delta EEF in World Space - 6 = 3 position + 3 rotation (axis angle)
            if self.learn_orientations:
                self.cfg["env"]["numActions"] = 6
            else:
                self.cfg["env"]["numActions"] = 3
        else:
            # Observations include: EEF_position -> 3
            self.cfg["env"]["numObservations"] = 3
            # Actions include: delta position -> 3
            self.cfg["env"]["numActions"] = 3
        if self.observe_force:
            self.cfg["env"]["numObservations"] += 3
        if self.learn_stiffness:
            self.cfg["env"]["numActions"] += 6


        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed

        # Tensor placeholders
        self._root_state = None                 # State of root body        (n_envs, 13)
        self._dof_state = None                  # State of all joints       (n_envs, n_dof)
        self._q = None                          # Joint positions           (n_envs, n_dof)
        self._qd = None                         # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None           # State of all rigid bodies (n_envs, n_bodies, 13)
        self._contact_forces = None             # Contact forces in sim
        self._eef_state = None                  # end effector state (at grasping point)
        self._eef_lf_state = None               # end effector state (at left fingertip)
        self._eef_rf_state = None               # end effector state (at left fingertip)
        self._j_eef = None                      # Jacobian for end effector
        self._mm = None                         # Mass matrix
        self._arm_control = None                # Tensor buffer for controlling arm
        self._pos_control = None                # Position actions
        self._effort_control = None             # Torque actions
        self._franka_effort_limits = None       # Actuator effort limits for franka
        self._global_indices = None             # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.axes_geom = gymutil.AxesGeometry(0.3)  # used to visualize the EEF reference frame

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            #[0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854], device=self.device
            [-0.1692,  0.4817,  0.1881, -1.8218, -0.1215,  2.3025,  0.8042], device=self.device
        )

        # create force sensor tensor
        if self.observe_force:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            sensors_per_env = 1
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

            _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
            self.contact_sensor_tensor = gymtorch.wrap_tensor(_net_cf)

            #dof_sensor_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            #self.dof_forces = gymtorch.wrap_tensor(dof_sensor_tensor).view(self.num_envs, 7)

        # Gains
        self.kp = to_torch([150.] * 6, device=self.device) if self.control_type == "osc" else to_torch([150.] * 7,
                                                                                                       device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.1
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

        if "asset" in self.cfg["env"]:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            asset_root = project_root + self.cfg["env"]["asset"]["assetRoot"]
            franka_asset_file = self.cfg["env"]["asset"]["assetFileNameFranka"]
        else:
            raise KeyError

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # create force sensors attached to the EEF
        if self.observe_force:
            eef_index = self.gym.find_asset_rigid_body_index(franka_asset, "panda_grip_site")
            sensor_pose = gymapi.Transform()
            sensor_pose.p = gymapi.Vec3(0., 0., 0.)
            sensor_pose.r = gymapi.Quat(0, 0, 0, 1)

            sensor_props = gymapi.ForceSensorProperties()
            sensor_props.enable_forward_dynamics_forces = False # default True
            sensor_props.enable_constraint_solver_forces = True # default True

            self.gym.create_asset_force_sensor(franka_asset, eef_index, sensor_pose, sensor_props)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # Create table asset
        insertion_object_height = 0.15
        table_thickness = 0.05
        table_pos = [0.0, 0.0, -table_thickness/2-insertion_object_height/2]  # make so that center of insertion object can be in (0,0,0)
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.7, 0.0,  -insertion_object_height/2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        # Create insertion box in which object should be inserted into
        insertion_box_height = 0.05
        insertion_box_outer_bounds = [0.2, 0.2]  # defines the size of the insertion box [x,y]
        insertion_box_hole_bounds = [0.05, 0.05]  # defines the size of the hole for insertion [x,y]
        insertion_box_pos = [0.0, 0.0, -insertion_object_height/2 + insertion_box_height / 2]
        insertion_box_opts = gymapi.AssetOptions()
        insertion_box_opts.fix_base_link = True
        insertion_box_asset_top = self.gym.create_box(self.sim, *[insertion_box_hole_bounds[0], insertion_box_outer_bounds[1]/2-insertion_box_hole_bounds[1]/2, insertion_box_height], insertion_box_opts)
        insertion_box_asset_bottom = self.gym.create_box(self.sim, *[insertion_box_hole_bounds[0], insertion_box_outer_bounds[1]/2-insertion_box_hole_bounds[1]/2, insertion_box_height], insertion_box_opts)
        insertion_box_asset_left = self.gym.create_box(self.sim, *[insertion_box_outer_bounds[0]/2-insertion_box_hole_bounds[0]/2, insertion_box_outer_bounds[1], insertion_box_height], insertion_box_opts)
        insertion_box_asset_right = self.gym.create_box(self.sim, *[insertion_box_outer_bounds[0]/2-insertion_box_hole_bounds[0]/2, insertion_box_outer_bounds[1], insertion_box_height], insertion_box_opts)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)

        # Define start pose for franka
        franka_offset = -0.65
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(franka_offset, 0.0, -insertion_object_height/2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for insertion box
        insertion_box_top_start_pose = gymapi.Transform()
        insertion_box_top_pos = insertion_box_pos.copy()
        insertion_box_top_pos[1] += insertion_box_outer_bounds[1]/2 - (insertion_box_outer_bounds[1]/2-insertion_box_hole_bounds[1]/2) / 2
        insertion_box_top_start_pose.p = gymapi.Vec3(*insertion_box_top_pos)
        insertion_box_top_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        insertion_box_bottom_start_pose = gymapi.Transform()
        insertion_box_bottom_pos = insertion_box_pos.copy()
        insertion_box_bottom_pos[1] -= insertion_box_outer_bounds[1]/2 - (insertion_box_outer_bounds[1]/2 - insertion_box_hole_bounds[1]/2) / 2
        insertion_box_bottom_start_pose.p = gymapi.Vec3(*insertion_box_bottom_pos)
        insertion_box_bottom_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        insertion_box_left_start_pose = gymapi.Transform()
        insertion_box_left_pos = insertion_box_pos.copy()
        insertion_box_left_pos[0] -= insertion_box_outer_bounds[0]/2 - (insertion_box_outer_bounds[0]/2-insertion_box_hole_bounds[0]/2) / 2
        insertion_box_left_start_pose.p = gymapi.Vec3(*insertion_box_left_pos)
        insertion_box_left_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        insertion_box_right_start_pose = gymapi.Transform()
        insertion_box_right_pos = insertion_box_pos.copy()
        insertion_box_right_pos[0] += insertion_box_outer_bounds[0]/2 - (insertion_box_outer_bounds[0]/2-insertion_box_hole_bounds[0]/2) / 2
        insertion_box_right_start_pose.p = gymapi.Vec3(*insertion_box_right_pos)
        insertion_box_right_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 1 + 1 + 4    # for 1 table, 1 table stand, 4 insertion box
        max_agg_shapes = num_franka_shapes + 1 + 1 + 4    # for 1 table, 1 table stand, 4 insertion box

        self.frankas = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(franka_offset + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)

            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            #self.gym.enable_actor_dof_force_sensors(env_ptr, franka_actor)



            friction_coefficient = 0.4  # see https://www.researchgate.net/publication/330003074_Wear_and_coefficient_of_friction_of_PLA_-_Graphite_composite_in_3D_printing_technology
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, franka_actor)
            props[-1].friction = friction_coefficient
            self.gym.set_actor_rigid_shape_properties(env_ptr, franka_actor, props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_actor)
            props[-1].friction = friction_coefficient
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, props)

            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_stand_actor)
            props[-1].friction = friction_coefficient
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_stand_actor, props)

            # Create insertion box
            insertion_box_top_actor = self.gym.create_actor(env_ptr, insertion_box_asset_top,
                                                            insertion_box_top_start_pose, "insertion_box_top",
                                                            i, 1, 0)
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, insertion_box_top_actor)
            props[-1].friction = friction_coefficient
            self.gym.set_actor_rigid_shape_properties(env_ptr, insertion_box_top_actor, props)

            insertion_box_bottom_actor = self.gym.create_actor(env_ptr, insertion_box_asset_bottom,
                                                            insertion_box_bottom_start_pose, "insertion_box_bottom",
                                                            i, 1, 0)
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, insertion_box_bottom_actor)
            props[-1].friction = friction_coefficient
            self.gym.set_actor_rigid_shape_properties(env_ptr, insertion_box_bottom_actor, props)

            insertion_box_left_actor = self.gym.create_actor(env_ptr, insertion_box_asset_left,
                                                            insertion_box_left_start_pose, "insertion_box_left",
                                                            i, 1, 0)
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, insertion_box_left_actor)
            props[-1].friction = friction_coefficient
            self.gym.set_actor_rigid_shape_properties(env_ptr, insertion_box_left_actor, props)

            insertion_box_right_actor = self.gym.create_actor(env_ptr, insertion_box_asset_right,
                                                            insertion_box_right_start_pose, "insertion_box_right",
                                                            i, 1, 0)
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, insertion_box_right_actor)
            props[-1].friction = friction_coefficient
            self.gym.set_actor_rigid_shape_properties(env_ptr, insertion_box_right_actor, props)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
            "insertion_obj": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "insertion_object"),

        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]

        # Initialize indices
        # *7 comes from Franka + Table + TableStand + 4 Insertion box!
        self._global_indices = torch.arange(self.num_envs * 7, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            # Franka
            "q": self._q[:, :],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
        })
        self.eef_pos_des = torch.zeros_like(self.states["eef_pos"])
        self.eef_pos_des = torch.tensor(self.goal_position, device=self.device)
        self.eef_orn_des = torch.zeros_like(self.states["eef_quat"])
        self.eef_orn_des[..., 0:4] = torch.tensor(self.goal_orientation, device=self.device)

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_insertion_reward(
            self.states["eef_pos"], self.states["eef_quat"], self.eef_pos_des, self.eef_orn_des, self.reset_buf, self.progress_buf, self.max_episode_length,
            enable_sparse_reward=self.enable_sparse_reward, reward_orientations=self.observe_orientations
        )

    def _process_states_for_observations(self, states):
        eef_pos = states["eef_pos"] * 3  # make this larger because we want the robot to move in a frame of around -1 to 1 in each axis (but he roughly moves in -0.3 to 0.3
        eef_quat = states["eef_quat"]
        eef_quat_wxyz = quat_xyzw_to_wxyz(eef_quat)  # transform quaternion so that real part is first

        return eef_pos, eef_quat_wxyz

    def compute_observations(self):
        self._refresh()

        eef_pos, eef_quat_wxyz = self._process_states_for_observations(self.states)

        if not self.observe_orientations:
            self.obs_buf = eef_pos
        else:
            if self.observation_type == "pos_and_rotMat":
                eef_R = th.SO3(quaternion=eef_quat_wxyz).to_matrix()
                self.obs_buf = torch.cat([eef_pos, eef_R.view(-1,9)], dim=-1)
            elif self.observation_type == "pos_and_quat":
                self.obs_buf = torch.cat([eef_pos, eef_quat_wxyz], dim=-1)
            else:
                raise NotImplementedError

        if self.observe_force:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.obs_buf = torch.cat((self.obs_buf, self.vec_sensor_tensor[..., 0:3]), dim=1)

            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.obs_buf[..., -3:] = self.contact_sensor_tensor[self.handles["insertion_obj"]][..., 0:3] / 150.

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 7), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))



        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def reset(self):
        """
        Overwrite since super class reset method does reset to (0,0) and this is called initally. We do not want to start in (0,0)
        """
        env_ids = torch.tensor(list(range(self.num_envs)), device=self.device)
        self.post_physics_step()

        return super().reset()

    def _compute_osc_torques(self, dpose, kp, kd, enable_nullspace=True):
        # Solve for Operational Space Control
        # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        if enable_nullspace:
            # Nullspace control torques `u_null` prevents large changes in joint configuration
            # They are added into the nullspace of OSC so that the end effector orientation remains constant
            # roboticsproceedings.org/rss07/p31.pdf
            j_eef_inv = m_eef @ self._j_eef @ mm_inv
            u_null = self.kd_null * -qd + self.kp_null * (
                    (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
            u_null[:, 7:] *= 0
            u_null = self._mm @ u_null.unsqueeze(-1)
            u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

        return u

    def _compute_differentiable_inverse_kinematics_torques(self, dpose):
        raise NotImplementedError

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
        if self.control_type == "osc":
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
        self.actions = actions.clone().to(self.device)
        delta_pos = self.actions[..., 0:3]*0.05

        # Set (possibly learned) stiffness and damping matrices
        if self.learn_stiffness:
            kp, kv = self._create_stiffness_and_damping_matrices(actions[..., 6:12].clone())
        else:
            kp = self.kp
            kv = self.kd

        if self.learn_orientations:
            delta_orn = self.actions[..., 3:6]
        else:
            self._refresh()
            delta_orn = orientation_error(self.eef_orn_des, self.states["eef_quat"])

        if self.enable_PD_to_goal:
            signal_to_goal_pos = -self.states["eef_pos"]

            # clip first the PD signal
            velocity_norm = torch.linalg.vector_norm(signal_to_goal_pos + 1e-6, dim=1, ord=np.inf)
            scale_ratio = torch.clip(velocity_norm, 0., self.max_delta_pos) / velocity_norm
            # add PD to signal from policy
            delta_pos += scale_ratio.view(-1, 1) * signal_to_goal_pos

            # clip signal from PD
            signal_to_goal_orn = orientation_error(self.eef_orn_des, self.states["eef_quat"])
            velocity_norm = torch.linalg.vector_norm(signal_to_goal_orn + 1e-6, dim=1, ord=np.inf)
            scale_ratio = torch.clip(velocity_norm, 0., self.max_delta_orn) / velocity_norm
            delta_orn += scale_ratio.view(-1, 1) * signal_to_goal_orn

        # clip delta_pos by norm if larger than max_delta_pos
        delta_pos_norm = torch.linalg.vector_norm(delta_pos + 1e-6, dim=1, ord=np.inf)
        delta_pos_scale_ratio = torch.clip(delta_pos_norm, 0.0, self.max_delta_pos) / delta_pos_norm
        delta_pos = delta_pos_scale_ratio.view(-1, 1) * delta_pos

        # clip delta_orn by norm if larger than max_delta_orn
        delta_orn_norm = torch.linalg.vector_norm(delta_orn + 1e-6, dim=1, ord=np.inf)
        delta_orn_scale_ratio = torch.clip(delta_orn_norm, 0.0, self.max_delta_orn) / delta_orn_norm
        delta_orn = delta_orn_scale_ratio.view(-1, 1) * delta_orn

        # concatenate deltas in position and orientation
        u_arm = torch.cat([delta_pos, delta_orn], dim=-1)
        # Control arm
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm, kp=kp, kd=kv, enable_nullspace=True)
        else:
            u_arm = self._compute_differentiable_inverse_kinematics_torques(dpose=u_arm)

        self._arm_control[:, :] = u_arm

       # print("u_arm", u_arm[0])

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]

            for i in range(self.num_envs):
                env = self.envs[i]

                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(*eef_pos[i])
                pose.r = gymapi.Quat(*eef_rot[i])

                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, pose)



#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero and convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat

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
def orientation_error(desired, current):
    """
    https://studywolf.wordpress.com/2018/12/03/force-control-of-task-space-orientation/
    """
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

@torch.jit.script
def compute_insertion_reward(
        eef_pos, eef_quat, eef_pos_des, eef_quat_des, reset_buf, progress_buf, max_episode_length,
        enable_sparse_reward=False, reward_orientations=False
):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, bool, bool) -> Tuple[torch.Tensor, torch.Tensor]

    pos_error = eef_pos_des - eef_pos
    ee_pos_dist = torch.sqrt(
        pos_error[..., 0] * pos_error[..., 0] +
        pos_error[..., 1] * pos_error[..., 1] +
        pos_error[..., 2] * pos_error[..., 2]
    )
    reward = -ee_pos_dist * 3 # scale error according to scaling in observations

    orn_error = orientation_error(eef_quat_des, eef_quat)
    ee_orn_dist = torch.sqrt(
        orn_error[..., 0] * orn_error[..., 0] +
        orn_error[..., 1] * orn_error[..., 1] +
        orn_error[..., 2] * orn_error[..., 2]
    )
    if reward_orientations:
        reward -= (ee_orn_dist / np.pi) * 5  # scale the error to be in 0 to 1

    max_pos_dist = 0.025
    max_orn_dist = 3.14159 / 180. * 5. # equals 5 degrees
    condition = torch.logical_or(progress_buf >= max_episode_length - 1, torch.logical_and(torch.abs(pos_error).max(dim=1)[0] < max_pos_dist, torch.abs(orn_error).max(dim=1)[0] < max_orn_dist))

    reset = torch.where(condition, torch.ones_like(reset_buf), reset_buf)

    if enable_sparse_reward:
        reward = reward * 0 - 1

    return reward, reset
