"""Configuration for dexterous hand from Contactile.
The following configurations are available:
* :obj: 'CONTACTILE_HAND_CFG': Contactile Hand with implicit actuator model.
"""
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
##
# Configuration
##
usd_path = os.path.join(
    os.path.dirname(__file__),
    "contactile_hand_usd",
    "contactile_hand.usd"
)
usd_path = os.path.abspath(usd_path)
CONTACTILE_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            max_depenetration_velocity=100.0,
            max_linear_velocity=10.0,
            max_angular_velocity=100.0
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.001,
            stabilization_threshold=0.001,
            fix_root_link=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=10.0, damping=0.5),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        rot=(0.7071, 0.0, 0.0, 0.7071),
        joint_pos={
            "thumb_joint_1": -1.57,   
            "thumb_joint_2": 0.0,
            "index_joint_1": 0.0,
            "index_joint_2": 0.0,
            "middle_joint_1": 0.0,
            "middle_joint_2": 0.0,
            "ring_joint_1": 0.0,
            "ring_joint_2": 0.0,
            "pinky_joint_1": 0.0,
            "pinky_joint_2": 0.0,
            },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                "thumb_joint_.*",
                "index_joint_.*",
                "middle_joint_.*",
                "ring_joint_.*",
                "pinky_joint_.*"
            ],
            effort_limit_sim={
                "thumb_joint_1": 5.0,
                "thumb_joint_2": 5.0,
                "index_joint_1": 5.0,
                "index_joint_2": 5.0,
                "middle_joint_1": 5.0,
                "middle_joint_2": 5.0,
                "ring_joint_1": 5.0,
                "ring_joint_2": 5.0,
                "pinky_joint_1": 5.0,
                "pinky_joint_2": 5.0,
            },
            stiffness={
                "thumb_joint_1": 20.0,
                "thumb_joint_2": 20.0,
                "index_joint_1": 20.0,
                "index_joint_2": 15.0,
                "middle_joint_1": 20.0,
                "middle_joint_2": 15.0,
                "ring_joint_1": 20.0,
                "ring_joint_2": 15.0,
                "pinky_joint_1": 20.0,
                "pinky_joint_2": 15.0,

            },
            damping={
                "thumb_joint_1": 0.5,  
                "thumb_joint_2": 0.5,
                "index_joint_1": 0.5,
                "index_joint_2": 0.4,
                "middle_joint_1": 0.5,
                "middle_joint_2": 0.4,
                "ring_joint_1": 0.5,
                "ring_joint_2": 0.4,
                "pinky_joint_1": 0.5,
                "pinky_joint_2": 0.4,

            },
            velocity_limit={
                ".*": 100.0,
            }
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of Contactile Hand robot"""