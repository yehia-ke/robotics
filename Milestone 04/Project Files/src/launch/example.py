"""
Milestone 4 launch: MuJoCo lab scene (torque-controlled Lite6 + gripper), the motor-level
controller and the GUI.

    ros2 launch mujoco_ros2 example.py                 # simulator window + controller + GUI
    ros2 launch mujoco_ros2 example.py gui:=false      # no Tk GUI (use run_task.py instead)
    ros2 launch mujoco_ros2 example.py headless:=true  # no MuJoCo window (batch validation)

Data flow:
    mujoco_node (C++, 1 kHz)  --/joint_state, /ee_pose-->  controller_node.py (250 Hz)
    controller_node.py        --/joint_commands (6 torques + gripper force)-->  mujoco_node
    sliders_node.py (GUI)     --/controller/trajectory, /jog_target, /gripper, /command-->  controller_node.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    directory = get_package_share_directory('mujoco_ros2')  # Gets relative path of mujoco_ros2 package

    xmlScenePath = os.path.join(directory, 'model', 'ufactory_lite6', 'lab-scene.xml')

    if not os.path.exists(xmlScenePath):
        raise FileNotFoundError(f"Scene file does not exist: {xmlScenePath}.")

    # The Python nodes run from the source tree. Where it is was recorded at build time
    # (share/mujoco_ros2/source_dir.txt), so building from any folder works.
    source_dir_file = os.path.join(directory, 'source_dir.txt')
    candidates = []
    if os.path.exists(source_dir_file):
        with open(source_dir_file) as handle:
            candidates.append(os.path.join(handle.read().strip(), 'src', 'nodes'))
    workspace_dir = os.path.abspath(os.path.join(directory, '..', '..', '..', '..'))
    candidates.append(os.path.join(workspace_dir, 'src', 'src', 'nodes'))   # old layout
    nodes_dir = next((c for c in candidates if os.path.isdir(c)), candidates[-1])

    guiScriptPath = os.path.join(nodes_dir, 'sliders_node.py')
    controllerScriptPath = os.path.join(nodes_dir, 'controller_node.py')

    for path in (guiScriptPath, controllerScriptPath):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Script does not exist: {path}.")

    headless = DeclareLaunchArgument('headless', default_value='false',
                                     description='Run MuJoCo without a window')
    gui = DeclareLaunchArgument('gui', default_value='true',
                                description='Start the Tk GUI (sliders_node.py)')

    mujoco = Node(
        package="mujoco_ros2",
        executable="mujoco_node",
        output="screen",
        arguments=[xmlScenePath],
        parameters=[
            {"joint_state_topic_name": "joint_state"},
            {"joint_command_topic_name": "joint_commands"},
            {"control_mode": "TORQUE"},                # Milestone 4: joint torques from controller_node.py
            {"simulation_frequency": 1000},
            {"torque_steps_per_command": 4},           # lockstep: controller runs every 4 ms of sim time
            {"command_timeout": 0.5},
            {"initial_keyframe": "lab_home"},
            {"end_effector_site": "end_effector"},     # TCP between the fingertips -> /ee_pose
            {"headless": LaunchConfiguration('headless')},
            {"visualisation_frequency": 30},
            {"camera_focal_point": [0.0, 0.0, 0.15]},
            {"camera_distance": 1.6},
            {"camera_azimuth": 135.0},
            {"camera_elevation": -30.0},
            {"camera_orthographic": False}
        ]
    )

    controller = ExecuteProcess(
        cmd=['python3', controllerScriptPath,
             '--ros-args',
             '-p', 'control_period:=0.004'],
        output='screen'
    )

    sliders = ExecuteProcess(
        cmd=['python3', guiScriptPath,
             '--ros-args',
             '-p', f'model_path:={xmlScenePath}'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('gui'))
    )

    return LaunchDescription([headless, gui, mujoco, controller, sliders])
