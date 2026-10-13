import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    directory = get_package_share_directory('mujoco_ros2')  # Gets relative path of mujoco_ros2 package

    xmlScenePath = os.path.join(directory, 'model', 'ufactory_lite6', 'scene.xml')

    if not os.path.exists(xmlScenePath):
        raise FileNotFoundError(f"Scene file does not exist: {xmlScenePath}.")

    workspace_dir = os.path.abspath(
        os.path.join(directory, '..', '..', '..', '..')
    )

    sliderScriptPath = os.path.join(
        workspace_dir, 'src', 'src', 'nodes', 'sliders_node.py'
    )

    if not os.path.exists(sliderScriptPath):
        raise FileNotFoundError(f"Slider script does not exist: {sliderScriptPath}.")

    mujoco = Node(
        package="mujoco_ros2",
        executable="mujoco_node",
        output="screen",
        arguments=[xmlScenePath],
        parameters=[
            {"joint_state_topic_name": "joint_state"},
            {"joint_command_topic_name": "joint_commands"},
            {"control_mode": "POSITION"},
            {"simulation_frequency": 1000},
            {"visualisation_frequency": 20},
            {"camera_focal_point": [0.0, 0.0, 0.25]},
            {"camera_distance": 2.5},
            {"camera_azimuth": 135.0},
            {"camera_elevation": -20.0},
            {"camera_orthographic": True}
        ]
    )

    sliders = ExecuteProcess(
        cmd=['python3', sliderScriptPath,
             '--ros-args',
             '-p', 'mode:=position',
             '-p', f'model_path:={xmlScenePath}'],
        output='screen'
    )

    return LaunchDescription([mujoco, sliders])