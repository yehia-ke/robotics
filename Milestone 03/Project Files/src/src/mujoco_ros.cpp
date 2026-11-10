
/**
 * @file   mujoco_ros.cpp
 * @author Jon Woolfrey
 * @email  jonathan.woolfrey@gmail.com
 * @date   April 2025
 * @version 1.1
 * @brief  A class for connecting a MuJoCo simulation with ROS2 communication.
 *
 * @details This class launches a MuJoCo simulation and provides communication
 * channels in ROS2 for controlling it.
 *
 * @copyright Copyright (c) 2025 Jon Woolfrey
 * @license GNU General Public License V3
 */

#include <mujoco_ros2/mujoco_ros.hpp>

#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

namespace
{
    /**
     * Returns the joint ID associated with an actuator.
     *
     * This implementation supports actuators transmitted through scalar
     * joints, such as the Lite6 hinge joints.
     */
    int get_actuator_joint_id(const mjModel* model, int actuator_id)
    {
        if (model == nullptr)
        {
            throw std::runtime_error("MuJoCo model is null.");
        }

        if (actuator_id < 0 || actuator_id >= model->nu)
        {
            throw std::out_of_range("Invalid actuator ID.");
        }

        const int transmission_type = model->actuator_trntype[actuator_id];

        if (transmission_type != mjTRN_JOINT &&
            transmission_type != mjTRN_JOINTINPARENT)
        {
            throw std::runtime_error(
                "Unsupported actuator transmission. Expected a joint transmission.");
        }

        const int joint_id = model->actuator_trnid[2 * actuator_id];

        if (joint_id < 0 || joint_id >= model->njnt)
        {
            throw std::runtime_error(
                "Actuator is associated with an invalid joint.");
        }

        const int joint_type = model->jnt_type[joint_id];

        if (joint_type != mjJNT_HINGE && joint_type != mjJNT_SLIDE)
        {
            throw std::runtime_error(
                "Actuator must be associated with a scalar hinge or slide joint.");
        }

        return joint_id;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                         Constructor                                            //
////////////////////////////////////////////////////////////////////////////////////////////////////

MuJoCoROS::MuJoCoROS(const std::string &xmlLocation)
    : Node("mujoco_node")
{
    // Declare and get ROS parameters.

    const std::string jointStateTopicName =
        this->declare_parameter<std::string>(
            "joint_state_topic_name", "joint_state");

    const std::string jointCommandTopicName =
        this->declare_parameter<std::string>(
            "joint_command_topic_name", "joint_commands");

    const std::string controlMode =
        this->declare_parameter<std::string>(
            "control_mode", "TORQUE");

    _simFrequency =
        this->declare_parameter<int>("simulation_frequency", 1000);

    const int visualisationFrequency =
        this->declare_parameter<int>("visualisation_frequency", 20);

    if (_simFrequency <= 0)
    {
        throw std::invalid_argument(
            "Simulation frequency must be greater than zero.");
    }

    if (visualisationFrequency <= 0)
    {
        throw std::invalid_argument(
            "Visualisation frequency must be greater than zero.");
    }

    // Load the MuJoCo model.

    char errorMessage[1000] = "Could not load model.";

    _model = mj_loadXML(
        xmlLocation.c_str(),
        nullptr,
        errorMessage,
        sizeof(errorMessage));

    if (_model == nullptr)
    {
        throw std::runtime_error(
            "[MuJoCo NODE] Problem loading model: " +
            std::string(errorMessage));
    }

    _model->opt.timestep = 1.0 / static_cast<double>(_simFrequency);

    // Initialize simulation data.

    _jointState = mj_makeData(_model);

    if (_jointState == nullptr)
    {
        mj_deleteModel(_model);
        _model = nullptr;

        throw std::runtime_error(
            "[MuJoCo NODE] Failed to allocate MuJoCo simulation data.");
    }

    /*
     * IMPORTANT:
     *
     * nq = number of position coordinates.
     * nv = number of velocity coordinates.
     * njnt = number of joints.
     * nu = number of actuators.
     *
     * The six free-joint boxes increase nq and nv, but do not add actuators.
     *
     * The ROS interface therefore publishes and accepts one value per
     * actuator, using the joint associated with each actuator.
     */

    const int actuatorCount = _model->nu;

    if (actuatorCount <= 0)
    {
        throw std::runtime_error(
            "[MuJoCo NODE] The model does not contain any actuators.");
    }

    _jointStateMessage.name.resize(actuatorCount);
    _jointStateMessage.position.resize(actuatorCount);
    _jointStateMessage.velocity.resize(actuatorCount);
    _jointStateMessage.effort.resize(actuatorCount);

    _torqueInput.assign(actuatorCount, 0.0);

    // Record the joint name associated with each actuator.

    for (int i = 0; i < actuatorCount; ++i)
    {
        const int joint_id = get_actuator_joint_id(_model, i);

        const char* jointName =
            mj_id2name(_model, mjOBJ_JOINT, joint_id);

        if (jointName == nullptr)
        {
            throw std::runtime_error(
                "[MuJoCo NODE] An actuator's joint does not have a name.");
        }

        _jointStateMessage.name[i] = jointName;
    }

    // Set the control mode.

    if (controlMode == "POSITION")
    {
        _controlMode = POSITION;
    }
    else if (controlMode == "VELOCITY")
    {
        _controlMode = VELOCITY;
    }
    else if (controlMode == "TORQUE")
    {
        _controlMode = TORQUE;
    }
    else
    {
        throw std::invalid_argument(
            "[MuJoCo NODE] Unknown control mode. "
            "Expected POSITION, VELOCITY, or TORQUE. Received: " +
            controlMode);
    }

    // Create simulation and visualization timers.

    _simTimer = this->create_wall_timer(
        std::chrono::duration<double>(
            1.0 / static_cast<double>(_simFrequency)),
        std::bind(&MuJoCoROS::update_simulation, this));

    _visTimer = this->create_wall_timer(
        std::chrono::duration<double>(
            1.0 / static_cast<double>(visualisationFrequency)),
        std::bind(&MuJoCoROS::update_visualization, this));

    // Create joint state publisher and command subscriber.

    _jointCommandSubscriber =
        this->create_subscription<std_msgs::msg::Float64MultiArray>(
            jointCommandTopicName,
            1,
            std::bind(
                &MuJoCoROS::joint_command_callback,
                this,
                std::placeholders::_1));

    _jointStatePublisher =
        this->create_publisher<sensor_msgs::msg::JointState>(
            jointStateTopicName, 1);

    // Initialize GLFW.

    if (!glfwInit())
    {
        throw std::runtime_error(
            "[MuJoCo NODE] Failed to initialize GLFW.");
    }

    _window = glfwCreateWindow(
        1200,
        900,
        "MuJoCo Visualization",
        nullptr,
        nullptr);

    if (_window == nullptr)
    {
        glfwTerminate();

        throw std::runtime_error(
            "[MuJoCo NODE] Failed to create GLFW window.");
    }

    // Make the OpenGL context current.

    glfwMakeContextCurrent(_window);
    glfwSwapInterval(1);

    // Initialize MuJoCo rendering structures.

    mjv_defaultCamera(&_camera);
    mjv_defaultOption(&_renderingOptions);
    mjv_defaultPerturb(&_perturbation);
    mjr_defaultContext(&_context);

    mjv_makeScene(_model, &_scene, 1000);

    // Declare and retrieve camera parameters.

    _camera.azimuth =
        this->declare_parameter<double>("camera_azimuth", 135);

    _camera.distance =
        this->declare_parameter<double>("camera_distance", 2.5);

    _camera.elevation =
        this->declare_parameter<double>("camera_elevation", -35);

    _camera.orthographic =
        this->declare_parameter<bool>("camera_orthographic", true);

    const auto focalPoint =
        this->declare_parameter<std::vector<double>>(
            "camera_focal_point",
            {0.0, 0.0, 0.5});

    if (focalPoint.size() != 3)
    {
        throw std::invalid_argument(
            "[MuJoCo NODE] camera_focal_point must contain exactly 3 values.");
    }

    for (int i = 0; i < 3; ++i)
    {
        _camera.lookat[i] = focalPoint[i];
    }

    // Create the MuJoCo rendering context.

    glfwMakeContextCurrent(_window);

    mjr_makeContext(
        _model,
        &_context,
        mjFONTSCALE_100);

    RCLCPP_INFO(
        this->get_logger(),
        "MuJoCo simulation initiated. "
        "Publishing joint states to '%s'. "
        "Subscribing to %s commands via '%s'. "
        "Model dimensions: nq=%d, nv=%d, njnt=%d, nu=%d.",
        jointStateTopicName.c_str(),
        controlMode.c_str(),
        jointCommandTopicName.c_str(),
        _model->nq,
        _model->nv,
        _model->njnt,
        _model->nu);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                    Update the simulation                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

void MuJoCoROS::update_simulation()
{
    if (_model == nullptr || _jointState == nullptr)
    {
        RCLCPP_ERROR(
            this->get_logger(),
            "MuJoCo model or simulation data is not initialized.");

        return;
    }

    // Apply torque commands.

    if (_controlMode == TORQUE)
    {
        for (int i = 0; i < _model->nu; ++i)
        {
            const int joint_id = get_actuator_joint_id(_model, i);

            const int dofAddress =
                _model->jnt_dofadr[joint_id];

            /*
             * ctrl is indexed by actuator, not by joint DOF.
             *
             * qfrc_bias and qvel are indexed by generalized velocity DOF.
             *
             * This mapping is essential when free joints are present.
             */

            _jointState->ctrl[i] =
                _torqueInput[i]
                + _jointState->qfrc_bias[dofAddress]
                - 0.01 * _jointState->qvel[dofAddress];

            _torqueInput[i] = 0.0;
        }
    }

    // Advance the MuJoCo simulation.

    mj_step(_model, _jointState);

    // Publish state for actuated joints only.

    for (int i = 0; i < _model->nu; ++i)
    {
        const int joint_id = get_actuator_joint_id(_model, i);

        const int qposAddress =
            _model->jnt_qposadr[joint_id];

        const int dofAddress =
            _model->jnt_dofadr[joint_id];

        _jointStateMessage.position[i] =
            _jointState->qpos[qposAddress];

        _jointStateMessage.velocity[i] =
            _jointState->qvel[dofAddress];

        _jointStateMessage.effort[i] =
            _jointState->actuator_force[i];
    }

    _jointStateMessage.header.stamp =
        this->get_clock()->now();

    _jointStatePublisher->publish(_jointStateMessage);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                    Handle joint commands                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

void MuJoCoROS::joint_command_callback(
    const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (_model == nullptr || _jointState == nullptr)
    {
        RCLCPP_ERROR(
            this->get_logger(),
            "Cannot process commands: MuJoCo is not initialized.");

        return;
    }

    // Expect exactly one command per actuator.

    if (msg->data.size() != static_cast<size_t>(_model->nu))
    {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            5000,
            "Received %zu commands, but the model expects %d actuator commands.",
            msg->data.size(),
            _model->nu);

        return;
    }

    switch (_controlMode)
    {
        case POSITION:
        {
            // Position actuator targets.

            for (int i = 0; i < _model->nu; ++i)
            {
                _jointState->ctrl[i] = msg->data[i];
            }

            break;
        }

        case VELOCITY:
        {
            /*
             * Preserve the original velocity-control behavior:
             * integrate velocity commands into position actuator targets.
             */

            for (int i = 0; i < _model->nu; ++i)
            {
                _jointState->ctrl[i] +=
                    msg->data[i] / static_cast<double>(_simFrequency);
            }

            break;
        }

        case TORQUE:
        {
            // Store one torque command per actuator.

            for (int i = 0; i < _model->nu; ++i)
            {
                _torqueInput[i] = msg->data[i];
            }

            break;
        }

        default:
        {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                5000,
                "Unknown control mode. Unable to set joint commands.");

            break;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                    Update the 3D simulation                                    //
////////////////////////////////////////////////////////////////////////////////////////////////////

void MuJoCoROS::update_visualization()
{
    if (_model == nullptr ||
        _jointState == nullptr ||
        _window == nullptr)
    {
        return;
    }

    glfwMakeContextCurrent(_window);

    mjv_updateScene(
        _model,
        _jointState,
        &_renderingOptions,
        nullptr,
        &_camera,
        mjCAT_ALL,
        &_scene);

    // Get framebuffer dimensions.

    int width = 0;
    int height = 0;

    glfwGetFramebufferSize(_window, &width, &height);

    if (width <= 0 || height <= 0)
    {
        glfwPollEvents();
        return;
    }

    const mjrRect viewport = {0, 0, width, height};

    // Render the scene.

    mjr_render(
        viewport,
        &_scene,
        &_context);

    // Swap buffers and process events.

    glfwSwapBuffers(_window);
    glfwPollEvents();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                           Destructor                                           //
////////////////////////////////////////////////////////////////////////////////////////////////////

MuJoCoROS::~MuJoCoROS()
{
    // Release MuJoCo simulation data.

    if (_jointState != nullptr)
    {
        mj_deleteData(_jointState);
        _jointState = nullptr;
    }

    if (_model != nullptr)
    {
        mj_deleteModel(_model);
        _model = nullptr;
    }

    // Release rendering resources.

    mjv_freeScene(&_scene);
    mjr_freeContext(&_context);

    // Release GLFW resources.

    if (_window != nullptr)
    {
        glfwDestroyWindow(_window);
        _window = nullptr;
    }

    glfwTerminate();
}
