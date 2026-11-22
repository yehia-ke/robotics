
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

    // Milestone 4: optionally start from a keyframe (e.g. "lab_home": arm folded, labware at rest).

    const std::string initialKeyframe =
        this->declare_parameter<std::string>("initial_keyframe", "");

    if (!initialKeyframe.empty())
    {
        const int keyId = mj_name2id(_model, mjOBJ_KEY, initialKeyframe.c_str());

        if (keyId < 0)
        {
            throw std::invalid_argument(
                "[MuJoCo NODE] Keyframe '" + initialKeyframe + "' not found in the model.");
        }

        mj_resetDataKeyframe(_model, _jointState, keyId);
    }

    mj_forward(_model, _jointState);

    // Milestone 4: torque-mode options.
    //  torque_steps_per_command > 0 runs the simulation in LOCKSTEP with the controller:
    //  after that many steps the simulation waits for the next torque command, so the
    //  controller's computation / ROS latency never shows up as extra delay in the loop.
    //  command_timeout: wall seconds without a command before the arm is put in safe hold.

    _stepsPerCommand =
        this->declare_parameter<int>("torque_steps_per_command", 0);

    _commandTimeout =
        this->declare_parameter<double>("command_timeout", 0.5);

    const std::string endEffectorSite =
        this->declare_parameter<std::string>("end_effector_site", "end_effector");

    const std::string endEffectorTopicName =
        this->declare_parameter<std::string>("ee_pose_topic_name", "ee_pose");

    _endEffectorSite = mj_name2id(_model, mjOBJ_SITE, endEffectorSite.c_str());

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

    // headless = true: no window (batch validation runs / machines without a display).
    const bool headless = this->declare_parameter<bool>("headless", false);

    if (!headless)
    {
        _visTimer = this->create_wall_timer(
            std::chrono::duration<double>(
                1.0 / static_cast<double>(visualisationFrequency)),
            std::bind(&MuJoCoROS::update_visualization, this));
    }

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
            jointStateTopicName, 10);

    if (_endEffectorSite >= 0)
    {
        _endEffectorPublisher =
            this->create_publisher<geometry_msgs::msg::PoseStamped>(
                endEffectorTopicName, 10);
    }

    _lastCommandWallTime = _wallClock.now();
    _lastRepublishWallTime = _wallClock.now();
    enter_safe_hold();

    _window = nullptr;
    mjv_defaultScene(&_scene);
    mjr_defaultContext(&_context);

    if (!headless)
    {
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

        // Milestone 4: mouse = move the camera, C = cycle through the scene's fixed cameras.
        glfwSetWindowUserPointer(_window, this);
        glfwSetMouseButtonCallback(_window, MuJoCoROS::mouse_button_callback);
        glfwSetCursorPosCallback(_window, MuJoCoROS::cursor_position_callback);
        glfwSetScrollCallback(_window, MuJoCoROS::scroll_callback);
        glfwSetKeyCallback(_window, MuJoCoROS::key_callback);
    }

    RCLCPP_INFO(
        this->get_logger(),
        "MuJoCo simulation initiated. "
        "Publishing joint states to '%s'. "
        "Subscribing to %s commands via '%s'. "
        "Model dimensions: nq=%d, nv=%d, njnt=%d, nu=%d.",
        jointStateTopicName.c_str(),
        controlMode.c_str(),
        jointCommandTopicName.c_str(),
        static_cast<int>(_model->nq),
        static_cast<int>(_model->nv),
        static_cast<int>(_model->njnt),
        static_cast<int>(_model->nu));

    if (_controlMode == TORQUE)
    {
        RCLCPP_INFO(
            this->get_logger(),
            "TORQUE mode: commands are held until the next one arrives; %s. "
            "The arm is held in place (safe hold) until the first command.",
            _stepsPerCommand > 0
                ? ("lockstep with the controller, " + std::to_string(_stepsPerCommand) +
                   " steps per command").c_str()
                : "free running");
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//          Camera control (Milestone 4): left drag rotate, right drag pan, scroll zoom, C camera   //
////////////////////////////////////////////////////////////////////////////////////////////////////

void MuJoCoROS::mouse_button_callback(GLFWwindow *window, int, int, int)
{
    auto *self = static_cast<MuJoCoROS *>(glfwGetWindowUserPointer(window));
    self->_buttonLeft = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    self->_buttonRight = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    glfwGetCursorPos(window, &self->_lastX, &self->_lastY);
}

void MuJoCoROS::cursor_position_callback(GLFWwindow *window, double x, double y)
{
    auto *self = static_cast<MuJoCoROS *>(glfwGetWindowUserPointer(window));
    const double dx = x - self->_lastX;
    const double dy = y - self->_lastY;
    self->_lastX = x;
    self->_lastY = y;

    if (!self->_buttonLeft && !self->_buttonRight)
    {
        return;
    }

    if (self->_camera.type != mjCAMERA_FREE)                                                        // leave a fixed camera
    {
        self->_camera.type = mjCAMERA_FREE;
        self->_cameraIndex = -1;
    }

    int width = 0, height = 0;
    glfwGetWindowSize(window, &width, &height);
    const bool shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
    const mjtMouse action = self->_buttonRight ? (shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V)
                                               : (shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V);

    mjv_moveCamera(self->_model, action, dx / height, dy / height, &self->_camera);
}

void MuJoCoROS::scroll_callback(GLFWwindow *window, double, double dy)
{
    auto *self = static_cast<MuJoCoROS *>(glfwGetWindowUserPointer(window));

    if (self->_camera.type != mjCAMERA_FREE)
    {
        return;
    }

    mjv_moveCamera(self->_model, mjMOUSE_ZOOM, 0.0, -0.05 * dy, &self->_camera);
}

void MuJoCoROS::key_callback(GLFWwindow *window, int key, int, int action, int)
{
    auto *self = static_cast<MuJoCoROS *>(glfwGetWindowUserPointer(window));

    if (action != GLFW_PRESS || key != GLFW_KEY_C)
    {
        return;
    }

    // free camera -> camera 0 -> camera 1 -> ... -> free camera
    self->_cameraIndex = (self->_cameraIndex + 1 < self->_model->ncam) ? self->_cameraIndex + 1 : -1;

    if (self->_cameraIndex < 0)
    {
        self->_camera.type = mjCAMERA_FREE;
    }
    else
    {
        self->_camera.type = mjCAMERA_FIXED;
        self->_camera.fixedcamid = self->_cameraIndex;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                         Safe hold (Milestone 4): no controller connected                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

void MuJoCoROS::enter_safe_hold()
{
    _safeHold = true;
    _holdPosition.assign(_model->nu, 0.0);

    for (int i = 0; i < _model->nu; ++i)
    {
        const int joint_id = get_actuator_joint_id(_model, i);
        _holdPosition[i] = _jointState->qpos[_model->jnt_qposadr[joint_id]];
    }
}

void MuJoCoROS::apply_safe_hold()
{
    /*
     * Keeps the arm still while no controller is running (start-up, or the controller
     * stopped sending commands): gravity/Coriolis compensation from MuJoCo + a stiff PD
     * on the held angles. Not used while the Python controller is active.
     */
    for (int i = 0; i < _model->nu; ++i)
    {
        const int joint_id = get_actuator_joint_id(_model, i);

        if (_model->jnt_type[joint_id] != mjJNT_HINGE)
        {
            _jointState->ctrl[i] = _torqueInput[i];                                                 // gripper: keep last force
            continue;
        }

        const int qposAddress = _model->jnt_qposadr[joint_id];
        const int dofAddress = _model->jnt_dofadr[joint_id];

        _jointState->ctrl[i] =
            _jointState->qfrc_bias[dofAddress]
            + 100.0 * (_holdPosition[i] - _jointState->qpos[qposAddress])
            - 10.0 * _jointState->qvel[dofAddress];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                     Publish joint state + TCP pose, stamped with SIMULATION time               //
////////////////////////////////////////////////////////////////////////////////////////////////////

void MuJoCoROS::publish_state()
{
    for (int i = 0; i < _model->nu; ++i)
    {
        const int joint_id = get_actuator_joint_id(_model, i);

        _jointStateMessage.position[i] = _jointState->qpos[_model->jnt_qposadr[joint_id]];
        _jointStateMessage.velocity[i] = _jointState->qvel[_model->jnt_dofadr[joint_id]];
        _jointStateMessage.effort[i] = _jointState->actuator_force[i];
    }

    // Simulation time (not wall time): the controller and the validation logs use it as
    // their clock, so results do not depend on how fast the computer runs the simulation.
    const rclcpp::Time simTime(static_cast<int64_t>(_jointState->time * 1e9));

    _jointStateMessage.header.stamp = simTime;
    _jointStatePublisher->publish(_jointStateMessage);

    if (_endEffectorPublisher)
    {
        // After mj_step, site_xpos still belongs to the state BEFORE the integration step
        // (mj_step = forward dynamics, then integrate qpos). Recompute the kinematics so the
        // published TCP pose matches the published joint angles exactly.
        mj_kinematics(_model, _jointState);

        geometry_msgs::msg::PoseStamped pose;
        pose.header.stamp = simTime;
        pose.header.frame_id = "world";

        const mjtNum *position = _jointState->site_xpos + 3 * _endEffectorSite;
        mjtNum quaternion[4];
        mju_mat2Quat(quaternion, _jointState->site_xmat + 9 * _endEffectorSite);

        pose.pose.position.x = position[0];
        pose.pose.position.y = position[1];
        pose.pose.position.z = position[2];
        pose.pose.orientation.w = quaternion[0];
        pose.pose.orientation.x = quaternion[1];
        pose.pose.orientation.y = quaternion[2];
        pose.pose.orientation.z = quaternion[3];

        _endEffectorPublisher->publish(pose);
    }
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

    // Apply torque commands (Milestone 4).

    if (_controlMode == TORQUE)
    {
        const rclcpp::Time now = _wallClock.now();

        if (_commandReceived && !_safeHold &&
            (now - _lastCommandWallTime).seconds() > _commandTimeout)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "No torque command for %.2f s: controller stopped? Holding the arm in place.",
                _commandTimeout);

            enter_safe_hold();
        }

        if (_safeHold)
        {
            apply_safe_hold();
        }
        else
        {
            if (_stepsPerCommand > 0 && _stepsSinceCommand >= _stepsPerCommand)
            {
                // Lockstep: wait (do not step) for the controller's next command.
                // Re-publish the state now and then in case the last message was lost.
                if ((now - _lastRepublishWallTime).seconds() > 0.02)
                {
                    publish_state();
                    _lastRepublishWallTime = now;
                }

                return;
            }

            // ctrl is indexed by actuator. The commanded torques are applied as they are
            // (no hidden gravity compensation: the Python controller does it) and held
            // until the next command arrives (zero-order hold).
            for (int i = 0; i < _model->nu; ++i)
            {
                _jointState->ctrl[i] = _torqueInput[i];
            }
        }
    }

    // Advance the MuJoCo simulation.

    mj_step(_model, _jointState);
    ++_stepsSinceCommand;

    // Publish state for actuated joints only (+ the TCP pose).

    publish_state();
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
            static_cast<int>(_model->nu));

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
            // Store one torque command per actuator (N*m for the arm, N for the gripper).

            for (int i = 0; i < _model->nu; ++i)
            {
                _torqueInput[i] = msg->data[i];
            }

            if (_safeHold)
            {
                RCLCPP_INFO(this->get_logger(), "Torque controller connected.");
            }

            _commandReceived = true;
            _safeHold = false;
            _stepsSinceCommand = 0;
            _lastCommandWallTime = _wallClock.now();

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
