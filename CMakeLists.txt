cmake_minimum_required(VERSION 3.8)
project(my_robot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# Install all necessary directories
install(DIRECTORY
  launch
  urdf
  rviz
  meshes
  config
  worlds
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  robot_control/omni_controller.py
  robot_control/odometry_publisher.py
  robot_control/omni_controller_nopid.py
  navigations/error_calculation.py
  rl/environments/continous_tracker_env.py
  rl/environments/ddqn_env.py
  rl/environments/hrl_env.py
  rl/environments/hrl_env123.py
  rl/environments/omni_rl_env.py
  rl/environments/velocity_tracking_env.py
  rl/evaluation/evaluate_dqn.py
  rl/evaluation/evaluate_goal.py
  rl/evaluation/evaluate_hrl.py
  rl/evaluation/evaluate_hrl123.py
  rl/evaluation/evaluate_ll.py
  rl/evaluation/evaluate_rl.py
  rl/training/ddqn_train.py
  rl/training/train_hl_sac.py
  rl/training/train_ppo.py
  rl/training/train_sac_tracker.py
  simulation_tools/obstacles/obstacle_move.py
  simulation_tools/obstacles/obstacle_move2.py
  tests/test_nopid.py

  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()