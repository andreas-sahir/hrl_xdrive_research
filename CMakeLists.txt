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
  scripts/omni_controller.py
  scripts/odometry_publisher.py
  scripts/error_calculation.py
  scripts/omni_controller_nopid.py
  scripts/motion_test.py
  scripts/omni_rl_env.py
  scripts/tracker_rl.py
  scripts/train_sac_tracker.py
  scripts/train_ppo.py
  scripts/evaluate_rl.py
  scripts/evaluate_goal.py
  scripts/test_nopid.py
  scripts/ddqn_train.py
  scripts/ddqn_env.py
  scripts/Low_level_rl.py
  scripts/evaluate_ll.py
  scripts/evaluate_ll_prev.py
  scripts/ppo_ll_prev.py
  scripts/hrl_env.py
  scripts/hrl_env123.py
  scripts/train_hl_sac.py
  scripts/obstacle_move.py
  scripts/obstacle_move2.py
  scripts/evaluate_hrl.py
  scripts/evaluate_hrl123.py
  scripts/hrl_nav2_bridge.py
  scripts/auto_explorer.py
  scripts/test_crash.py
  scripts/dummy_map.py

  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()