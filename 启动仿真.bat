@echo off
title ManiSkill Demo
rem 清除残留的坏环境变量（指向不存在的 D:\anaconda，会破坏 Python 启动）
set PYTHONHOME=
set PYTHONPATH=
set MS_ASSET_DIR=D:\trae\111\maniskill_data
"D:\trae\111\miniforge3\envs\maniskill\python.exe" "D:\trae\111\run_demo.py"
pause
