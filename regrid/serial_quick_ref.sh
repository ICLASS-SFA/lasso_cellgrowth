#!/bin/bash
# Quick Reference: Serial Processing on Login Node
# =================================================

# STEP 1: Generate task list (run once)
# --------------------------------------
cd /ccsopen/home/zhe1feng1/program/lasso/cellgrowth/regrid
bash generate_serial_tasks.sh
# Creates: serial_tasks.txt (864 tasks) and run_serial_tasks.sh


# STEP 2: Start processing in background
# ----------------------------------------
nohup bash run_serial_tasks.sh > serial_processing.log 2>&1 &
# Note the PID that is displayed, e.g., [1] 12345


# STEP 3: Monitor progress (real-time)
# -------------------------------------
tail -f serial_processing.log
# Press Ctrl+C to exit (won't stop the process)


# STEP 4: Check if still running
# -------------------------------
ps aux | grep run_serial_tasks
# Or
jobs -l


# STEP 5: Check progress percentage
# ----------------------------------
COMPLETED=$(grep "completed successfully" serial_processing.log | wc -l)
echo "Progress: ${COMPLETED}/864 tasks ($(( COMPLETED * 100 / 864 ))%)"


# STEP 6: Check for failures
# ---------------------------
grep "FAILED" serial_processing.log | wc -l
grep -B 3 "FAILED" serial_processing.log


# STEP 7: Stop if needed
# -----------------------
# Find PID first
PID=$(pgrep -f run_serial_tasks)
kill $PID
# Or force kill:
# kill -9 $PID


# ========================================
# ALTERNATIVE: Using screen (recommended if you want interactive control)
# ========================================

# Start screen session
screen -S regrid_serial

# Inside screen, run normally
cd /ccsopen/home/zhe1feng1/program/lasso/cellgrowth/regrid
bash run_serial_tasks.sh

# Detach from screen: Press Ctrl+A, then D
# You can now logout safely

# Reattach later
screen -r regrid_serial

# List all screens
screen -ls


# ========================================
# ALTERNATIVE: Using tmux
# ========================================

# Start tmux session
tmux new -s regrid_serial

# Inside tmux, run normally
cd /ccsopen/home/zhe1feng1/program/lasso/cellgrowth/regrid
bash run_serial_tasks.sh

# Detach from tmux: Press Ctrl+B, then D

# Reattach later
tmux attach -t regrid_serial

# List all sessions
tmux ls


# ========================================
# USEFUL MONITORING COMMANDS
# ========================================

# View last 50 lines of log
tail -n 50 serial_processing.log

# Watch progress (updates every 2 seconds)
watch -n 2 'tail -n 20 serial_processing.log'

# Search for specific case
grep "20190123" serial_processing.log | grep "completed"

# Check resource usage
top -u $USER

# Detailed process info (replace 12345 with your PID)
ps -p 12345 -o pid,ppid,cmd,%mem,%cpu,etime


# ========================================
# RESUME AFTER INTERRUPTION
# ========================================

# Find last completed task number
LAST_COMPLETED=$(grep "completed successfully" serial_processing.log | tail -1 | grep -oP 'Task \K\d+')
echo "Last completed: Task ${LAST_COMPLETED}"

# Create resume task file (skip completed tasks)
tail -n +$((LAST_COMPLETED + 1)) serial_tasks.txt > serial_tasks_resume.txt

# Edit run_serial_tasks.sh to use serial_tasks_resume.txt
# Or run manually:
# while read CMD; do eval $CMD; done < serial_tasks_resume.txt
