# Serial Processing Guide for Login Node

When SLURM queue is full or unavailable, you can run the regridding tasks serially on a login node in the background.

## Quick Start

### 1. Generate Task List
```bash
cd /ccsopen/home/zhe1feng1/program/lasso/cellgrowth/regrid
bash generate_serial_tasks.sh
```

This creates:
- `serial_tasks.txt` - List of all commands to run (864 tasks)
- `run_serial_tasks.sh` - Script to execute all tasks sequentially

### 2. Run in Background with nohup
```bash
# Start processing in background
nohup bash run_serial_tasks.sh > serial_processing.log 2>&1 &

# Note the process ID that is displayed
```

### 3. Monitor Progress
```bash
# View live log (Ctrl+C to exit, won't stop the process)
tail -f serial_processing.log

# Check last 50 lines
tail -n 50 serial_processing.log

# Search for failures
grep "FAILED" serial_processing.log

# Count completed tasks
grep "completed successfully" serial_processing.log | wc -l
```

### 4. Check if Still Running
```bash
# Find the process
ps aux | grep run_serial_tasks

# Or check by job control
jobs -l

# Monitor system usage
top -u $USER
```

---

## Detailed Usage

### Understanding nohup

**nohup** (no hang up) makes your command immune to hangups, which means:
- ✅ Process continues after you logout
- ✅ Process survives terminal closure
- ✅ Output redirected to file
- ✅ Can be monitored remotely

**Syntax**:
```bash
nohup COMMAND > output.log 2>&1 &
```

**Explanation**:
- `nohup` - Ignore hangup signals
- `COMMAND` - Your script or command
- `> output.log` - Redirect stdout to file
- `2>&1` - Redirect stderr to stdout (combine all output)
- `&` - Run in background

### Starting the Job

**Method 1: Simple nohup (recommended)**
```bash
cd /ccsopen/home/zhe1feng1/program/lasso/cellgrowth/regrid
nohup bash run_serial_tasks.sh > serial_processing.log 2>&1 &
```

**Method 2: With screen (alternative)**
```bash
# Start screen session
screen -S regrid_serial

# Inside screen, run normally
bash run_serial_tasks.sh

# Detach: Press Ctrl+A, then D
# Reattach later: screen -r regrid_serial
```

**Method 3: With tmux (alternative)**
```bash
# Start tmux session
tmux new -s regrid_serial

# Inside tmux, run normally
bash run_serial_tasks.sh

# Detach: Press Ctrl+B, then D
# Reattach later: tmux attach -t regrid_serial
```

### Finding Your Background Job

After starting with nohup, note the **Process ID (PID)**:
```bash
[1] 12345
```

**To find it later**:
```bash
# By script name
ps aux | grep run_serial_tasks

# By your username
ps -u $USER | grep bash

# With full command
ps -ef | grep run_serial_tasks
```

**Output example**:
```
zhe1feng1  12345  0.0  0.0  115420  2048 ?  S  10:30  0:00 bash run_serial_tasks.sh
```

### Stopping the Job

**If you need to stop it**:
```bash
# Find PID first
ps aux | grep run_serial_tasks

# Kill gracefully
kill 12345

# Force kill if needed
kill -9 12345
```

---

## Monitoring and Diagnostics

### Real-time Monitoring

**Watch log file grow**:
```bash
tail -f serial_processing.log
```

**Monitor with updates every 2 seconds**:
```bash
watch -n 2 'tail -n 20 serial_processing.log'
```

**Check progress percentage**:
```bash
TOTAL=864
COMPLETED=$(grep "completed successfully" serial_processing.log | wc -l)
echo "Progress: ${COMPLETED}/${TOTAL} ($(( COMPLETED * 100 / TOTAL ))%)"
```

### Checking Failures

```bash
# Find all failures
grep "FAILED" serial_processing.log

# Count failures
grep "FAILED" serial_processing.log | wc -l

# Get failed task details
grep -B 5 "FAILED" serial_processing.log
```

### Resource Monitoring

```bash
# CPU and memory usage
top -u $USER

# Detailed process info
ps -p 12345 -o pid,ppid,cmd,%mem,%cpu,etime

# Check I/O
iotop -u $USER  # May need sudo
```

---

## Task List Structure

The `serial_tasks.txt` file contains commands like:
```bash
cd /path/to/regrid && python regrid_lasso_batch.py --case-date 20181129 --ensemble gefs00 --start-hour 12 --end-hour 12 --file-type methamsl --domain d3 --n-workers 1
cd /path/to/regrid && python regrid_lasso_batch.py --case-date 20181129 --ensemble gefs00 --start-hour 13 --end-hour 13 --file-type methamsl --domain d3 --n-workers 1
...
```

**Total tasks**: 18 cases × 2 file types × 2 domains × 12 hours = **864 tasks**

**Estimated time**: 
- ~2-5 minutes per task (serial mode)
- Total: 29-72 hours for all tasks

---

## Customization

### Process Subset of Cases

Edit `generate_serial_tasks.sh` to process only specific cases:

```bash
# Example: Only process one case
CASE_DATES=("20190123")
ENSEMBLES=("gefs18")

# Example: Only process one domain
DOMAINS=("d4")

# Example: Only process one file type
FILE_TYPES=("methamsl")
```

### Resume After Interruption

If interrupted, you can create a resume script:

```bash
# Find last completed task
LAST_COMPLETED=$(grep "completed successfully" serial_processing.log | tail -1)

# Extract task number
LAST_NUM=$(echo "$LAST_COMPLETED" | grep -oP 'Task \K\d+')

# Create new task file starting from next task
tail -n +$((LAST_NUM + 1)) serial_tasks.txt > serial_tasks_resume.txt

# Run remaining tasks
nohup bash run_serial_tasks.sh > serial_processing_resume.log 2>&1 &
```

---

## Troubleshooting

### Problem: "nohup: ignoring input and appending output to 'nohup.out'"
**Solution**: You forgot output redirection. Use:
```bash
nohup bash run_serial_tasks.sh > serial_processing.log 2>&1 &
```

### Problem: Process stops when I logout
**Solution**: Make sure you used `nohup` and `&`:
```bash
nohup bash run_serial_tasks.sh > serial_processing.log 2>&1 &
```

### Problem: Can't find the process
**Solution**: Check all variations:
```bash
ps aux | grep -E "run_serial|regrid_lasso_batch"
pgrep -af python
```

### Problem: Too slow on login node
**Solutions**:
1. Request an interactive compute node:
   ```bash
   salloc --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64GB --time=48:00:00
   # Then run your script normally on the compute node
   ```

2. Use fewer cases/domains to reduce total tasks

3. Wait for SLURM queue and use parallel processing

---

## Best Practices

### For Login Nodes

✅ **DO**:
- Use `--n-workers 1` (serial mode)
- Monitor resource usage
- Process during off-peak hours
- Test with one case first

❌ **DON'T**:
- Use `--n-workers >1` (parallel mode uses too much memory)
- Process all cases at once if login node is slow
- Leave many background jobs running
- Use excessive I/O operations

### Testing First

Before processing all 864 tasks:
```bash
# Test one task manually
cd /ccsopen/home/zhe1feng1/program/lasso/cellgrowth/regrid
python regrid_lasso_batch.py \
    --case-date 20190123 \
    --ensemble gefs18 \
    --start-hour 12 \
    --end-hour 12 \
    --file-type methamsl \
    --domain d4 \
    --n-workers 1

# If successful, process one full case (12 hours)
# Create test task file with just one case
head -24 serial_tasks.txt > test_tasks.txt

# Modify run_serial_tasks.sh to use test_tasks.txt
# Then run it
```

---

## Summary Commands

```bash
# 1. Generate task list
bash generate_serial_tasks.sh

# 2. Start processing in background
nohup bash run_serial_tasks.sh > serial_processing.log 2>&1 &

# 3. Monitor progress
tail -f serial_processing.log

# 4. Check if running
ps aux | grep run_serial_tasks

# 5. Check progress percentage
COMPLETED=$(grep "completed successfully" serial_processing.log | wc -l)
echo "Progress: ${COMPLETED}/864"

# 6. Stop if needed
kill $(pgrep -f run_serial_tasks)
```

---

## Expected Output

The log file will show:
```
==================================================
Starting serial task execution
Total tasks: 864
Started at: Thu Oct 30 20:00:00 EDT 2025
==================================================

==================================================
Task 1/864
Command: cd /path/to/regrid && python regrid_lasso_batch.py ...
Started: Thu Oct 30 20:00:00 EDT 2025
==================================================
[Processing output from regrid_lasso_batch.py]
✓ Task 1 completed successfully
Finished: Thu Oct 30 20:02:15 EDT 2025

==================================================
Task 2/864
...
```
