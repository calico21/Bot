@echo off
echo 🚀 Launching 6 Optimization Workers on i7-13620H...

:: Performance Cores (Heavy Lifting)
start "Worker 1" python research_lab.py --mode optimize
start "Worker 2" python research_lab.py --mode optimize
start "Worker 3" python research_lab.py --mode optimize
start "Worker 4" python research_lab.py --mode optimize
start "Worker 5" python research_lab.py --mode optimize
start "Worker 6" python research_lab.py --mode optimize

:: Optional: Uncomment these if you are AFK (Away From Keyboard) for max speed
start "Worker 7" python research_lab.py --mode optimize
start "Worker 8" python research_lab.py --mode optimize

echo ✅ All workers started.
echo ⚠️  If you see "Database Locked" errors, ignore them. They will retry automatically.
pause