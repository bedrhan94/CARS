"""Run a command pinned to the efficiency cores (logical CPUs 16-27) at below-normal priority, so that auxiliary work
(document builds, light scripts) does not compete with timed cells on the performance cores. Children inherit the
affinity and priority. Usage: python ecore.py <command> [args...]"""
import subprocess, sys
import psutil

p = psutil.Process()
p.cpu_affinity(list(range(16, 28)))
p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
sys.exit(subprocess.call(sys.argv[1:]))
