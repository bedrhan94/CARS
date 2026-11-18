"""Run a command pinned to one logical CPU at normal priority (children inherit the affinity).
Usage: python pin_cpu.py <cpu> <command> [args...]"""
import subprocess, sys
import psutil

psutil.Process().cpu_affinity([int(sys.argv[1])])
sys.exit(subprocess.call(sys.argv[2:]))
