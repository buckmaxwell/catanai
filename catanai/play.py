#!/usr/bin/env python3
import os
import sys
import shutil

# Wrapper entrypoint: always cd into the project root, then run a default Catanatron play command

def main():
    """
    Ensures we run from the repo root and then executes a fixed
    `catanatron-play` invocation with our default bot settings.
    """
    # Change to project root (directory containing this file)
    project_root = os.path.abspath(os.path.dirname(__file__))
    os.chdir(project_root)

    # Locate the catanatron-play executable
    cmd = shutil.which("catanatron-play")
    if not cmd:
        print("Error: 'catanatron-play' not found in PATH. Did you install the experimental extras?", file=sys.stderr)
        sys.exit(1)

    # Default arguments for our Catalina bot
    default_args = [
        "--code=ai/players/catalina.py",
        "--players=R,R,R,Catalina",
        #"--players=F,F,F,Catalina",
        "--num=10" # 10
    ]

    # Replace current process with catanatron-play + default args
    os.execv(cmd, [cmd] + default_args)

if __name__ == "__main__":
    main()
