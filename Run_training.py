import subprocess
import os

subprocess.Popen(
    ["python", "Emnist_training.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL,
    preexec_fn=os.setpgrp
)

# subprocess.Popen(
#     ["python", "Mnist_training.py"],
#     stdout=subprocess.DEVNULL,
#     stderr=subprocess.DEVNULL,
#     stdin=subprocess.DEVNULL,
#     preexec_fn=os.setpgrp
# )