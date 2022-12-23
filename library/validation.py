import os


def folder_check():
    """Creates necessary folders if not present"""

    os.makedirs("../logs", exist_ok=True)
    with open ("../logs/logfile.log", "w") as fo:
        pass
