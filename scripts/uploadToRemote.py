import os
import paramiko
from settings import *

REMOTE_IP = "136.38.166.236"
REMOTE_PORT = 34738
REMOTE_USER = "root"
REMOTE_DIR = "/workspace/weatherPredict/data/test"
LOCAL_DIR = TRAINING_DATA_PATH


def get_remote_latest_file_date():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_IP, port=REMOTE_PORT, username=REMOTE_USER)

    # List files in Vast.ai dataset directory
    stdin, stdout, stderr = ssh.exec_command(f"ls {REMOTE_DIR}")
    files = stdout.readlines()

    latest_date = None
    for file in files:
        file_name = file.strip()

        # Extract the first 19 characters (YYYY_MM_DD_HH_MM_SS)
        date_str = file_name[:19]
        try:
            file_date = datetime.strptime(date_str, "%Y_%m_%d_%H_%M_%S")
            if latest_date is None or file_date > latest_date:
                latest_date = file_date
        except ValueError:
            continue  # Skip invalid filenames

    ssh.close()
    return latest_date


def upload_new_images(latest_date):
    # Check local images and upload those with a newer date
    for file_name in os.listdir(LOCAL_DIR):
        local_path = os.path.join(LOCAL_DIR, file_name)

        # Extract the first 19 characters (YYYY_MM_DD_HH_MM_SS)
        date_str = file_name[:19]
        try:
            file_date = datetime.strptime(date_str, "%Y_%m_%d_%H_%M_%S")
        except ValueError:
            continue  # Skip files with no valid date

        # If the file date is later than the latest uploaded date, upload it
        if latest_date is None or file_date > latest_date:
            print(f"Uploading {file_name}...")
            os.system(f"scp -P {REMOTE_PORT} {local_path} {REMOTE_USER}@{REMOTE_IP}:{REMOTE_DIR}")


# Run the script
latest_date = get_remote_latest_file_date()
upload_new_images(latest_date)