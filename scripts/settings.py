from datetime import datetime
import time


SELECTED_INDICES = [6, 7, 10, 11, 12, 13]
BATCH_SIZE = 64
EPOCHS_TO_SAVE = 10

NUM_LABELS = 8
LATENT_DIM = 100


def safe_float(val):
    try:
        return float(val)
    except ValueError:
        return


def print_error(message):
    print(f"\033[91m[ERROR] {datetime.utcnow().isoformat()} - {message}\033[0m")
    print(f"[ERROR] {datetime.utcnow().isoformat()} - {message}")


def get_elapsed_ms(last_time_ref):
    current_time = time.perf_counter()
    elapsed_time = (current_time - last_time_ref[0]) * 1000
    last_time_ref[0] = current_time
    return elapsed_time
