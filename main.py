import os
import sys

# Ensure Python can find the training module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train_hybrid import run_training


def setup_environment():
    """
    Creates the required folder structure if it doesn't exist.
    We don't count files here; we let train_hybrid.py handle data validation.
    """
    base_path = os.path.join("data", "raw")
    folders = ["flickr30k", "BossBase and BOWS2"]

    created = False
    for folder in folders:
        path = os.path.join(base_path, folder)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"[SETUP] Created missing folder: {path}")
            created = True

    if created:
        print("[INFO] Folder structure created.")
        print("       Please add your JPGs to 'data/raw/flickr30k'")
        print("       Please add your PGMs/PNGs to 'data/raw/bossbase'")


if __name__ == "__main__":
    print("==========================================")
    print("       Steganography Defense System       ")
    print("         Hybrid Evolutionary Run          ")
    print("==========================================")

    # 1. Just create the folders if missing
    setup_environment()

    # 2. Run Training (This handles the file counting/validation)
    try:
        run_training()

    except ValueError as e:
        # Catch the specific "Not enough data" error from train_hybrid
        print(f"\n[ERROR] Data Setup Issue: {e}")
    except KeyboardInterrupt:
        print("\n[STOP] Training stopped by user.")
    except Exception as e:
        print(f"\n[CRITICAL] Error during training: {e}")
        raise e