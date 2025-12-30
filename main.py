import os
import sys
from training.train_hybrid import run_training

# --- FIX 1: Ensure Python can find the file if it's in the same folder ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_data_setup():
    """
    Verifies that the dataset is correctly placed in 'data/raw'.
    """
    data_path = os.path.join("data", "raw")

    # 1. Check if folder exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"[WARN] Created missing folder: {data_path}")
        print("   -> Please move your Flickr/Dataset images into this folder.")
        return False

    # 2. Count images
    valid_exts = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(data_path) if f.lower().endswith(valid_exts)]

    if len(files) < 20:
        print(f"[ERROR] Found only {len(files)} images in '{data_path}'.")
        print("   To see real learning, you need at least 50-100 images.")
        return False

    print(f"[OK] Found {len(files)} images ready for training.")
    return True


if __name__ == "__main__":
    print("==========================================")
    print("       Steganography Defense System       ")
    print("         Evolutionary Pilot Run           ")
    print("==========================================")

    if check_data_setup():
        try:
            run_training()

        except KeyboardInterrupt:
            print("\n[STOP] Training stopped by user.")
        except Exception as e:
            print(f"\n[CRITICAL] Error during training: {e}")
            raise e