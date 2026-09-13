import os
import sys

def main():
    print("AI-Driven Mobile AR National Heroes Pipeline Manager")
    print("Step 1: Generating AR media mock assets...")
    from create_mock_assets import generate_assets
    generate_assets()

    print("Step 2: Checking dataset directories...")
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    print("Pipeline ready.")

if __name__ == "__main__":
    main()
