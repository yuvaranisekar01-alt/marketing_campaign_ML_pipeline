import subprocess
import sys

def run_pipeline():
    steps = [
        "python src/data_ingestion.py",
        "python src/preprocessing.py",
        "python src/feature_engineering.py",
        "python src/model_training_reg.py",
        "python src/model_training_cls.py",
        "python src/model_evaluation.py"
    ]

    for step in steps:
        print(f"\n▶ Running: {step}")
        result = subprocess.run(step, shell=True)
        if result.returncode != 0:
            print(f"❌ Failed at: {step}")
            sys.exit(1)
        print(f"✅ Done: {step}")

    print("\n🎉 Full pipeline completed!")
    print("Run: streamlit run dashboard.py")

if __name__ == "__main__":
    run_pipeline()
