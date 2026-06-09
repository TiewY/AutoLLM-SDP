import subprocess

def run_phase(script_name):
    print(f"\n🚀 Running {script_name}...")
    subprocess.run(["python", script_name])

def run_autollm_pipeline():
    print("\n========== AutoLLM PIPELINE START ==========")

    # Phase A
    run_phase("phase_a_baseline.py")

    # Phase B
    run_phase("phase_b_codebert.py")

    # Phase C
    run_phase("phase_c_codet5.py")

    # Phase D
    run_phase("phase_d_automl_compare.py")

    # Phase E
    run_phase("phase_e_autollm.py")

    print("\n🎉 AutoLLM PIPELINE COMPLETED!")

if __name__ == "__main__":
    run_autollm_pipeline()