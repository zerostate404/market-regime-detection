"""Entrypoint: Launch the Streamlit dashboard.

Usage:
    python scripts/run_dashboard.py
"""

import subprocess
import sys


def main():
    print("Launching Market Regime Drift Detector Dashboard...")
    print("Press Ctrl+C in this terminal to stop the server.\n")

    try:
        # We use subprocess to call the streamlit CLI
        subprocess.run(
            ["streamlit", "run", "src/mrd/phase6_dashboard/app.py"], check=True
        )
    except FileNotFoundError:
        print("ERROR: Streamlit is not installed.")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
