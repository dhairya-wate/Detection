"""
Convenience entrypoint to run the Waste Detection API and Camera Detector.

Usage:
    python run.py             # Starts Web API & Live Camera Dashboard on http://127.0.0.1:8000
    python run.py --desktop   # Starts the Desktop OpenCV Camera Window directly
    python run.py --help      # Show options
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR / "waste-api"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TACO Dataset Waste Detection Runner")
    parser.add_argument(
        "--desktop", "--camera", "-c",
        action="store_true",
        help="Launch the desktop OpenCV webcam window instead of the web server",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Web server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    args = parser.parse_args()

    if args.desktop:
        from detect_camera import run_camera
        run_camera()
    else:
        import uvicorn
        print("=" * 65)
        print("  Starting EcoVision TACO Waste Detection System")
        print(f"  Live Camera Dashboard : http://127.0.0.1:{args.port}/")
        print(f"  Interactive API Docs   : http://127.0.0.1:{args.port}/docs")
        print(f"  Health Check Endpoint : http://127.0.0.1:{args.port}/health")
        print("=" * 65)
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=True)
