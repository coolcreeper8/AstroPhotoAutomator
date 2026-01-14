import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    print("Checking imports...")
    from core.video_loader import VideoLoader
    from core.processing import FrameAnalyzer
    from core.stacking import Stacker
    from core.post_processing import WaveletEnhancer, ColorCorrector
    from gui.main_window import MainWindow
    
    print("Classes imported successfully.")
    
    # Check instantiation of non-GUI classes
    stacker = Stacker()
    print("Stacker instantiated.")
    
    # We can't instantiate MainWindow easily without QApplication, but import is good enough check for now
    
    print("All checks passed.")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
