# AstroPhotoAutomator

A desktop application for processing planetary astrophotography videos. Combines frame stacking with post-processing to produce high-quality planetary images from raw video captures.

## Features

- **Multi-Video Support**: Load and combine multiple video files for better signal-to-noise ratio
- **AutoStakkert 3-Inspired Stacking**: Reference-based alignment for superior quality
- **Two-Stage Workflow**: Stack first, then interactively adjust post-processing
- **6-Layer Wavelet Sharpening**: Registax-style multi-scale detail enhancement
- **Color Correction**: Auto white balance and chromatic aberration correction
- **Export Options**: Save as PNG, TIFF, or JPEG

## Installation

### Prerequisites
- Python 3.10+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AstroPhotoAutomator.git
cd AstroPhotoAutomator

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
cd src
python main.py
```

### Workflow

1. **Load Videos**: Click "Add Video(s)" to select one or more planetary videos
2. **Configure Stacking**: Set the percentage of best frames to stack (e.g., 20%)
3. **Stack Frames**: Click "Stack Frames" to create initial stacked image
4. **Adjust Post-Processing**: Modify wavelet layers (1-6) and color settings
5. **Apply**: Click "Apply Post-Processing" to preview changes
6. **Save**: Export your final image

## Architecture

```
src/
├── main.py                 # Application entry point
├── core/                   # Processing algorithms
│   ├── video_loader.py     # Video frame extraction
│   ├── processing.py       # Frame analysis & quality metrics
│   ├── stacking.py         # Frame alignment & stacking
│   └── post_processing.py  # Wavelet sharpening & color correction
└── gui/                    # User interface
    ├── main_window.py      # Main window & controls
    └── workers.py          # Background processing threads
```

## Technical Highlights

- **Quality Estimation**: Laplacian variance for frame sharpness measurement
- **Sub-pixel Alignment**: FFT-based phase correlation for precise frame registration
- **Reference Stacking**: Creates reference from top 50% of frames for better alignment
- **Non-blocking UI**: QThread workers keep interface responsive during processing

## Dependencies

- PyQt6 - GUI framework
- OpenCV - Image/video processing
- NumPy - Numerical operations
- scikit-image - Image alignment algorithms
- SciPy - Signal processing

## Screenshots

*Coming soon*

## License

MIT License

## Acknowledgments

Inspired by professional astrophotography tools:
- [AutoStakkert](https://www.autostakkert.com/) - Stacking methodology
- [Registax](https://www.astronomie.be/registax/) - Wavelet sharpening approach
