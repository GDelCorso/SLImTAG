"""Executable entry point for SLImTAG."""

from slimtag_app.application import SegmentationApp


def main():
    """Start the desktop application."""
    SegmentationApp().mainloop()


if __name__ == "__main__":
    main()
