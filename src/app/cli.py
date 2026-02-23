# argparse / flags (camera vs video, headless, debug)

"""
cli.py

This module defines the command line interface (CLI) for running the application.
It parses arguments such as:
- camera index or video file path
- headless mode
- debug mode
- config path
- max frames

Inputs:
- Command line arguments (argv)

Outputs:
- argparse.Namespace with validated runtime settings
- Loaded configuration dictionary


"""