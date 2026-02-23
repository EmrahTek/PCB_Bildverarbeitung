# argparse / flags (camera vs video, headless, debug)
# Saubere CLI: Kamera vs Video, Debug, Headless, max-frames, config-path.
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
"""
Zu implementierende Funktionen

    build_arg_parser() -> argparse.ArgumentParser

    parse_args(argv=None) -> Namespace

    load_config(config_path: Path) -> dict

    (Optional) validate_args(args) -> None


   argparse:
https://docs.python.org/3/library/argparse.html 

    
"""