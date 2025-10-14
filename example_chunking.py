"""
Example Chunking Runner (All Modes)
-----------------------------------
Runs the real CLI chunking command (handle_chunk) for all modes:
- Time-based (all roles)
- Role-based (grouped by role)
- Hybrid (role + time)

Usage:
    python example_chunking.py fdb_logs.duckdb
"""

import sys
from argparse import Namespace
from cli.main import CLI  # import your CLI class


def main():
    if len(sys.argv) < 2:
        print("Usage: python example_chunking.py <db_path>")
        sys.exit(1)

    db_path = sys.argv[1]
    cli = CLI()

    # Example role for hybrid mode
    target_role = "SS"

    # Each mode config: (mode_name, time_interval, role_filter)
    modes = [
        ("time", 60, None),
        ("role", None, None),
        ("hybrid", 10, target_role),
    ]

    for mode, interval, role in modes:
        print("\n" + "=" * 80)
        print(f"RUNNING CHUNKING MODE: {mode.upper()}")
        if role:
            print(f"Role filter: {role}")
        print("=" * 80)

        # Create argparse-like args namespace
        args = Namespace(
            db=db_path,
            mode=mode,
            interval=interval if interval else 60,
            limit=5,
            role=role,
        )

        # Run the real chunking logic from CLI
        cli.handle_chunk(args)

    print("\n" + "=" * 80)
    print("All chunking modes completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
