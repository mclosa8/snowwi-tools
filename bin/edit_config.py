#!/usr/bin/env python3
"""Edit a field in a .cfg file using configobj."""

import argparse
import sys
from configobj import ConfigObj


def main():
    parser = argparse.ArgumentParser(
        description="Edit a field in a .cfg file."
    )
    parser.add_argument("config_file", help="Path to the .cfg file")
    parser.add_argument("section", help="Section name in the config file")
    parser.add_argument("field", help="Field name within the section")
    parser.add_argument("value", help="New value to set")
    args = parser.parse_args()

    config = ConfigObj(args.config_file)

    if args.section not in config:
        print(f"Error: section '{args.section}' not found in {args.config_file}", file=sys.stderr)
        sys.exit(1)

    if args.field not in config[args.section]:
        print(f"Error: field '{args.field}' not found in section '{args.section}'", file=sys.stderr)
        sys.exit(1)

    config[args.section][args.field] = args.value
    config.write()
    print(f"Set [{args.section}] {args.field} = {args.value}")


if __name__ == "__main__":
    main()
