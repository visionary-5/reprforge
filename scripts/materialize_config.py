"""Resolve anonymized input roots without modifying experimental settings."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    replacements = {
        "${INPUTS}": str(args.inputs.resolve()),
        "${CODE}": str(Path(__file__).resolve().parents[1]),
    }

    def resolve(value):
        if isinstance(value, str):
            for key, root in replacements.items():
                value = value.replace(key, root)
            if "${" in value:
                raise ValueError("Unresolved template variable")
        elif isinstance(value, list):
            return [resolve(x) for x in value]
        elif isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        return value

    data = resolve(json.loads(args.template.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(data, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
