import sys
from pathlib import Path

SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
}

SKIP_FILES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
    "uv.lock",
}

SKIP_EXTENSIONS = {
    ".pyc",
}

OUTPUT_FILE = "bundle.txt"


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True

    if path.name in SKIP_FILES:
        return True

    if path.suffix in SKIP_EXTENSIONS:
        return True

    return False


def bundle_folder(root: str):
    root = Path(root).resolve()

    if not root.is_dir():
        print(f"Error: '{root}' is not a directory.")
        sys.exit(1)

    output = root / OUTPUT_FILE

    with open(output, "w", encoding="utf-8") as bundle:
        for path in sorted(root.rglob("*")):

            if not path.is_file():
                continue

            if path == output:
                continue

            if should_skip(path):
                continue

            relative_path = path.relative_to(root)

            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                print(f"Skipping unreadable/binary file: {relative_path}")
                continue

            bundle.write("=" * 80 + "\n")
            bundle.write(f"FILE: {relative_path}\n")
            bundle.write("=" * 80 + "\n\n")
            bundle.write(content)
            bundle.write("\n\n")

    print(f"Bundled project into: {output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bundler.py <directory>")
        sys.exit(1)

    bundle_folder(sys.argv[1])