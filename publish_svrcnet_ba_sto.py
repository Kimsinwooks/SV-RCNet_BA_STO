#!/usr/bin/env python3
"""
Validate and publish /home/everest/SV-RCNet_BA_STO to GitHub.

Default repository:
    https://github.com/Kimsinwooks/SV-RCNet_BA_STO

The script:
1. validates the dataset structure;
2. installs Git LFS attributes for *.npy;
3. writes .gitignore;
4. initializes Git and creates a commit;
5. creates the GitHub repository when it does not exist;
6. pushes the main branch.

The GitHub token is entered securely at runtime and is not saved in this file.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path("/home/everest/SV-RCNet_BA_STO")
DATASET_ROOT = ROOT / "dataset"
GITHUB_OWNER = "Kimsinwooks"
REPOSITORY_NAME = "SV-RCNet_BA_STO"
REMOTE_URL = f"https://github.com/{GITHUB_OWNER}/{REPOSITORY_NAME}.git"

EXPECTED_CASE_COUNTS = {
    "train": 28,
    "val": 4,
    "test": 8,
}

EXPECTED_ID_RANGES = {
    "train": range(1, 29),
    "val": range(29, 33),
    "test": range(33, 41),
}

METADATA_CASE_IDS = set(range(22, 28))
METADATA_TYPES = ("gauze", "tool", "vessel", "organ")


def run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command))
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        check=check,
        capture_output=capture,
    )


def case_name(case_id: int) -> str:
    return f"STOa_PS{case_id:03d}_STOs01"


def validate_dataset() -> None:
    if not ROOT.is_dir():
        raise RuntimeError(f"Repository directory does not exist: {ROOT}")
    if not DATASET_ROOT.is_dir():
        raise RuntimeError(f"Dataset directory does not exist: {DATASET_ROOT}")

    errors: list[str] = []
    total_case_dirs = 0
    total_npy_files = 0

    for split, expected_count in EXPECTED_CASE_COUNTS.items():
        split_dir = DATASET_ROOT / split
        if not split_dir.is_dir():
            errors.append(f"Missing split directory: {split_dir}")
            continue

        case_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        actual_names = [path.name for path in case_dirs]
        expected_names = [case_name(case_id) for case_id in EXPECTED_ID_RANGES[split]]

        if len(case_dirs) != expected_count:
            errors.append(
                f"{split}: expected {expected_count} case directories, "
                f"found {len(case_dirs)}"
            )

        if actual_names != expected_names:
            errors.append(
                f"{split}: case directory sequence is incorrect.\n"
                f"  expected: {expected_names}\n"
                f"  actual:   {actual_names}"
            )

        total_case_dirs += len(case_dirs)

        for case_dir in case_dirs:
            match = re.fullmatch(r"STOa_PS(\d{3})_STOs01", case_dir.name)
            if not match:
                errors.append(f"Invalid case directory name: {case_dir}")
                continue

            case_id = int(match.group(1))
            prefix = case_dir.name

            phase_file = case_dir / f"{prefix}_phase.npy"
            if not phase_file.is_file():
                errors.append(f"Missing phase file: {phase_file}")

            expected_types = {"phase"}
            if case_id in METADATA_CASE_IDS:
                expected_types.update(METADATA_TYPES)

            npy_files = sorted(case_dir.glob("*.npy"))
            total_npy_files += len(npy_files)

            actual_types: set[str] = set()
            for npy_file in npy_files:
                expected_prefix = prefix + "_"
                if not npy_file.name.startswith(expected_prefix):
                    errors.append(
                        f"Filename does not match its directory: {npy_file}"
                    )
                    continue

                file_type = npy_file.stem[len(expected_prefix):]
                actual_types.add(file_type)

            if actual_types != expected_types:
                errors.append(
                    f"{case_dir}: expected file types {sorted(expected_types)}, "
                    f"found {sorted(actual_types)}"
                )

    if total_case_dirs != 40:
        errors.append(f"Expected 40 total case directories, found {total_case_dirs}")

    if total_npy_files != 64:
        errors.append(f"Expected 64 total NumPy files, found {total_npy_files}")

    required_files = [
        ROOT / "README.md",
        ROOT / "prepare_sto_dataset.py",
        ROOT / "split_mapping.csv",
    ]
    for required_file in required_files:
        if not required_file.is_file():
            errors.append(f"Missing repository file: {required_file}")

    if errors:
        print("\n[VALIDATION FAILED]")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\n[VALIDATION PASSED]")
    print("- train: 28 cases")
    print("- val:    4 cases")
    print("- test:   8 cases")
    print("- total: 40 cases")
    print("- NumPy files: 64")


def write_gitignore() -> None:
    gitignore = """__pycache__/
*.pyc
*.pyo
*.log
.ipynb_checkpoints/
.DS_Store
"""
    (ROOT / ".gitignore").write_text(gitignore, encoding="utf-8")
    print("[OK] Wrote .gitignore")


def initialize_git() -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is not installed.")
    if shutil.which("git-lfs") is None:
        raise RuntimeError(
            "git-lfs is not installed. Install it with: sudo apt install git-lfs"
        )

    if not (ROOT / ".git").exists():
        run(["git", "init"])

    run(["git", "branch", "-M", "main"])
    run(["git", "lfs", "install"])
    run(["git", "lfs", "track", "*.npy"])

    if not (ROOT / ".gitattributes").is_file():
        raise RuntimeError(".gitattributes was not created by Git LFS.")

    user_name = run(
        ["git", "config", "--get", "user.name"],
        check=False,
        capture=True,
    ).stdout.strip()
    user_email = run(
        ["git", "config", "--get", "user.email"],
        check=False,
        capture=True,
    ).stdout.strip()

    if not user_name:
        run(["git", "config", "user.name", GITHUB_OWNER])

    if not user_email:
        email = input("Git commit email: ").strip()
        if not email:
            raise RuntimeError("A Git commit email is required.")
        run(["git", "config", "user.email", email])


def stage_and_commit() -> None:
    run(["git", "add", "."])

    staged_npy = run(
        ["git", "diff", "--cached", "--name-only"],
        capture=True,
    ).stdout.splitlines()
    staged_npy_count = sum(name.endswith(".npy") for name in staged_npy)

    if staged_npy_count not in (0, 64):
        raise RuntimeError(
            f"Expected either 0 or 64 staged NumPy files, found {staged_npy_count}"
        )

    status = run(["git", "status", "--porcelain"], capture=True).stdout.strip()
    if not status:
        print("[OK] Nothing new to commit.")
        return

    run(
        [
            "git",
            "commit",
            "-m",
            "Add SV-RCNet_BA_STO dataset with 28-4-8 split",
        ]
    )

    lfs_count_output = run(
        ["git", "lfs", "ls-files"],
        capture=True,
    ).stdout.splitlines()
    if len(lfs_count_output) != 64:
        raise RuntimeError(
            f"Expected 64 Git LFS files after commit, found {len(lfs_count_output)}"
        )

    print("[OK] Commit created and 64 NumPy files are tracked by Git LFS.")


def github_request(
    method: str,
    endpoint: str,
    token: str,
    payload: dict | None = None,
) -> tuple[int, dict]:
    url = f"https://api.github.com{endpoint}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "SV-RCNet_BA_STO-publisher",
        },
    )

    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = {"message": body}
        return exc.code, parsed


def ensure_github_repository(token: str) -> None:
    status, account = github_request("GET", "/user", token)
    if status != 200:
        raise RuntimeError(
            f"GitHub authentication failed ({status}): "
            f"{account.get('message', account)}"
        )

    login = account.get("login")
    if login.lower() != GITHUB_OWNER.lower():
        raise RuntimeError(
            f"The token belongs to '{login}', not '{GITHUB_OWNER}'."
        )
    print(f"[OK] Authenticated as {login}")

    endpoint = f"/repos/{GITHUB_OWNER}/{REPOSITORY_NAME}"
    status, repository = github_request("GET", endpoint, token)

    if status == 200:
        print(f"[OK] Repository already exists: {repository.get('html_url')}")
        return

    if status != 404:
        raise RuntimeError(
            f"Unable to check repository ({status}): "
            f"{repository.get('message', repository)}"
        )

    status, repository = github_request(
        "POST",
        "/user/repos",
        token,
        {
            "name": REPOSITORY_NAME,
            "description": (
                "Dataset configuration for surgical phase recognition "
                "in laparoscopic gastrectomy"
            ),
            "private": False,
            "has_issues": True,
            "has_projects": False,
            "has_wiki": False,
        },
    )

    if status != 201:
        raise RuntimeError(
            f"Repository creation failed ({status}): "
            f"{repository.get('message', repository)}"
        )

    print(f"[OK] Created repository: {repository.get('html_url')}")


def configure_remote() -> None:
    remotes = run(["git", "remote"], capture=True).stdout.split()
    if "origin" in remotes:
        run(["git", "remote", "set-url", "origin", REMOTE_URL])
    else:
        run(["git", "remote", "add", "origin", REMOTE_URL])

    print(f"[OK] origin = {REMOTE_URL}")


def push(token: str) -> None:
    # Use a temporary askpass helper so the token is not embedded in the remote URL.
    askpass_path = ROOT / ".git-askpass-temp.sh"
    askpass_path.write_text(
        """#!/bin/sh
case "$1" in
  *Username*) printf '%s\\n' "$GITHUB_USERNAME" ;;
  *Password*) printf '%s\\n' "$GITHUB_TOKEN" ;;
esac
""",
        encoding="utf-8",
    )
    askpass_path.chmod(0o700)

    env = os.environ.copy()
    env["GIT_ASKPASS"] = str(askpass_path)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GITHUB_USERNAME"] = GITHUB_OWNER
    env["GITHUB_TOKEN"] = token

    try:
        print("+ git push -u origin main")
        subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=ROOT,
            env=env,
            text=True,
            check=True,
        )
    finally:
        askpass_path.unlink(missing_ok=True)

    print(f"\n[DONE] Published to https://github.com/{GITHUB_OWNER}/{REPOSITORY_NAME}")


def main() -> None:
    print(f"Repository directory: {ROOT}")
    validate_dataset()
    write_gitignore()
    initialize_git()
    stage_and_commit()

    token = getpass.getpass("GitHub Personal Access Token: ").strip()
    if not token:
        raise RuntimeError("A GitHub token is required.")

    ensure_github_repository(token)
    configure_remote()
    push(token)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\n[COMMAND FAILED] Exit code: {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
