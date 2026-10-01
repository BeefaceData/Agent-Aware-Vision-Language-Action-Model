"""Explicit, small personal-account smoke. No GitHub mutation or real issue work."""
import argparse
import json
from pathlib import Path
import tempfile
import sys
from runtime import Stop, invoke_model, read_json, run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("codex", "cursor"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    parent = root / ".ralph/smoke"
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=args.backend + "-", dir=parent))
    tree = directory / "worktree"
    tree.mkdir()
    (tree / "temperature.py").write_text("def fahrenheit(celsius):\n    raise NotImplementedError\n", encoding="utf-8")
    (tree / "test_temperature.py").write_text(
        "import unittest\nfrom temperature import fahrenheit\n\nclass TemperatureTests(unittest.TestCase):\n"
        "    def test_freezing(self): self.assertEqual(fahrenheit(0), 32)\n"
        "    def test_boiling(self): self.assertEqual(fahrenheit(100), 212)\n"
        "    def test_negative(self): self.assertEqual(fahrenheit(-40), -40)\n", encoding="utf-8")
    run(["git", "init", "-q"], tree)
    run(["git", "add", "temperature.py", "test_temperature.py"], tree)
    run(["git", "diff", "--cached"], tree)
    run(["git", "-c", "user.name=Ralph smoke", "-c", "user.email=ralph-smoke@example.invalid",
         "commit", "-qm", "test: :white_check_mark: seed isolated CLI smoke",
         "-m", "Exercise the pinned coding model on a synthetic fixture without touching capstone issues."], tree)
    policy = read_json(root / "ralph/policy.json")
    prompt = ("This is an authorized, isolated CLI smoke test. Implement fahrenheit(celsius) in temperature.py. "
              "Use the existing public tests. Edit only temperature.py, leave tests unchanged, and run "
              "python -m unittest -v. No network, credentials, git commits, subprocess agents or other files. "
              "Return a short summary. Work directly; no clarification is needed.")
    invoke_model(args.backend, policy["workers"][args.backend], tree, prompt,
                 directory / "worker", 240)
    check = run([sys.executable, "-m", "unittest", "-v"], tree, log=directory / "checks.log")
    diff = run(["git", "diff", "--name-only"], tree).stdout.strip().splitlines()
    if diff != ["temperature.py"]:
        raise Stop("Smoke changed unexpected files or did not implement the fixture.")
    if args.backend == "codex":
        review_prompt = ("Read-only review of this synthetic smoke. Read temperature.py and test_temperature.py, "
            "do not execute code or edit files. Check the conversion against all three tests. "
            'Return only JSON: {"verdict":"pass" or "revise", "reason":"concrete reasoning"}.')
        response = invoke_model("codex", policy["final_review"], tree, review_prompt,
                                directory / "review", 180, review=True)
        review = json.loads(response)
        if review.get("verdict") != "pass":
            raise Stop("Astra smoke review did not pass.")
    write_json(directory / "result.json", {"backend": args.backend, "worker": policy["workers"][args.backend],
        "tests_exit": check.returncode, "changed_files": diff,
        "reviewer_tested": policy["final_review"] if args.backend == "codex" else None})
    print(f"PASS: {directory}")


if __name__ == "__main__":
    main()
