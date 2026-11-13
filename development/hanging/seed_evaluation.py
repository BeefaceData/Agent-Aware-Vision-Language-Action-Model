"""Paired policy-only / supervised evaluation over several seeds, with a PDF report.

Every run lands in one navigable folder::

    data/evaluations/<name>/
        README.md            index of the runs and the headline numbers
        report.pdf           full report
        metrics.json, metrics.csv
        launches.jsonl       one row per launch attempt (command, times, exit code)
        logs/                Modal console output per run
        seed<N>/policy_only/ frozen-policy baseline (no Claude calls)
        seed<N>/supervised/  full pipeline: policy, planner, public RGB-D geometry, supervisor

Runs execute serially (the policy worker keeps one observation history). The
``run`` command is resumable: a run whose summary/report already exists is
skipped, and an incomplete folder is set aside, never overwritten.

    python -m development.hanging.seed_evaluation run --root data/evaluations/NAME --seeds 15,16,17,18,19
    python -m development.hanging.seed_evaluation report --root data/evaluations/NAME
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
KINDS = ("policy_only", "supervised")
# USD per million tokens for claude-opus-5-5 (Anthropic first-party API). Thinking
# tokens are billed as output. Cache-write price is an assumption (1.25x input);
# it is listed separately in the report.
PRICES = dict(input=4.00, output=20.00, cache_read=0.20, cache_write=5.00)
# Approximate Modal on-demand USD per GPU hour; the report states they are estimates.
GPU_PRICE_PER_HOUR = {"T4": 0.59, "L4": 0.80, "A10G": 1.10}


def _complete(target, kind):
    return (target/("summary.json" if kind == "policy_only" else "report.json")).is_file()


def _command(kind, seed, target, model):
    common = ["--seed", str(seed), "--max-actions", "900", "--chunk-size", "16", "--output", str(target)]
    if kind == "policy_only":
        return [sys.executable, "-m", "modal", "run", "modal_apps/simulator.py::baseline", *common,
                "--time-limit-minutes", "70", "--perception-camera", "close"]
    return [sys.executable, "-m", "modal", "run", "modal_apps/simulator.py::supervised_hanging",
            "--model", model, *common, "--time-limit-minutes", "75", "--max-turns", "80",
            "--perception-view", "close", "--complementary-view"]


def run_all(root, seeds, *, model, gpu, attempts=2, launcher=subprocess.run):
    root = Path(root)
    (root/"logs").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MODAL_IMAGE_BUILDER_VERSION": "2025.06", "MODAL_ENVIRONMENT": "main",
           "SIM_GPU": gpu, "PYTHONIOENCODING": "utf-8"}
    (root/"plan.json").write_text(json.dumps(dict(seeds=seeds, kinds=KINDS, model=model, gpu=gpu,
        max_actions=900, chunk_size=16, created=datetime.now(timezone.utc).isoformat()), indent=2)+"\n")
    for seed in seeds:
        for kind in KINDS:
            target = root/f"seed{seed}"/kind
            for attempt in range(1, attempts+1):
                if _complete(target, kind):
                    break
                if target.exists():
                    index = 1
                    while target.with_name(f"{kind}.incomplete-{index}").exists():
                        index += 1
                    target.rename(target.with_name(f"{kind}.incomplete-{index}"))
                target.parent.mkdir(parents=True, exist_ok=True)
                command = _command(kind, seed, target, model)
                log = root/"logs"/f"seed{seed}_{kind}_attempt{attempt}.log"
                started = time.time()
                print(f"[{datetime.now():%H:%M}] seed {seed} {kind} attempt {attempt}", flush=True)
                with log.open("w", encoding="utf-8") as stream:
                    result = launcher(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
                with (root/"launches.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(seed=seed, kind=kind, attempt=attempt, command=command[2:],
                        log=log.relative_to(root).as_posix(), started_unix=started, ended_unix=time.time(),
                        wall_seconds=time.time()-started, exit_code=result.returncode,
                        complete=_complete(target, kind)))+"\n")
    return build_report(root)


# ----------------------------------------------------------------------------- metrics

def _read_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _gripper_open_action(actions):
    """Action at which the right gripper last started opening (>0.1) after being closed (<0.05).

    The last opening is the release at the rack; earlier ones belong to the handover.
    """
    closed, last = False, None
    for row in actions:
        if row.get("event") != "action_confirmed":
            continue
        command = row["command_state"][13]
        if command < .05:
            closed = True
        elif closed and command > .1:
            last, closed = row["index"], False
    return last


def _thumbnail(folder, name="final_view.png"):
    """Save the last saved observation's best view as a PNG for the report."""
    observations = sorted((folder/"observations").glob("observation_*.npz"))
    if not observations:
        return None
    target = folder/name
    if target.is_file():
        return target
    try:
        import numpy as np
        from PIL import Image
        with np.load(observations[-1]) as data:
            for camera in ("perception_camera", "head_camera"):
                if camera in data.files:
                    image = Image.fromarray(np.asarray(data[camera]))
                    image.thumbnail((480, 360))
                    image.save(target)
                    return target
    except Exception:
        return None
    return None


def policy_metrics(folder, launches):
    summary = json.loads((folder/"summary.json").read_text())
    actions = _read_jsonl(folder/"actions.jsonl")
    return dict(kind="policy_only", outcome=summary.get("outcome"), success=summary.get("native_success") is True,
        policy_actions=summary.get("policy_actions"), success_action=(summary.get("policy_actions")
            if summary.get("native_success") else None),
        gripper_first_opened_action=_gripper_open_action(actions),
        run_seconds=summary.get("elapsed_seconds"), wall_seconds=launches.get("wall_seconds"),
        claude_requests=0, cost_claude_usd=0., error_type=summary.get("error_type"),
        thumbnail=_thumbnail(folder))


def _cost(usage):
    total = (usage or {}).get("total", {})
    return (total.get("input_tokens", 0)*PRICES["input"] + total.get("output_tokens", 0)*PRICES["output"]
            + total.get("cache_read_input_tokens", 0)*PRICES["cache_read"]
            + total.get("cache_creation_input_tokens", 0)*PRICES["cache_write"])/1e6


def supervised_metrics(folder, launches):
    report = json.loads((folder/"report.json").read_text())
    summary = report.get("summary") or {}
    events = _read_jsonl(folder/"supervisor_events.jsonl")
    decisions = [e for e in events if e.get("event_type") == "hanging_supervisor_decision"]
    entry = next((e.get("entry_sequence") for e in decisions if e.get("entry_sequence") is not None), None)
    handed = next((e for e in decisions if e.get("decision") == "hand_back_to_policy"), None)
    takeover = next((e for e in decisions if e.get("decision") == "grounded_contact_recovery"), None)
    agreements = [e for e in events if e.get("event_type") == "aperture_two_frame_agreement"]
    recovery = summary.get("hanging_recovery") or {}
    planner = [e["planner_latency_seconds"] for e in events
               if e.get("event_type") == "planner_selection" and "planner_latency_seconds" in e]
    usage = report.get("claude_usage") or {}
    success = summary.get("task_success") is True
    acted = ("supervisor" if takeover is not None and recovery.get("stages") else "policy")
    gripper = _gripper_open_action([dict(event="action_confirmed", index=e.get("observation"),
        command_state=e["command_state"]) for e in events
        if e.get("event_type") == "action_confirmed" and "command_state" in e])
    return dict(kind="supervised", outcome=report.get("status"), success=success,
        policy_actions=summary.get("policy_actions"),
        success_action=summary.get("policy_actions") if success else None,
        gripper_first_opened_action=gripper, planner_turns=summary.get("planner_turns"),
        lift_entry_sequence=entry, decisions={d: sum(e.get("decision") == d for e in decisions)
                                               for d in sorted({e.get("decision") for e in decisions})},
        handed_back_reason=None if handed is None else handed.get("reason"),
        handed_back_at=None if handed is None else handed.get("observation_sequence"),
        takeover_at=None if takeover is None else takeover.get("observation_sequence"),
        recovery_status=recovery.get("status"), recovery_stages=list((recovery.get("stages") or {})),
        success_by=(acted if success else None), analytic_ticks=summary.get("analytic_ticks"),
        geometry_checks=len(agreements),
        geometry_agreed=sum(e.get("status") == "agreed" for e in agreements),
        geometry_crops=sum(e.get("event_type") == "public_geometry_crop" for e in events),
        claude_requests=report.get("model_requests"), perception_requests=report.get("perception_requests"),
        claude_usage=usage, cost_claude_usd=_cost(usage),
        claude_latency_mean_s=(usage.get("total") or {}).get("latency_seconds_mean"),
        claude_latency_max_s=(usage.get("total") or {}).get("latency_seconds_max"),
        planner_latency_median_s=statistics.median(planner) if planner else None,
        planner_latency_max_s=max(planner) if planner else None,
        run_seconds=report.get("elapsed_seconds"), wall_seconds=launches.get("wall_seconds"),
        error_type=(report.get("failure") or {}).get("type") if isinstance(report.get("failure"), dict) else None,
        thumbnail=_thumbnail(folder))


def collect(root):
    root = Path(root)
    plan = json.loads((root/"plan.json").read_text()) if (root/"plan.json").is_file() else {}
    launches = {}
    for row in _read_jsonl(root/"launches.jsonl"):
        launches[(row["seed"], row["kind"])] = row
    gpu = (plan.get("gpu") or "T4").split(",")[0]
    rows = []
    for seed in plan.get("seeds") or sorted(int(p.name[4:]) for p in root.glob("seed*") if p.name[4:].isdigit()):
        for kind in KINDS:
            folder = root/f"seed{seed}"/kind
            launch = launches.get((seed, kind), {})
            try:
                metrics = (policy_metrics if kind == "policy_only" else supervised_metrics)(folder, launch)
            except FileNotFoundError:
                metrics = dict(kind=kind, outcome="missing" if not folder.exists() else "incomplete",
                               success=False, claude_requests=0, cost_claude_usd=0., thumbnail=None,
                               wall_seconds=launch.get("wall_seconds"))
            seconds = metrics.get("wall_seconds") or metrics.get("run_seconds") or 0
            metrics.update(seed=seed, folder=folder.relative_to(root).as_posix(),
                           cost_gpu_usd_estimate=seconds/3600*GPU_PRICE_PER_HOUR.get(gpu, .8))
            rows.append(metrics)
    return plan, rows


# ----------------------------------------------------------------------------- outputs

CSV_FIELDS = ("seed", "kind", "outcome", "success", "success_action", "policy_actions",
              "gripper_first_opened_action", "lift_entry_sequence", "handed_back_reason", "handed_back_at",
              "takeover_at", "recovery_status", "success_by", "geometry_checks", "geometry_agreed",
              "geometry_crops", "claude_requests", "perception_requests", "cost_claude_usd",
              "claude_latency_mean_s", "claude_latency_max_s", "planner_latency_median_s",
              "run_seconds", "wall_seconds", "cost_gpu_usd_estimate", "folder")


def _fmt(value, digits=2):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _totals(rows):
    out = {}
    for kind in KINDS:
        group = [r for r in rows if r["kind"] == kind]
        done = [r for r in group if r.get("outcome") not in ("missing", "incomplete")]
        out[kind] = dict(runs=len(group), completed=len(done), successes=sum(r["success"] for r in group),
            claude_requests=sum(r.get("claude_requests") or 0 for r in group),
            cost_claude_usd=sum(r.get("cost_claude_usd") or 0 for r in group),
            cost_gpu_usd_estimate=sum(r.get("cost_gpu_usd_estimate") or 0 for r in group),
            wall_hours=sum(r.get("wall_seconds") or 0 for r in group)/3600)
    return out


def build_report(root):
    root = Path(root)
    plan, rows = collect(root)
    totals = _totals(rows)
    serial = [{k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()} for r in rows]
    (root/"metrics.json").write_text(json.dumps(dict(plan=plan, prices_usd_per_million_tokens=PRICES,
        gpu_price_per_hour_estimate=GPU_PRICE_PER_HOUR, totals=totals, runs=serial), indent=2)+"\n")
    with (root/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in serial:
            writer.writerow({k: _fmt(row.get(k), 4) if not isinstance(row.get(k), str) else row[k]
                             for k in CSV_FIELDS})
    _write_readme(root, plan, rows, totals)
    _write_pdf(root, plan, rows, totals)
    return dict(totals=totals, runs=len(rows))


def _write_readme(root, plan, rows, totals):
    lines = [f"# Seed evaluation: {root.name}", "",
             f"Seeds {', '.join(map(str, plan.get('seeds', [])))}; model `{plan.get('model', '-')}`; "
             f"GPU request `{plan.get('gpu', '-')}`; 900 policy actions per run.", "",
             "Open `report.pdf` for the full report; `metrics.csv` / `metrics.json` hold every number.", "",
             "| Condition | Successes | Completed runs | Claude requests | Claude cost (USD) |",
             "|---|---|---|---|---|"]
    for kind in KINDS:
        t = totals[kind]
        lines.append(f"| {kind} | {t['successes']}/{t['runs']} | {t['completed']}/{t['runs']} | "
                     f"{t['claude_requests']} | {t['cost_claude_usd']:.2f} |")
    lines += ["", "| Seed | Condition | Outcome | Success | Folder |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['kind']} | {r.get('outcome')} | {_fmt(r['success'])} | "
                     f"[{r['folder']}]({r['folder']}) |")
    lines += ["", "Each run folder has `episode.mp4`, `observations/`, `final_view.png` and its "
              "summary (`summary.json` for policy-only, `report.json` + `supervisor_events.jsonl` "
              "for supervised). Console logs are in `logs/`."]
    (root/"README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def _write_pdf(root, plan, rows, totals):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    styles = getSampleStyleSheet()
    body, small = styles["BodyText"], styles["BodyText"].clone("small", fontSize=7.5, leading=9)
    story = []

    def table(data, widths=None, size=7.5):
        t = Table([[Paragraph(_fmt(c), small) if not isinstance(c, Paragraph) else c for c in row]
                   for row in data], colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f3e4e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .25, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), size), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f7")])]))
        return t

    p, s = totals["policy_only"], totals["supervised"]
    story += [Paragraph(f"Hanging-phase seed evaluation ({root.name})", styles["Title"]),
        Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M}. Seeds "
                  f"{', '.join(map(str, plan.get('seeds', [])))}; Claude model {plan.get('model', '-')}; "
                  f"simulator GPU request {plan.get('gpu', '-')}; 900 frozen-policy actions per run, "
                  "chunk size 16.", body), Spacer(1, 6),
        Paragraph("<b>Conditions.</b> <i>Policy-only</i>: the frozen Neotix policy alone (no planner, "
                  "perception service, Claude or analytic motion). <i>Supervised</i>: the full pipeline. "
                  "A Claude planner selects bounded skills, public RGB-D perception with Claude visual "
                  "proposals checks lift and hanging geometry from two close fixed cameras, and the "
                  "supervisor may take over with a guarded approach, seat, release and retreat. The "
                  "supervisor gives the policy 160 actions after a confirmed lift and hands control back "
                  "to the policy instead of ending the run when it cannot confirm geometry.", body),
        Spacer(1, 6),
        Paragraph("<b>Success</b> is RoboTwin's native checker latching (the run stops there). It does "
                  "not by itself verify sustained support after release.", body), Spacer(1, 10),
        Paragraph("Headline", styles["Heading2"]),
        table([["Condition", "Successes", "Completed runs", "Claude requests", "Claude cost (USD)",
                "GPU cost (USD, est.)", "Wall time (h)"],
               ["Policy-only", f"{p['successes']}/{p['runs']}", f"{p['completed']}/{p['runs']}",
                p["claude_requests"], f"{p['cost_claude_usd']:.2f}", f"{p['cost_gpu_usd_estimate']:.2f}",
                f"{p['wall_hours']:.2f}"],
               ["Supervised", f"{s['successes']}/{s['runs']}", f"{s['completed']}/{s['runs']}",
                s["claude_requests"], f"{s['cost_claude_usd']:.2f}", f"{s['cost_gpu_usd_estimate']:.2f}",
                f"{s['wall_hours']:.2f}"]], size=8.5),
        Spacer(1, 10), Paragraph("Per seed, paired", styles["Heading2"])]
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["kind"]] = r
    data = [["Seed", "Policy-only outcome", "Success at action", "Supervised outcome", "Success at action",
             "Who completed it", "Supervisor involvement"]]
    for seed, pair in by_seed.items():
        po, su = pair.get("policy_only", {}), pair.get("supervised", {})
        involvement = ("took over at obs %s (%s)" % (su.get("takeover_at"), su.get("recovery_status"))
                       if su.get("takeover_at") is not None else
                       "handed back at obs %s: %s" % (su.get("handed_back_at"), su.get("handed_back_reason"))
                       if su.get("handed_back_reason") else
                       "lift confirmed at obs %s, no takeover" % su.get("lift_entry_sequence")
                       if su.get("lift_entry_sequence") is not None else "no confirmed lift")
        data.append([seed, po.get("outcome"), po.get("success_action"), su.get("outcome"),
                     su.get("success_action"), su.get("success_by"), involvement])
    story += [table(data, [1.2*cm, 3.2*cm, 2*cm, 4*cm, 2*cm, 2.4*cm, 9*cm]), Spacer(1, 6),
              Paragraph("Because the frozen policy is not repeatable on a seed (commanded joints of two "
                        "seed-5 runs differed by up to 0.53 rad), a supervised run may follow a different "
                        "trajectory from its policy-only pair. Compare rates across seeds, not single "
                        "pairs.", small), PageBreak()]

    sup = [r for r in rows if r["kind"] == "supervised"]
    story += [Paragraph("Supervisor and geometry", styles["Heading2"]),
        table([["Seed", "Lift at obs", "Decisions", "Geometry checks", "Two-frame agreed", "Crops",
                "Takeover", "Recovery stages", "Handed back"]] +
              [[r["seed"], r.get("lift_entry_sequence"),
                ", ".join(f"{k}: {v}" for k, v in (r.get("decisions") or {}).items()),
                r.get("geometry_checks"), r.get("geometry_agreed"), r.get("geometry_crops"),
                r.get("takeover_at"), " > ".join(r.get("recovery_stages") or []) or "-",
                r.get("handed_back_reason")] for r in sup],
              [1.2*cm, 1.6*cm, 8.5*cm, 1.8*cm, 1.8*cm, 1.2*cm, 1.6*cm, 3.5*cm, 3.8*cm]),
        Spacer(1, 10), Paragraph("Claude calls, cost and latency (supervised runs)", styles["Heading2"])]
    data = [["Seed", "Requests", "Failed", "Input tok", "Output tok", "Cache read tok", "Cost (USD)",
             "Latency mean (s)", "Latency max (s)", "Planner median (s)", "Run time (min)"]]
    for r in sup:
        total = (r.get("claude_usage") or {}).get("total", {})
        data.append([r["seed"], total.get("requests", r.get("claude_requests")), total.get("failed"),
                     total.get("input_tokens"), total.get("output_tokens"), total.get("cache_read_input_tokens"),
                     f"{r.get('cost_claude_usd') or 0:.2f}", r.get("claude_latency_mean_s"),
                     r.get("claude_latency_max_s"), r.get("planner_latency_median_s"),
                     (r.get("run_seconds") or 0)/60])
    story.append(table(data))
    tools = {}
    for r in sup:
        for tool, t in ((r.get("claude_usage") or {}).get("by_tool") or {}).items():
            agg = tools.setdefault(tool, dict(requests=0, input_tokens=0, output_tokens=0, latency=0.))
            agg["requests"] += t.get("requests", 0)
            agg["input_tokens"] += t.get("input_tokens", 0)
            agg["output_tokens"] += t.get("output_tokens", 0)
            agg["latency"] += t.get("latency_seconds_total", 0.)
    if tools:
        story += [Spacer(1, 8), Paragraph("By purpose (all supervised runs)", styles["Heading3"]),
                  table([["Tool (purpose)", "Requests", "Input tok", "Output tok", "Mean latency (s)"]] +
                        [[tool, t["requests"], t["input_tokens"], t["output_tokens"],
                          t["latency"]/t["requests"] if t["requests"] else None] for tool, t in tools.items()])]
    story += [Spacer(1, 6), Paragraph(
        f"Prices (USD per million tokens, claude-opus-5-5): input {PRICES['input']}, output "
        f"{PRICES['output']} (thinking billed as output), cache read {PRICES['cache_read']}, cache write "
        f"{PRICES['cache_write']} (assumed 1.25x input). GPU cost is an estimate from wall time at about "
        + ", ".join(f"{name} {price:.2f}" for name, price in GPU_PRICE_PER_HOUR.items())
        + " USD/h for the first requested GPU type; the type actually used is not "
        "recorded in run reports.", small), PageBreak()]

    story.append(Paragraph("Final views", styles["Heading2"]))
    cells = []
    for r in rows:
        thumb = r.get("thumbnail")
        caption = Paragraph(f"seed {r['seed']} {r['kind']}: {r.get('outcome')}", small)
        cells.append([Image(str(thumb), width=6.2*cm, height=4.6*cm) if thumb else Paragraph("no frame", small),
                      caption])
    grid = []
    for i in range(0, len(cells), 4):
        row = cells[i:i+4]
        grid.append([c[0] for c in row] + [""]*(4-len(row)))
        grid.append([c[1] for c in row] + [""]*(4-len(row)))
    if grid:
        story.append(Table(grid, colWidths=[6.6*cm]*4))
    story += [PageBreak(), Paragraph("Limitations", styles["Heading2"])]
    for text in (
        "Five seeds per condition is a small sample: a 3/5 versus 4/5 difference is not statistically "
        "meaningful. The policy is also not repeatable within a seed.",
        "Success is RoboTwin's native checker. The runs stop when it latches, so sustained support after "
        "release is not observed. Public visual support is reported as unknown.",
        "The close cameras were placed from seed 4 and seed 5 entries only. On these seeds the mug or "
        "handle may be partly out of view, which limits the supervisor, not the policy.",
        "Claude geometry proposals are uncalibrated and unreviewed. A takeover additionally requires depth "
        "checks, two-frame agreement below 0.85 model confidence, a kept peg reference, both handle faces "
        "and clearance.",
        "Claude, GPU and wall time include Modal start-up and image builds where they occurred."):
        story.append(Paragraph("&bull; " + text, body))
    story += [Spacer(1, 8), Paragraph("Run folders", styles["Heading2"]),
              table([["Seed", "Condition", "Folder"]] + [[r["seed"], r["kind"], r["folder"]] for r in rows])]
    SimpleDocTemplate(str(root/"report.pdf"), pagesize=landscape(A4), leftMargin=1.2*cm, rightMargin=1.2*cm,
                      topMargin=1.2*cm, bottomMargin=1.2*cm,
                      title=f"Seed evaluation {root.name}").build(story)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--root", type=Path, required=True)
    run_parser.add_argument("--seeds", required=True)
    run_parser.add_argument("--model", default="claude-opus-5-5")
    run_parser.add_argument("--gpu", default="L4,A10G,T4")
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        seeds = [int(s) for s in args.seeds.split(",")]
        result = run_all(args.root, seeds, model=args.model, gpu=args.gpu)
    else:
        result = build_report(args.root)
    print(json.dumps(result["totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
