"""Generate exercise plans via Ollama API for each variant x profile x model x run."""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import urllib.error
import sys

sys.path.insert(0, str(Path(__file__).parent))
from constants import (
    OLLAMA_MODELS,
    GENERATION_TEMPERATURE,
    GENERATION_RUNS,
    GENERATION_SEED,
)
from profiles import PROFILES, get_all_profile_ids
from prompts import build_prompt, get_variant_names


def call_ollama(model: str, prompt: str, base_url: str = "http://localhost:11434",
                temperature: float = GENERATION_TEMPERATURE,
                seed: int = GENERATION_SEED) -> dict:
    """Call Ollama generate API and return response + full metrics."""
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return {"error": str(e), "duration_ms": 0}
    except Exception as e:
        return {"error": str(e), "duration_ms": 0}

    duration_ms = int((time.time() - start_time) * 1000)

    return {
        "response": result.get("response", ""),
        "duration_ms": duration_ms,
        "total_duration_ns": result.get("total_duration", 0),
        "load_duration_ns": result.get("load_duration", 0),
        "prompt_eval_count": result.get("prompt_eval_count", 0),
        "prompt_eval_duration_ns": result.get("prompt_eval_duration", 0),
        "eval_count": result.get("eval_count", 0),
        "eval_duration_ns": result.get("eval_duration", 0),
    }


def generate_plan(model: str, profile_id: str, variant: str, run: int,
                  output_dir: Path, base_url: str,
                  skip_existing: bool = True) -> dict | None:
    """Generate a single exercise plan and save to JSON."""
    model_safe = model.replace(":", "_").replace("/", "_")
    out_file = output_dir / variant / model_safe / f"{profile_id}_run{run}.json"

    tag = f"{variant}/{model}/{profile_id}/run{run}"

    if skip_existing and out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("response") and not existing.get("error"):
                print(f"  [{tag}] SKIP (exists)", flush=True)
                return None
        except (json.JSONDecodeError, KeyError):
            pass

    profile = PROFILES[profile_id]
    prompt = build_prompt(profile, variant=variant)

    seed = GENERATION_SEED + run

    print(f"  [{tag}]...", end=" ", flush=True)

    result = call_ollama(model, prompt, base_url, seed=seed)

    if "error" in result and result["error"]:
        print(f"ERROR: {result['error']}")
    else:
        eval_count = result.get("eval_count", 0)
        eval_dur_ns = result.get("eval_duration_ns", 0)
        tok_per_sec = (eval_count / (eval_dur_ns / 1e9)) if eval_dur_ns > 0 else 0
        response_len = len(result.get("response", ""))
        print(f"OK ({result['duration_ms']}ms, {eval_count} tok, "
              f"{tok_per_sec:.1f} tok/s, {response_len} chars)")

    record = {
        "profile_id": profile_id,
        "model": model,
        "variant": variant,
        "run": run,
        "prompt": prompt,
        "response": result.get("response", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": result.get("duration_ms", 0),
        "total_duration_ns": result.get("total_duration_ns", 0),
        "load_duration_ns": result.get("load_duration_ns", 0),
        "prompt_eval_count": result.get("prompt_eval_count", 0),
        "prompt_eval_duration_ns": result.get("prompt_eval_duration_ns", 0),
        "eval_count": result.get("eval_count", 0),
        "eval_duration_ns": result.get("eval_duration_ns", 0),
        "temperature": GENERATION_TEMPERATURE,
        "seed": seed,
        "error": result.get("error"),
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return record


def main():
    ap = argparse.ArgumentParser(description="Generate exercise plans via Ollama")
    ap.add_argument("--models", type=str, default=None,
                    help="Comma-separated model names (default: all 4)")
    ap.add_argument("--profiles", type=str, default=None,
                    help="Comma-separated profile IDs (default: all 8)")
    ap.add_argument("--variants", type=str, default=None,
                    help="Comma-separated prompt variants (default: all 3)")
    ap.add_argument("--runs", type=int, default=GENERATION_RUNS,
                    help=f"Runs per combo (default: {GENERATION_RUNS})")
    ap.add_argument("--output-dir", type=str, default="results/plans",
                    help="Output directory")
    ap.add_argument("--base-url", type=str, default="http://localhost:11434",
                    help="Ollama API base URL")
    args = ap.parse_args()

    models = args.models.split(",") if args.models else OLLAMA_MODELS
    profile_ids = args.profiles.split(",") if args.profiles else get_all_profile_ids()
    variants = args.variants.split(",") if args.variants else get_variant_names()
    output_dir = Path(args.output_dir)

    total = len(variants) * len(models) * len(profile_ids) * args.runs
    print(f"=== Exercise Plan Generation ===")
    print(f"Variants: {variants}")
    print(f"Models:   {models}")
    print(f"Profiles: {profile_ids}")
    print(f"Runs:     {args.runs}")
    print(f"Total:    {total} plans")
    print(f"Output:   {output_dir}")
    print(f"Ollama:   {args.base_url}")
    print()

    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_csv = log_dir / "generation_log.csv"
    log_fields = [
        "model", "profile_id", "variant", "run", "timestamp",
        "duration_ms", "total_duration_ns", "load_duration_ns",
        "prompt_eval_count", "prompt_eval_duration_ns",
        "eval_count", "eval_duration_ns",
        "response_length", "is_refusal", "error",
    ]
    log_rows = []

    completed = 0
    skipped = 0
    errors = 0
    start_all = time.time()

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        for variant in variants:
            print(f"\n  --- Variant: {variant} ---")
            for profile_id in profile_ids:
                for run in range(1, args.runs + 1):
                    record = generate_plan(
                        model, profile_id, variant, run, output_dir, args.base_url
                    )
                    if record is None:
                        skipped += 1
                        continue

                    if record.get("error"):
                        errors += 1
                    else:
                        completed += 1

                    response = record.get("response", "")
                    is_refusal = (
                        len(response) < 100
                        or "cannot" in response.lower()[:200]
                        or "i'm not able" in response.lower()[:200]
                    )
                    log_rows.append({
                        "model": model,
                        "profile_id": profile_id,
                        "variant": variant,
                        "run": run,
                        "timestamp": record.get("timestamp", ""),
                        "duration_ms": record.get("duration_ms", 0),
                        "total_duration_ns": record.get("total_duration_ns", 0),
                        "load_duration_ns": record.get("load_duration_ns", 0),
                        "prompt_eval_count": record.get("prompt_eval_count", 0),
                        "prompt_eval_duration_ns": record.get("prompt_eval_duration_ns", 0),
                        "eval_count": record.get("eval_count", 0),
                        "eval_duration_ns": record.get("eval_duration_ns", 0),
                        "response_length": len(response),
                        "is_refusal": is_refusal,
                        "error": record.get("error", ""),
                    })

        if model != models[-1]:
            print(f"\nPausing 10s before next model...")
            time.sleep(10)

    elapsed = time.time() - start_all

    if log_rows:
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"\nSaved generation log: {log_csv}")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_planned": total,
        "completed": completed,
        "skipped": skipped,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_minutes": round(elapsed / 60, 1),
        "models": models,
        "variants": variants,
        "profiles": profile_ids,
        "runs": args.runs,
        "temperature": GENERATION_TEMPERATURE,
        "base_seed": GENERATION_SEED,
        "ollama_url": args.base_url,
    }

    if log_rows:
        model_stats = {}
        for model in models:
            rows = [r for r in log_rows if r["model"] == model]
            if not rows:
                continue
            model_stats[model] = {
                "plans_generated": len(rows),
                "refusals": sum(1 for r in rows if r["is_refusal"]),
                "avg_duration_ms": round(sum(r["duration_ms"] for r in rows) / len(rows)),
                "avg_eval_count": round(sum(r["eval_count"] for r in rows) / len(rows)),
                "avg_response_length": round(sum(r["response_length"] for r in rows) / len(rows)),
                "total_tokens": sum(r["eval_count"] for r in rows),
            }
        summary["model_stats"] = model_stats

    summary_path = log_dir / "generation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved generation summary: {summary_path}")

    print(f"\n{'='*60}")
    print(f"Done: {completed} generated | {skipped} skipped | {errors} errors")
    print(f"Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} h)")


if __name__ == "__main__":
    main()
