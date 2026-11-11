"""Command line entry point.

python -m neo_sentinel download            fetch and verify the dataset
python -m neo_sentinel train               train and save models/neo_sentinel.joblib
python -m neo_sentinel experiment          run the leakage and approach history experiments
python -m neo_sentinel serve               start the API on $PORT (default 8000)
python -m neo_sentinel live                score this week's real close approaches
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date

from neo_sentinel.config import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, DEFAULT_TARGET_RECALL


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="neo_sentinel", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download and verify the NASA dataset")
    p.add_argument("--dest", default=str(DEFAULT_DATA_PATH))
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("train", help="train, compare and save the model")
    p.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    p.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    p.add_argument("--n-iter", type=int, default=12, help="random search iterations per model")
    p.add_argument("--only", nargs="*", help="limit candidates, e.g. --only hist_gradient_boosting")

    p = sub.add_parser("experiment", help="measure split leakage and what approach history adds")
    p.add_argument("--data", default=str(DEFAULT_DATA_PATH))

    p = sub.add_parser("serve", help="run the API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))

    p = sub.add_parser("live", help="score real close approaches from NASA's feed")
    p.add_argument("--start", type=date.fromisoformat, default=date.today())

    args = parser.parse_args(argv)

    if args.command == "download":
        from neo_sentinel.data import download_dataset

        download_dataset(args.dest, force=args.force)
    elif args.command == "train":
        from neo_sentinel.train import train

        train(args.data, args.out, args.target_recall, args.n_iter, only=args.only)
    elif args.command == "experiment":
        from neo_sentinel.experiments import run, run_leakage

        run(args.data)
        run_leakage(args.data)
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("neo_sentinel.api.app:app", host=args.host, port=args.port)
    elif args.command == "live":
        import pandas as pd

        from neo_sentinel.bundle import ModelBundle
        from neo_sentinel.config import model_path
        from neo_sentinel.neows import fetch_feed

        bundle = ModelBundle.load(model_path())
        scored = bundle.score(fetch_feed(args.start)).sort_values("hazard_probability", ascending=False)
        cols = ["close_approach_date", "name", "hazard_probability", "risk_tier", "nasa_hazardous"]
        with pd.option_context("display.width", 120, "display.max_rows", 40):
            print(scored[cols].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
