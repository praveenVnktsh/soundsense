import pandas as pd
import os
import numpy as np
import csv


def main(args):
    rows = pd.read_csv(args.csv_file, header=None)
    print(len(rows), "episodes in total")
    np.random.seed(0)
    rval = np.random.uniform(size=len(rows))
    pos_idx, neg_idx = rval > 0.2, rval < 0.2
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    rows[pos_idx].to_csv(os.path.join(args.output_dir, "train.csv"), index=False, header=False)
    rows[neg_idx].to_csv(os.path.join(args.output_dir, "val.csv"), index=False, header=False)


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--csv_file",
        type=str,
        help="csv file to read in ",
        default="episode_names_to_csv.csv",
    )
    argparser.add_argument("--output_dir", type=str, default="data")

    args = argparser.parse_args()

    main(args)