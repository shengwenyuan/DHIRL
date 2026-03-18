"""Compute mean of train_ll and test_ll from a CSV file."""
import csv
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python mean_ll.py <path_to_csv>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    with open(path) as f:
        r = csv.DictReader(f)
        rows = list(r)
    train_ll = [float(r["train_ll"]) for r in rows]
    test_ll = [float(r["test_ll"]) for r in rows]
    print("train_ll mean:", sum(train_ll) / len(train_ll))
    print("test_ll mean:", sum(test_ll) / len(test_ll))

if __name__ == "__main__":
    main()
