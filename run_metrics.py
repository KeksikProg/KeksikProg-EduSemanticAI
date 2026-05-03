import argparse
from pathlib import Path

from semantic_metrics import analyze_document_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-3 semantic metrics and save JSON report.")
    parser.add_argument("--meta", default="out/meta.json", help="Path to meta.json")
    parser.add_argument("--emb", default="out/emb_FRIDA.npy", help="Path to embeddings .npy")
    parser.add_argument("--out", default="out/document_metrics_report_v3.json", help="Output report path")
    args = parser.parse_args()

    report = analyze_document_metrics(
        meta_path=Path(args.meta),
        emb_path=Path(args.emb),
        out_path=Path(args.out),
    )

    print("saved:", args.out)
    print("model:", report["model"])
    print("version:", report["version"])
    print("n_segments:", report["n_segments"])
    print("risk_score_v3:", round(float(report["summary"]["risk_score_v3"]), 2))


if __name__ == "__main__":
    main()
