"""Command-line entry point for the end-to-end pipeline."""

from .pipeline import batch_process, process_image


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Mechanism image -> OCSR -> arrow list -> SM-SMARTS"
    )
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("-o", "--output", default="outputs/pipeline")
    parser.add_argument("--llm-model", default=None,
                        help="Merged Qwen3-VL model or PEFT adapter directory")
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--ocsr-checkpoint", default=None)
    parser.add_argument("--ocsr-repo", default="")
    parser.add_argument("--ocsr-filename", default="molnextr.pth")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--sam-device", default="cuda")
    parser.add_argument("--ocsr-device", default="cpu")
    parser.add_argument("--llm-device", default="cuda")
    args = parser.parse_args()

    common = dict(
        llm_model_path=args.llm_model,
        scale=args.scale,
        sam_device=args.sam_device,
        molnextr_device=args.ocsr_device,
        llm_device=args.llm_device,
        sam_checkpoint=args.sam_checkpoint,
        ocsr_checkpoint=args.ocsr_checkpoint,
        ocsr_repo=args.ocsr_repo,
        ocsr_filename=args.ocsr_filename,
    )
    if os.path.isdir(args.input):
        batch_process(args.input, args.output, **common)
    else:
        process_image(args.input, args.output, **common)


if __name__ == "__main__":
    main()
