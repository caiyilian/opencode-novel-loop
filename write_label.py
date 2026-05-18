import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="追加名字到 labeled.txt")
    parser.add_argument(
        "--labels", type=Path, default=Path("labeled.txt"), help="标注结果路径（默认 labeled.txt）"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        action="append",
        help="要追加的名字（可以多次使用，按顺序）",
    )
    args = parser.parse_args()

    names = args.name

    # 追加到文件，每个名字一行
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    with args.labels.open("a", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")

    print(f"已标注 {len(names)} 个角色：{', '.join(names)}")


if __name__ == "__main__":
    main()
