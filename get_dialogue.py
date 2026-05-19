import argparse
from pathlib import Path

from dialoop.local_tools import DialogueIndex, LabelStore


def get_label_count(labels_path):
    return LabelStore(labels_path).count()


def extract_dialogue_with_line_numbers(text):
    return [(dialogue.line_number, dialogue.text) for dialogue in DialogueIndex.from_text(text).dialogues]


def main():
    parser = argparse.ArgumentParser(description="获取待标注的对话")
    parser.add_argument(
        "--novel", type=Path, default=Path("novel.txt"), help="小说文本路径（默认 novel.txt）"
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("labeled.txt"), help="已标注结果路径（默认 labeled.txt）"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="最大一次性获取的对话数量（默认1）"
    )
    parser.add_argument(
        "--threshold", type=int, default=10, help="视为连续对话的最大行号间隔（默认10）"
    )
    args = parser.parse_args()

    dialogue_index = DialogueIndex.from_file(args.novel)
    now_count = get_label_count(args.labels)

    if now_count >= dialogue_index.total:
        print("已经标注完毕")
        return

    batch = dialogue_index.next_batch(
        labeled_count=now_count,
        batch_size=args.batch_size,
        max_line_gap=args.threshold,
    )

    if len(batch) == 0:
        # 这不应该发生，但为了安全
        print("已经标注完毕")
        return

    # 输出批次信息
    if len(batch) == 1:
        # 单句标注模式
        dialogue = batch[0]
        line_num = dialogue.line_number
        print(f"待标注对话：第{line_num}行「{dialogue.text}」")
        print()
        print(f"请仔细分析 `{args.novel}` 中对应行的上下文，判断说话角色。")
        print(f'然后调用 write_label.py --labels "{args.labels}" --name <角色名>')
    else:
        # 批量标注模式
        print(f"待标注对话批次（{len(batch)}句）：")
        for i, dialogue in enumerate(batch, 1):
            print(f"{i}. 第{dialogue.line_number}行：「{dialogue.text}」")
        print()
        print(f"请仔细分析 `{args.novel}` 中对应行的上下文，判断每句话的说话角色。")
        print(f'然后调用 write_label.py --labels "{args.labels}" --name 角色名1 --name 角色名2 ... 按顺序标注。')


if __name__ == "__main__":
    main()
