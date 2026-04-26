import json
import os

input_path = 'src/environment/training_samples.jsonl'
output_path = 'src/environment/training_samples_final.jsonl'

print(f"Reading {input_path}...")
with open(input_path, 'r', encoding='utf-8') as f:
    content = f.read().strip()

# Restore brackets if missing
if not content.startswith('['):
    content = '[' + content
if not content.endswith(']'):
    content = content + ']'

try:
    data = json.loads(content)
except Exception as e:
    # If the trailing comma exists, try to remove it
    if content.endswith(',]'):
        content = content[:-2] + ']'
        data = json.loads(content)
    else:
        raise e

print(f"Converting {len(data)} entries...")
with open(output_path, 'w', encoding='utf-8') as f:
    for entry in data:
        text_content = (
            f"### State\n"
            f"Confusion: {entry.get('confusion')}\n"
            f"Action: {entry.get('action')}\n"
            f"Reward: {entry.get('reward')}\n"
            f"Next Confusion: {entry.get('next_confusion')}"
        )
        json.dump({"text": text_content}, f, ensure_ascii=False)
        f.write('\n')

print(f"Done! Successfully created {output_path}")
