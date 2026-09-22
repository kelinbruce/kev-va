import re

with open("openspec/changes/add-kev-qwen3-decision-serving/tasks.md", "r") as f:
    lines = f.readlines()

def should_check(line):
    m = re.match(r'- \[ \] (\d+\.\d+) ', line)
    if not m:
        return False
    task_num = m.group(1)
    if task_num.startswith('8.'): return True
    return False

for i in range(len(lines)):
    if should_check(lines[i]):
        lines[i] = lines[i].replace('- [ ]', '- [x]')

with open("openspec/changes/add-kev-qwen3-decision-serving/tasks.md", "w") as f:
    f.writelines(lines)

