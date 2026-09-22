import re

with open("openspec/changes/add-kev-qwen3-decision-serving/tasks.md", "r") as f:
    lines = f.readlines()

def should_check(line):
    # Check off tasks in parts 2, 3 (except 3.4), 4, 5, 6 (except 6.4)
    m = re.match(r'- \[ \] (\d+\.\d+) ', line)
    if not m:
        return False
    task_num = m.group(1)
    if task_num.startswith('2.'): return True
    if task_num in ['3.1', '3.2', '3.3']: return True
    if task_num.startswith('4.'): return True
    if task_num.startswith('5.'): return True
    if task_num in ['6.1', '6.2', '6.3']: return True
    return False

for i in range(len(lines)):
    if should_check(lines[i]):
        lines[i] = lines[i].replace('- [ ]', '- [x]')

with open("openspec/changes/add-kev-qwen3-decision-serving/tasks.md", "w") as f:
    f.writelines(lines)

