import sys
import pathlib

for p in sys.argv[1:]:
    path = pathlib.Path(p)
    if not path.exists():
        print(f"skip: {p} not found")
        continue
    content = path.read_bytes()
    fixed = content.replace(b'\r\n', b'\n')
    if fixed != content:
        path.write_bytes(fixed)
        print(f"fixed: {p}")
    else:
        print(f"ok: {p}")
