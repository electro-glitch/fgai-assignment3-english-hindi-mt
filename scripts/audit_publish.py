"""Check the Git index and reachable history before publishing; never print secrets."""
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def audit():
    paths = git('ls-files', '-z').decode().split('\0')
    paths = [p for p in paths if p]
    forbidden_parts = {'.venv', '.cache', '.aws', '.ssh', 'submission', 'checkpoints', '__pycache__'}
    issues = []
    patterns = [
        rb'hf_[A-Za-z0-9]{20,}',
        rb'gh[pousr]_[A-Za-z0-9]{20,}',
        rb'github_pat_[A-Za-z0-9_]{20,}',
        rb'AKIA[0-9A-Z]{16}',
        rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        rb'https?://[^\s/@:]+:[^\s/@]+@',
        rb'(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*[\x22\x27][A-Za-z0-9_+/=-]{20,}[\x22\x27]',
    ]
    for path in paths:
        parts = PurePosixPath(path).parts
        name = parts[-1].lower()
        if forbidden_parts.intersection(parts) or name == '.env' or name.startswith('.env.') or name.endswith(('.pem', '.key')):
            issues.append(f'Excluded path is tracked: {path}')
        content = git('show', ':' + path)
        if any(re.search(pattern, content) for pattern in patterns):
            issues.append(f'Potential credential in indexed file: {path}')
        if path.endswith('.md'):
            text = content.decode('utf-8')
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if '://' in target or target.startswith(('#', 'mailto:')):
                    continue
                target = target.split('#')[0].strip('<>')
                resolved = (ROOT / path).parent.joinpath(target).resolve()
                try:
                    relative = resolved.relative_to(ROOT).as_posix()
                except ValueError:
                    issues.append(f'External local link in {path}')
                    continue
                if relative not in paths:
                    issues.append(f'Unpublished link in {path}: {target}')
    objects = git('rev-list', '--objects', '--all').decode().splitlines()
    blobs = 0
    for obj in objects:
        sha = obj.split(' ', 1)[0]
        if git('cat-file', '-t', sha).strip() != b'blob':
            continue
        blobs += 1
        if any(re.search(pattern, git('cat-file', 'blob', sha)) for pattern in patterns):
            issues.append(f'Potential credential in historical blob: {sha}')
    if issues:
        raise SystemExit('\n'.join(issues))
    print(f'PASS: {len(paths)} indexed files, {blobs} historical blobs; no credential-pattern matches or broken Markdown links.')
    print('Pattern scanning is a precaution, not a guarantee that every possible secret format is detectable.')


if __name__ == '__main__':
    audit()
