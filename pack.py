"""Build an allowlisted cloud handoff with prepared data and no reference essays."""
import hashlib
from pathlib import Path
import tarfile

from common import config
from train import validate_data


def main():
    validate_data(config())
    files = [Path(name) for name in (
        'README.md', 'SOURCES.md', 'config.json', 'style_pairs.jsonl', 'eval_prompts.jsonl',
        'pyproject.toml', 'uv.lock', '.python-version',
        'common.py', 'style.py', 'prepare.py', 'train.py', 'chat.py', 'evaluate.py',
        'collect.py', 'reference.py', 'pack.py', 'tests/test_pipeline.py',
    )]
    files += sorted(p for folder in ('data/prepared', 'data/tokenizer')
                    for p in Path(folder).rglob('*') if p.is_file())
    output = Path('rationalist-lora-cloud.tar.gz')
    with tarfile.open(output, 'w:gz') as archive:
        for path in files:
            archive.add(path, arcname='rationalist-lora/' + path.as_posix(), recursive=False)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + '.sha256').write_text(f'{checksum}  {output.name}\n')
    print(f'{output}: {output.stat().st_size / 1024**2:.1f} MiB, {len(files)} files')
    print(f'SHA256 {checksum}')


if __name__ == '__main__':
    main()
