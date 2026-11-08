"""Verify exact original inputs or explicitly download only missing pinned versions."""
from pathlib import Path
import argparse
import json
import os
from study.acquisition import acquire, manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,
        default=Path(os.environ.get('ALLOCATION_INPUTS',Path(__file__).resolve().parent/'data/cache')))
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    if args.inputs.is_symlink() or any(p.is_symlink() for p in args.inputs.absolute().parents):
        raise ValueError('Input directory must not use symlinks')
    rows=manifest()
    result=[acquire(row,args.inputs,'download' if args.download else 'verify') for row in rows]
    args.inputs.mkdir(parents=True,exist_ok=True)
    # The loader consumes this neutral provenance view; it changes no expected
    # hash and never edits an agency original.
    (args.inputs/'acquisition.json').write_text(json.dumps({'files':rows},indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
