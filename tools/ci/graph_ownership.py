"""CPU-only graph-cache ownership probe for a materialized engine source.

Extracts the Carry, round Graphs and draft PassGraphs declarations verbatim,
composes them between the fixture parts in a temporary directory with the real
cuda module, and runs the executable probe with zig. With --capped the round
cache is the capped one (graph_cache.zig) and its eviction cases run too.
Exits nonzero on any probe failure.
"""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def between(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i)]


def extract(source):
    round_source = (source / 'families/dsv41/round.zig').read_text()
    draft_source = (source / 'families/dsv41/draft.zig').read_text()
    carry = between(round_source, 'const Carry = struct {', "\n/// The rounds' stretches")
    round_decl = between(round_source, 'pub const Graphs = struct {', '\n/// A round of these rows')
    draft_decl = between(draft_source, 'pub const PassGraphs = struct {', '\npub const Drafter = struct {')
    return carry, round_decl, draft_decl


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, help='materialized engine source root (zig/src layout)')
    ap.add_argument('--zig', default='zig', help='zig executable (default: zig)')
    ap.add_argument('--capped', action='store_true', help='require the capped round cache and run its eviction cases')
    args = ap.parse_args(argv)
    source = Path(args.source)
    parts = ['cuda', 'families/dsv41/round.zig', 'families/dsv41/draft.zig']
    if args.capped:
        parts.append('families/dsv41/graph_cache.zig')
    for part in parts:
        if not (source / part).exists():
            print(f'error: {source / part} is missing from the source', file=sys.stderr)
            return 2
    try:
        carry, round_decl, draft_decl = extract(source)
    except ValueError as exc:
        print(f'error: declaration marker not found: {exc}', file=sys.stderr)
        return 2
    prefix = (HERE / 'graph_ownership/prefix.zig').read_text()
    checks = (HERE / 'graph_ownership/checks.zig').read_text()
    capped = 'const capped = true;\nconst graph_cache = @import("graph_cache.zig");' if args.capped else 'const capped = false;'
    probe = '\n\n'.join(p.rstrip('\n') for p in (prefix, capped, carry, round_decl, draft_decl, checks)) + '\n'
    with tempfile.TemporaryDirectory(prefix='graph-ownership-probe-') as tmp:
        shutil.copytree(source / 'cuda', Path(tmp) / 'cuda')
        if args.capped:
            shutil.copy(source / 'families/dsv41/graph_cache.zig', Path(tmp) / 'graph_cache.zig')
        (Path(tmp) / 'probe.zig').write_text(probe)
        p = subprocess.run([args.zig, 'run', 'probe.zig', '-lc'], cwd=tmp)
        return p.returncode


if __name__ == '__main__':
    sys.exit(main())
