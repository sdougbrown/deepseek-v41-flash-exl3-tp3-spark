"""Derive a native recipe from a parent recipe and an edited source tree; no build, SSH or service changes."""
import argparse
import json
from pathlib import Path

try:
    from . import patch
except ImportError:
    import patch


def derive(parent,name,source,components,summary):
    spec,_=patch.recipe(parent)
    if (patch.HERE/'recipes'/f'{name}.json').exists():raise ValueError('Recipe exists: '+name)
    base=json.loads((patch.HERE/'base.json').read_bytes())['files']
    source=Path(source)
    files={}
    for p in sorted(source.rglob('*')):
        rel=p.relative_to(source).as_posix()
        if p.is_symlink():raise ValueError('Symlink in source: '+rel)
        if p.is_file() and not rel.startswith('.git/') and rel!='.git':files[rel]=patch.digest(p.read_bytes())
    overlays={}
    for rel,sha in sorted(files.items()):
        if base.get(rel)==sha:continue
        obj=patch.HERE/'objects'/(sha+Path(rel).suffix)
        if not obj.exists():obj.write_bytes((source/rel).read_bytes())
        overlays[rel]=dict(object=obj.relative_to(patch.HERE).as_posix(),sha256=sha)
    value=dict(spec,recipe=name,release_version='1.2.6-'+name.removeprefix('v126-'),parent_recipe=parent,
               components=spec['components']+[c for c in components if c not in spec['components']],
               qualification=dict(gpu_workflow_passed=False,status='experimental-unqualified',summary=summary),
               remove=sorted(set(base)-set(files)),overlays=overlays,source_files=files,source_tree_sha256=patch.tree_digest(files))
    (patch.HERE/'recipes'/f'{name}.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    changed=sorted(rel for rel,sha in files.items() if spec['source_files'].get(rel)!=sha)
    return dict(recipe=name,source_tree_sha256=value['source_tree_sha256'],changed_from_parent=changed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True);p.add_argument('--name',required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--component',action='append',default=[]);p.add_argument('--summary',required=True)
    a=p.parse_args()
    print(json.dumps(derive(a.parent,a.name,a.source,a.component,a.summary),indent=2))
