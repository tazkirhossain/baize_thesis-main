"""Stage the owner's complete worktree, including previously ignored artifacts.

Git's own metadata cannot be tracked as ordinary .git paths. Nested test-repo
metadata is preserved in an archive; the root repository's metadata is never
published. This prepares a local commit only and never pushes or deletes files.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def files_and_repositories():
    files,nested=[],[]
    for parent,dirs,names in os.walk(ROOT,followlinks=False):
        directory=Path(parent)
        if '.git' in dirs:
            if directory!=ROOT:
                nested.append(directory/'.git')
            dirs.remove('.git')
        for name in names:
            path=directory/name
            if name=='.git':
                if directory!=ROOT:
                    nested.append(path)
                continue
            files.append(path)
    return sorted(files),sorted(nested)


def archive_nested_metadata(nested):
    target=ROOT/'data/nested-test-git-metadata.zip'
    manifest=[]
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for metadata in nested:
            # Only historical/generated test repos may be bundled this way.
            relative=metadata.relative_to(ROOT).as_posix()
            if not relative.startswith('data/test-'):
                raise ValueError('unexpected nested Git metadata; inspect before publishing: '+relative)
            sources=[metadata] if metadata.is_file() else sorted(p for p in metadata.rglob('*') if p.is_file())
            for source in sources:
                name=source.relative_to(ROOT).as_posix()
                content=source.read_bytes()
                archive.writestr(name,content)
                manifest.append({'path':name,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)})
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError('nested metadata archive integrity failed')
    record=ROOT/'data/nested-test-git-metadata-manifest.json'
    record.write_text(json.dumps({'repositories':len(nested),'files':manifest},indent=2)+'\n')
    print(f'Preserved {len(nested)} nested test repositories in {target.name}',flush=True)


def main():
    _,nested=files_and_repositories()
    archive_nested_metadata(nested)
    files,_=files_and_repositories()
    large=[path for path in files if path.stat().st_size>=10*1024**2]
    attributes=ROOT/'.gitattributes'
    attributes.write_text('# Preserve data/manifest bytes across operating systems. Large artifacts use LFS.\n'
        'data/** -text\n.pytest_cache/** -text\n'+''.join(
        json.dumps(path.relative_to(ROOT).as_posix(),ensure_ascii=False)+' filter=lfs diff=lfs merge=lfs -text\n'
        for path in large),encoding='utf-8')
    subprocess.run(['git','-C',str(ROOT),'lfs','install','--local'],check=True)
    subprocess.run(['git','-C',str(ROOT),'config','core.longpaths','true'],check=True)
    files,_=files_and_repositories()
    # Git's plumbing stages ordinary files below embedded test repositories
    # without turning those directories into unusable submodule/gitlink entries.
    paths=[p.relative_to(ROOT).as_posix() for p in files]
    encoded='\n'.join(json.dumps(name,ensure_ascii=False) for name in paths)+'\n'
    print(f'Hashing/staging {len(paths):,} files; {len(large)} LFS paths; no ignore filtering.',flush=True)
    result=subprocess.run(['git','-C',str(ROOT),'hash-object','-w','--stdin-paths'],
                          input=encoded.encode('utf-8'),capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8',errors='replace'))
    hashes=result.stdout.splitlines()
    if len(hashes)!=len(paths):
        raise ValueError('Git hashing coverage mismatch')
    entries=b''.join(b'100644 '+digest+b'\t'+name.encode('utf-8')+b'\0' for name,digest in zip(paths,hashes))
    subprocess.run(['git','-C',str(ROOT),'update-index','-z','--index-info'],input=entries,check=True)
    print(f'Staged {len(paths):,} files. Credential and size audits are still required before push.',flush=True)


if __name__=='__main__':
    main()
