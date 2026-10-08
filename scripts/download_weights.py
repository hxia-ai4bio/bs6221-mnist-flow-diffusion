"""Download an immutable Hugging Face revision and verify the existing SHA-256 manifest."""
from pathlib import Path
import argparse,json,shutil,sys,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bs6221_flow.runtime import ROOT,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-id');p.add_argument('--revision',help='Full 40-character commit hash')
    a=p.parse_args();config=json.loads((ROOT/'models/hub.json').read_text())
    repo=a.repo_id or config['repo_id'];revision=a.revision or config['revision']
    if not repo or not revision:p.error('Hugging Face repo/revision pending. Supply --repo-id and --revision, or publish first.')
    if len(revision)!=40 or any(x not in '0123456789abcdef' for x in revision.lower()):p.error('Use a full commit hash to pin the weights.')
    snapshot=ROOT/config['local_snapshot'];manifest=json.loads((snapshot/'manifest.json').read_text())
    targets={snapshot/name:record for name,record in manifest['files_sha256'].items()}
    # Existing local files are not overwritten. Detect mismatches before any network work.
    for path,expected in targets.items():
        if path.exists() and sha256(path)!=expected:raise RuntimeError(f'Local file differs; not overwritten: {path}')
    from huggingface_hub import snapshot_download
    with tempfile.TemporaryDirectory(prefix='bs6221-models-') as temp:
        downloaded=Path(snapshot_download(repo_id=repo,revision=revision,allow_patterns=list(manifest['files_sha256']),local_dir=temp))
        for name,expected in manifest['files_sha256'].items():
            path=downloaded/name
            if sha256(path)!=expected:raise RuntimeError(f'Download hash mismatch: {name}')
        snapshot.mkdir(parents=True,exist_ok=True)
        for name in manifest['files_sha256']:
            if not (snapshot/name).exists():shutil.copy2(downloaded/name,snapshot/name)
    # Keep the former package paths compatible with its recorded manifest.
    package=ROOT/'Flow_Group_Package/weights';package.mkdir(parents=True,exist_ok=True)
    for name in ['flow_ema.pt','digit_classifier.pt']:
        target=package/name
        if target.exists() and sha256(target)!=sha256(snapshot/name):raise RuntimeError(f'Package weight differs: {name}')
        if not target.exists():shutil.copy2(snapshot/name,target)
    print('Verified frozen weights at',snapshot,'revision',revision)

if __name__=='__main__':main()
