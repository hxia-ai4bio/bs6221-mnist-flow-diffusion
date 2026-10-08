"""Create a reviewable inference-only model release; does not contact Hugging Face."""
from pathlib import Path
import json,shutil,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bs6221_flow.runtime import ROOT,sha256


def main():
    snapshot=ROOT/'checkpoints/frozen/20261002_flow15000_ddpm20000'
    manifest=json.loads((snapshot/'manifest.json').read_text())
    out=ROOT/'outputs/hf_release';out.mkdir(parents=True,exist_ok=True)
    for name,expected in manifest['files_sha256'].items():
        if sha256(snapshot/name)!=expected:raise RuntimeError(f'Frozen weight changed: {name}')
        shutil.copy2(snapshot/name,out/name)
    shutil.copy2(snapshot/'manifest.json',out/'manifest.json')
    (out/'config.json').write_text(json.dumps({'architecture':'conditional_tiny_unet_v1','channels':24,'num_classes':10,'image_shape':[1,28,28],
        'normalization':'pixel / 127.5 - 1','flow_training_steps':15000,'ddpm_training_steps':20000,'weights':'EMA','ddpm_steps':100},indent=2))
    (out/'README.md').write_text((ROOT/'docs/model_card.md').read_text())
    print('Hugging Face release ready:',out)

if __name__=='__main__':main()
