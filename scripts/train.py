"""Optional training CLI. Run from any location; no notebook required."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bs6221_flow.training import train_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['flow','diffusion'],default='flow')
    budget=parser.add_mutually_exclusive_group()
    budget.add_argument('--epochs',type=int);budget.add_argument('--steps',type=int)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--ema-decay',type=float,default=.995)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--validation-every',type=int,default=250)
    parser.add_argument('--validation-samples',type=int,default=256)
    parser.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    parser.add_argument('--run-dir',type=Path)
    a=parser.parse_args()
    out,history=train_model(a.model,a.epochs,a.steps,a.batch_size,a.lr,a.ema_decay,a.seed,
                      a.validation_every,a.validation_samples,a.device,a.run_dir)
    from bs6221_flow.output import configure_output
    from bs6221_flow.presentation import show_training
    configure_output(out)
    show_training(history=history,kind=a.model)
    print('Training outputs:',out)

if __name__=='__main__':main()
