import argparse
import platform
from .common import read_json, device_for


def main():
    parser = argparse.ArgumentParser(description='English-Hindi MT: ICT4403 Assignment 3')
    parser.add_argument('stage', choices=['doctor', 'prepare', 'basic', 'mt5', 'llama', 'gemma', 'report'])
    parser.add_argument('--config', default='configs/laptop.json')
    parser.add_argument('--run', default='runs/laptop')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.stage == 'doctor':
        import torch
        print('Python:', platform.python_version(), 'PyTorch:', torch.__version__)
        print('CUDA available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('GPU:', torch.cuda.get_device_name(), 'BF16:', torch.cuda.is_bf16_supported())
        from huggingface_hub import hf_hub_download
        from tempfile import TemporaryDirectory
        for model in ('google/mt5-small', 'meta-llama/Llama-3.2-1B-Instruct', 'google/gemma-3-1b-it'):
            try:
                with TemporaryDirectory() as folder:
                    hf_hub_download(model, 'config.json', cache_dir=folder)
                print(model, 'ACCESS OK')
            except Exception as exc:
                print(model, 'ACCESS NOT VERIFIED:', type(exc).__name__)
        return
    config = read_json(args.config)
    if args.stage == 'prepare':
        from .data import prepare
        prepare(config, args.run)
    elif args.stage == 'report':
        from .report import report
        report(args.run)
    else:
        from .experiment import experiment
        experiment(args.stage, config, args.run, device_for(args.device))


if __name__ == '__main__':
    main()
