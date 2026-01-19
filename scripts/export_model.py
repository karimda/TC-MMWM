"""
Export TC-MMWM Model for Deployment
-----------------------------------
This script exports a trained TC-MMWM model to TorchScript or ONNX format
for deployment on embedded systems or real-time inference pipelines.
"""

import torch
import argparse
from tc_mmwm.models.tc_mmwm import TC_MMWM
from tc_mmwm.utils.checkpointing import load_checkpoint
from tc_mmwm.preprocessing.build_dataset import MultiModalDataset
from tc_mmwm.utils.logging import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Export TC-MMWM Model")
    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output file path (without extension)')
    parser.add_argument('--format', type=str, default='torchscript', choices=['torchscript', 'onnx'], help='Export format')
    parser.add_argument('--device', type=str, default='cuda', help='Device for export')
    return parser.parse_args()


def export_torchscript(model, dummy_input, output_path):
    """
    Export the model to TorchScript format
    """
    scripted_model = torch.jit.trace(model, dummy_input)
    scripted_model.save(output_path + '.pt')
    print(f"TorchScript model saved to {output_path}.pt")


def export_onnx(model, dummy_input, output_path):
    """
    Export the model to ONNX format
    """
    torch.onnx.export(
        model,
        dummy_input,
        output_path + '.onnx',
        input_names=list(dummy_input.keys()),
        output_names=['latent_state', 'action'],
        opset_version=14,
        dynamic_axes={k: {0: 'batch'} for k in dummy_input.keys()}
    )
    print(f"ONNX model saved to {output_path}.onnx")


def main():
    args = parse_args()
    logger = setup_logger()
    logger.info(f"Exporting TC-MMWM model to {args.format} format on {args.device}")

    # Load dummy input for tracing
    dataset = MultiModalDataset(config_path=args.config, split='test')
    dummy_batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    dummy_batch = {k: v.to(args.device) for k, v in dummy_batch.items()}

    # Load model
    model = TC_MMWM(config_path=args.config).to(args.device)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Export
    if args.format == 'torchscript':
        export_torchscript(model, dummy_batch, args.output)
    elif args.format == 'onnx':
        export_onnx(model, dummy_batch, args.output)
    else:
        raise ValueError(f"Unsupported export format: {args.format}")


if __name__ == "__main__":
    main()
