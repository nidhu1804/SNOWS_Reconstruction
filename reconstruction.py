"""
SNOWS Reconstruction: Apply pruning mask and reconstruct model weights
Fixed version with proper SNOWS-style block-wise reconstruction

USAGE:
  python3 experiments/vit_pruning/reconstruction.py \\
    --adapter experiments/vit_pruning/artifacts/YOUR_ADAPTER/adapter.pt \\
    --mask experiments/vit_pruning/artifacts/YOUR_MASK/ffn_prune_masks.json \\
    --dataset cifar100 \\
    --output experiments/vit_pruning/reconstructed_model
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoModelForImageClassification
import json
import os
from torch.autograd import Variable


# ============= CONFIGURATION =============
def get_default_config():
    """Default configuration - can be overridden by command line args"""
    return {
        "base_model_pt": "experiments/vit_pruning/artifacts/20251024-215318/adapter.pt",
        "base_model_arch": "google/vit-base-patch16-224",
        "mask_json_path": "experiments/vit_pruning/artifacts/20251101-132101/ffn_prune_masks.json",
        "output_model_dir": "experiments/vit_pruning/reconstructed_model",
        "dataset": "cifar100",
        "test_pct": 0.2,
        "calibration_samples": 512,
        "newton_steps": 5,
        "cg_iters": 100,
        "batch_size": 32,
        "eval_batches": 5,
    }


# ============= HESSIAN-VECTOR PRODUCT UTILITIES =============

def hessian_vector_product_chunks(grad_W, W, vector, mask, max_chunk_size=5e4):
    """Compute Hessian-vector product in chunks to save memory."""
    device = W.device
    vector = Variable(vector).to(device)
    full_vector = torch.zeros_like(W, device=device)
    full_vector[mask] = vector

    hvp = torch.zeros_like(W, device=device)
    num_elements = int(mask.sum().item())
    if num_elements == 0:
        return hvp[mask]

    num_chunks = int(max(1, (num_elements + max_chunk_size - 1) // max_chunk_size))
    chunk_size = int((num_elements + num_chunks - 1) // num_chunks)
    idx = torch.nonzero(mask.flatten(), as_tuple=False).flatten().to(device)

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_elements)
        if start >= end:
            break
        cur_idx = idx[start:end]
        chunk_mask = torch.zeros_like(W, dtype=torch.bool, device=device).flatten()
        chunk_mask[cur_idx] = True
        chunk_mask = chunk_mask.view_as(W)
        chunk_hvp = torch.autograd.grad(
            torch.sum(grad_W * full_vector * chunk_mask),
            W,
            retain_graph=True
        )[0]
        hvp[chunk_mask] += chunk_hvp[chunk_mask]
    return hvp[mask]


def conjugate_gradient_sparse(hvp_fn, b, tol=1e-3, max_iter=100, lambda_reg=1e-4):
    """Solve Hx = b using conjugate gradient method with regularization."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = torch.sum(r * r)

    if rs_old == 0:
        return x

    for iteration in range(max_iter):
        Ap = hvp_fn(p) + lambda_reg * p
        denom = torch.sum(p * Ap)

        if denom.abs() < 1e-12:
            break

        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r)

        residual = torch.sqrt(rs_new).item()
        if residual < tol:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

        # Memory cleanup every 20 iterations
        if iteration % 20 == 0:
            del Ap
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return x


# ============= BLOCK-WISE RECONSTRUCTION =============

def reconstruct_layer_blockwise(base_model, pruned_model, param_name, mask,
                                 xdata, device='cuda', newton_steps=5,
                                 cg_iters=100, batch_size=32):
    """
    Reconstruct a single layer using block-wise output matching.
    This matches the SNOWS approach more closely.
    """
    base_model.eval()
    pruned_model.eval()

    # Get the parameter to reconstruct
    pruned_params = dict(pruned_model.named_parameters())
    if param_name not in pruned_params:
        raise KeyError(f"Parameter {param_name} not found")

    W_sparse = pruned_params[param_name]
    mask = mask.to(device).bool()

    num_masked = mask.sum().item()
    print(f"    Masked elements: {num_masked} / {mask.numel()} ({100*num_masked/mask.numel():.2f}%)")

    if num_masked == 0:
        print(f"    No elements to reconstruct, skipping...")
        return W_sparse.detach().cpu().clone()

    # Use smaller batches for stability
    effective_batch_size = min(batch_size, 16)
    num_batches = max(1, xdata.size(0) // effective_batch_size)

    # Reserve validation data for line search
    val_size = min(32, xdata.size(0) // 4)
    val_data = xdata[-val_size:].to(device)
    train_data = xdata[:-val_size]

    print(f"    Using {train_data.size(0)} train samples, {val_data.size(0)} val samples")

    alpha = 1.0

    # Newton optimization loop
    for step in range(newton_steps):
        print(f"    Newton step {step+1}/{newton_steps}...")

        # Accumulate gradients over multiple batches for better signal
        accumulated_loss = 0.0
        accumulated_grad = None

        # Use first few batches for gradient computation
        num_grad_batches = min(3, num_batches)

        for batch_idx in range(num_grad_batches):
            start_idx = batch_idx * effective_batch_size
            end_idx = min(start_idx + effective_batch_size, train_data.size(0))
            x_batch = train_data[start_idx:end_idx].to(device)

            # Forward pass through both models
            with torch.no_grad():
                out_dense = base_model(pixel_values=x_batch)
                y_dense = out_dense.logits if hasattr(out_dense, "logits") else out_dense

            # Clear any cached gradients
            if W_sparse.grad is not None:
                W_sparse.grad = None

            out_sparse = pruned_model(pixel_values=x_batch)
            y_sparse = out_sparse.logits if hasattr(out_sparse, "logits") else out_sparse

            # MSE loss between outputs
            loss = torch.nn.functional.mse_loss(y_sparse, y_dense)
            accumulated_loss += loss.item()

            # Compute gradient w.r.t. W_sparse
            try:
                grad_W = torch.autograd.grad(loss, W_sparse, create_graph=True, retain_graph=True)[0]

                if accumulated_grad is None:
                    accumulated_grad = grad_W
                else:
                    accumulated_grad = accumulated_grad + grad_W

            except RuntimeError as e:
                if "flash_attention" in str(e).lower():
                    print(f"      [WARN] Flash attention backward not available, using alternative...")
                    # Use L1 loss as alternative (doesn't need second derivatives)
                    loss = torch.nn.functional.l1_loss(y_sparse, y_dense)
                    grad_W = torch.autograd.grad(loss, W_sparse, retain_graph=False)[0]

                    if accumulated_grad is None:
                        accumulated_grad = grad_W
                    else:
                        accumulated_grad = accumulated_grad + grad_W
                else:
                    raise e

            # Clean up
            del out_sparse, y_sparse, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Average gradient over batches
        if accumulated_grad is not None:
            accumulated_grad = accumulated_grad / num_grad_batches
            avg_loss = accumulated_loss / num_grad_batches
        else:
            print(f"      [ERROR] No gradients computed, skipping step")
            break

        print(f"      Avg Loss: {avg_loss:.6f}")

        # Extract gradient for masked elements
        b = -accumulated_grad[mask]

        # Solve for Newton step using CG
        print(f"      Computing Newton step via CG...")
        try:
            # For CPU, we need to handle the case where create_graph=True fails
            if device == 'cpu':
                # Use simpler gradient descent instead of Newton's method on CPU
                newton_step_masked = b  # Just use gradient direction
                print(f"      Using gradient descent (Newton's method unavailable on CPU)")
            else:
                newton_step_masked = conjugate_gradient_sparse(
                    lambda v: hessian_vector_product_chunks(accumulated_grad, W_sparse, v, mask, max_chunk_size=1e4),
                    b,
                    max_iter=cg_iters,
                    tol=1e-3
                )
        except Exception as e:
            print(f"      [WARN] CG/Hessian failed: {e}")
            # Fall back to gradient descent
            newton_step_masked = b
            print(f"      Using gradient descent fallback")

        # Construct full step
        full_newton_step = torch.zeros_like(W_sparse, device='cpu')
        full_newton_step[mask.cpu()] = newton_step_masked.cpu()
        full_newton_step = full_newton_step.to(device)

        # Line search on validation set
        W_original = W_sparse.data.clone()

        # Compute baseline validation loss
        with torch.no_grad():
            out_dense_val = base_model(pixel_values=val_data)
            y_dense_val = out_dense_val.logits if hasattr(out_dense_val, "logits") else out_dense_val

        out_sparse_val = pruned_model(pixel_values=val_data)
        y_sparse_val = out_sparse_val.logits if hasattr(out_sparse_val, "logits") else out_sparse_val
        baseline_loss = torch.nn.functional.mse_loss(y_sparse_val, y_dense_val).item()

        # Backtracking line search
        alpha_current = alpha
        best_alpha = 0.0
        best_loss = baseline_loss

        for bt in range(10):
            # Try this step size
            W_new = W_original + alpha_current * full_newton_step
            with torch.no_grad():
                W_sparse.data = W_new

            # Evaluate on validation set
            out_sparse_val = pruned_model(pixel_values=val_data)
            y_sparse_val = out_sparse_val.logits if hasattr(out_sparse_val, "logits") else out_sparse_val
            new_loss = torch.nn.functional.mse_loss(y_sparse_val, y_dense_val).item()

            # Accept if loss decreased
            if new_loss < best_loss:
                best_loss = new_loss
                best_alpha = alpha_current

            # Sufficient decrease check
            if new_loss < baseline_loss - 1e-4 * alpha_current * torch.sum(accumulated_grad[mask] * newton_step_masked).item():
                alpha = alpha_current
                break

            alpha_current *= 0.7

            if alpha_current < 1e-4:
                # Restore best found
                with torch.no_grad():
                    W_sparse.data = W_original + best_alpha * full_newton_step
                alpha = best_alpha
                break

        print(f"      Step size: {alpha:.4f}, Loss: {baseline_loss:.6f} -> {best_loss:.6f}")

        # Clean up
        del full_newton_step, W_original, accumulated_grad
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Early stopping if no progress
        if best_alpha < 1e-4 and step > 0:
            print(f"      Early stopping: step size too small")
            break

    return W_sparse.detach().cpu().clone()


# ============= EVALUATION FUNCTIONS =============

@torch.no_grad()
def evaluate_top1(model, dataloader, device='cuda', max_batches=None, progress=False):
    """Evaluate top-1 accuracy."""
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(dataloader, desc="Evaluating", leave=False)
        except:
            iterator = dataloader
    else:
        iterator = dataloader

    batch_count = 0
    for batch in iterator:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def measure_latency(model, device='cuda', warmup=3, iters=10):
    """Measure inference latency per image."""
    import time
    model.eval()
    model.to(device)

    if torch.cuda.is_available() and device == 'cuda':
        torch.cuda.synchronize()

    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224, device=device)
        for _ in range(warmup):
            _ = model(pixel_values=dummy)

        if torch.cuda.is_available() and device == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(iters):
            _ = model(pixel_values=dummy)

        if torch.cuda.is_available() and device == 'cuda':
            torch.cuda.synchronize()

    return (time.time() - start) / iters


# ============= UTILITY FUNCTIONS =============

def get_num_classes_from_dataset(dataset):
    """Get number of classes from dataset name."""
    if dataset.lower() == "cifar100":
        return 100
    elif dataset.lower() == "cifar10":
        return 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_model_with_correct_head(architecture, checkpoint_path, num_classes, device='cuda'):
    """Load model with correct classifier head."""
    print(f"Loading base model and adapter checkpoint...")

    model = AutoModelForImageClassification.from_pretrained(architecture)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint to be a dict, got {type(checkpoint)}")

    print(f"\n[Checkpoint Info]")
    print(f"  Keys: {list(checkpoint.keys())}")

    saved_num_labels = checkpoint.get("num_labels", None)
    classifier_type = checkpoint.get("classifier_type", "Linear")
    extra = checkpoint.get("extra", {})

    if saved_num_labels:
        print(f"  Saved num_labels: {saved_num_labels}")
    if extra:
        print(f"  Extra metadata: {extra}")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        print(f"  Format: Full model checkpoint")
        model.load_state_dict(state_dict, strict=False)

    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        classifier_state = checkpoint["state_dict"]
        print(f"  Format: Classifier-only checkpoint")
        print(f"  Classifier keys: {list(classifier_state.keys())}")

        if "weight" in classifier_state:
            weight_shape = classifier_state["weight"].shape
            print(f"  Classifier weight shape: {weight_shape}")

            if classifier_type == "Linear":
                out_features, in_features = weight_shape
                model.classifier = nn.Linear(in_features, out_features)
                model.classifier.weight.data = classifier_state["weight"]
                if "bias" in classifier_state:
                    model.classifier.bias.data = classifier_state["bias"]
                model.config.num_labels = out_features
                print(f"  ✓ Loaded Linear classifier: {in_features} → {out_features}")

    elif "weight" in checkpoint and "bias" in checkpoint:
        weight_shape = checkpoint["weight"].shape
        print(f"  Format: Direct classifier weights")
        out_features, in_features = weight_shape
        model.classifier = nn.Linear(in_features, out_features)
        model.classifier.weight.data = checkpoint["weight"]
        model.classifier.bias.data = checkpoint["bias"]
        model.config.num_labels = out_features
        print(f"  ✓ Loaded Linear classifier: {in_features} → {out_features}")

    model.to(device)

    actual_num_classes = model.classifier.out_features
    print(f"\n[Model Configuration]")
    print(f"  Architecture: {architecture}")
    print(f"  Classifier: Linear({model.classifier.in_features} → {actual_num_classes})")
    print(f"  Config num_labels: {model.config.num_labels}")

    if actual_num_classes != num_classes:
        raise ValueError(
            f"\n❌ FATAL: Classifier mismatch!\n"
            f"  Loaded: {actual_num_classes} classes, Dataset: {num_classes} classes"
        )

    print(f"  ✓ Classifier matches dataset ({num_classes} classes)\n")
    return model


def load_calibration_data(num_samples=256, batch_size=64, dataset="cifar100"):
    """Load calibration images from CIFAR."""
    from datasets import load_dataset
    from torchvision.transforms import InterpolationMode

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

    ds_name = dataset.lower()
    raw_ds = load_dataset(ds_name, split="train")

    def extract_label(example):
        if "label" in example:
            return int(example["label"])
        if "fine_label" in example:
            return int(example["fine_label"])
        return int(example.get("labels", 0))

    def preprocess(example):
        img = example["img"]
        img = transform(img)
        return {"pixel_values": img, "labels": extract_label(example)}

    subset = raw_ds.select(range(min(num_samples, len(raw_ds))))
    processed = subset.map(preprocess)
    processed.set_format(type="torch", columns=["pixel_values", "labels"])

    sample_imgs = [processed[i]["pixel_values"] for i in range(len(processed))]
    return torch.stack(sample_imgs, dim=0)


def load_test_dataloader(batch_size=64, dataset="cifar100", test_pct=0.2):
    """Load CIFAR test set for evaluation."""
    from datasets import load_dataset
    from torchvision.transforms import InterpolationMode

    ds_name = dataset.lower()
    test_split = f"test[:{int(test_pct * 100)}%]" if test_pct else "test"
    raw_ds = load_dataset(ds_name, split=test_split)

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

    def extract_label(example):
        if "label" in example:
            return int(example["label"])
        if "fine_label" in example:
            return int(example["fine_label"])
        return int(example.get("labels", 0))

    def preprocess(example):
        img = example["img"]
        img = transform(img)
        return {"pixel_values": img, "labels": extract_label(example)}

    processed = raw_ds.map(preprocess)
    processed.set_format(type="torch", columns=["pixel_values", "labels"])

    return DataLoader(processed, batch_size=batch_size, shuffle=False, num_workers=2)


def parse_mask_file(mask_path, model):
    """Parse mask JSON and match to model parameters."""
    with open(mask_path, "r") as f:
        mask_data = json.load(f)

    masks_tensors = {}

    if "masks" in mask_data and isinstance(mask_data["masks"], list):
        print(f"Detected new mask format:")
        print(f"  Format version: {mask_data.get('format_version', 'unknown')}")
        print(f"  Stage: {mask_data.get('stage', 'unknown')}")
        print(f"  Strategy: {mask_data.get('strategy', 'unknown')}")
        print(f"  Target sparsity: {mask_data.get('s1_sparsity', 'unknown')}")

        masks_list = mask_data["masks"]
        print(f"  Found {len(masks_list)} neuron masks in file")

        model_params = dict(model.named_parameters())
        num_layers = len(masks_list) // 2
        print(f"  Assuming {num_layers} encoder layers (2 masks per layer)\n")

        for layer_idx in range(num_layers):
            mask_idx_intermediate = layer_idx * 2
            mask_idx_output = layer_idx * 2 + 1

            if mask_idx_output >= len(masks_list):
                break

            neuron_mask = torch.tensor(masks_list[mask_idx_intermediate], dtype=torch.bool)
            intermediate_dim = neuron_mask.size(0)

            print(f"  Layer {layer_idx}:")
            print(f"    Neuron mask size: {intermediate_dim}, sparsity: {100*neuron_mask.sum().item()/intermediate_dim:.2f}%")

            intermediate_param_name = f"vit.encoder.layer.{layer_idx}.intermediate.dense.weight"
            output_param_name = f"vit.encoder.layer.{layer_idx}.output.dense.weight"

            if intermediate_param_name in model_params:
                intermediate_shape = model_params[intermediate_param_name].shape
                intermediate_weight_mask = torch.zeros(intermediate_shape, dtype=torch.bool)
                intermediate_weight_mask[neuron_mask, :] = True
                masks_tensors[intermediate_param_name] = intermediate_weight_mask
                print(f"    ✓ {intermediate_param_name}: {intermediate_shape}")

            if output_param_name in model_params:
                output_shape = model_params[output_param_name].shape
                output_weight_mask = torch.zeros(output_shape, dtype=torch.bool)
                output_weight_mask[:, neuron_mask] = True
                masks_tensors[output_param_name] = output_weight_mask
                print(f"    ✓ {output_param_name}: {output_shape}\n")

    return masks_tensors


# ============= MAIN =============

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SNOWS Reconstruction")

    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--model", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar100")
    parser.add_argument("--output", type=str, default="experiments/vit_pruning/reconstructed_model")
    parser.add_argument("--test-pct", type=float, default=0.2)
    parser.add_argument("--calib-samples", type=int, default=512)
    parser.add_argument("--newton-steps", type=int, default=5)
    parser.add_argument("--cg-iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()
    defaults = get_default_config()

    config = {
        "base_model_pt": args.adapter or defaults["base_model_pt"],
        "base_model_arch": args.model,
        "mask_json_path": args.mask or defaults["mask_json_path"],
        "output_model_dir": args.output,
        "dataset": args.dataset,
        "test_pct": args.test_pct,
        "calibration_samples": args.calib_samples,
        "newton_steps": args.newton_steps,
        "cg_iters": args.cg_iters,
        "batch_size": args.batch_size,
        "eval_batches": args.eval_batches,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    if device == "cpu":
        print("⚠️  WARNING: Running on CPU. Reconstruction will use gradient descent instead of Newton's method.")
        print("   For best results, use a CUDA GPU.\n")

    # Print config
    print("=" * 60)
    print("RECONSTRUCTION CONFIGURATION")
    print("=" * 60)
    for k, v in config.items():
        print(f"{k:20s}: {v}")
    print("=" * 60)
    print()

    num_classes = get_num_classes_from_dataset(config["dataset"])

    # Step 1: Load base model
    print("=" * 60)
    print("STEP 1: Loading base model")
    print("=" * 60)
    base_model = load_model_with_correct_head(
        config["base_model_arch"],
        config["base_model_pt"],
        num_classes,
        device
    )
    print("✓ Base model loaded\n")

    # Step 2: Evaluate base model
    print("=" * 60)
    print("STEP 2: Evaluating base model")
    print("=" * 60)
    test_loader = load_test_dataloader(
        batch_size=64,
        dataset=config["dataset"],
        test_pct=config["test_pct"]
    )

    latency_base = measure_latency(base_model, device=device)
    acc_base = evaluate_top1(
        base_model, test_loader, device=device,
        max_batches=config["eval_batches"], progress=True
    )

    print(f"Base Accuracy: {acc_base*100:.2f}%")
    print(f"Base Latency: {latency_base*1000:.2f} ms/image\n")

    # Step 3: Create pruned model
    print("=" * 60)
    print("STEP 3: Applying pruning masks")
    print("=" * 60)

    pruned_model = load_model_with_correct_head(
        config["base_model_arch"],
        config["base_model_pt"],
        num_classes,
        device
    )

    masks_tensors = parse_mask_file(config["mask_json_path"], pruned_model)
    print(f"\nTotal masks: {len(masks_tensors)}\n")

    # Apply masks
    named_params = dict(pruned_model.named_parameters())
    total_params = 0
    pruned_params = 0

    for param_name, mask_tensor in masks_tensors.items():
        if param_name in named_params:
            param = named_params[param_name]
            mask_bool = mask_tensor.to(device).bool()

            with torch.no_grad():
                param.data[mask_bool] = 0.0

            pruned = mask_bool.sum().item()
            total = param.numel()
            total_params += total
            pruned_params += pruned

            print(f"  {param_name}: {100*pruned/total:.2f}% pruned")

    overall_sparsity = 100 * pruned_params / total_params if total_params > 0 else 0
    print(f"\n✓ Overall sparsity: {overall_sparsity:.2f}%\n")

    # Step 4: Evaluate pruned model (before reconstruction)
    print("=" * 60)
    print("STEP 4: Evaluating pruned model (before reconstruction)")
    print("=" * 60)

    latency_pruned_before = measure_latency(pruned_model, device=device)
    acc_pruned_before = evaluate_top1(
        pruned_model, test_loader, device=device,
        max_batches=config["eval_batches"], progress=True
    )

    print(f"Pruned Accuracy (before): {acc_pruned_before*100:.2f}%")
    print(f"Pruned Latency (before): {latency_pruned_before*1000:.2f} ms/image")
    print(f"Accuracy drop: {(acc_base - acc_pruned_before)*100:.2f}%\n")

    # Step 5: Load calibration data
    print("=" * 60)
    print("STEP 5: Loading calibration data")
    print("=" * 60)
    print(f"Loading {config['calibration_samples']} samples...")
    xdata = load_calibration_data(
        num_samples=config["calibration_samples"],
        batch_size=config["batch_size"],
        dataset=config["dataset"]
    )
    print(f"✓ Calibration data shape: {xdata.shape}\n")

    # Step 6: Reconstruct weights layer by layer
    print("=" * 60)
    print("STEP 6: Reconstructing pruned weights")
    print("=" * 60)
    print(f"Newton steps: {config['newton_steps']}, CG iters: {config['cg_iters']}\n")

    # Sort layers to reconstruct in order
    sorted_params = sorted(masks_tensors.items(),
                          key=lambda x: (int(x[0].split('.')[3]) if 'layer' in x[0] else -1))

    for idx, (param_name, mask_tensor) in enumerate(sorted_params):
        print(f"[{idx+1}/{len(sorted_params)}] Reconstructing: {param_name}")

        if param_name not in named_params:
            print(f"  [WARN] Parameter not found, skipping\n")
            continue

        mask = mask_tensor.to(device).bool()

        try:
            updated_weight = reconstruct_layer_blockwise(
                base_model, pruned_model, param_name, mask,
                xdata, device=device,
                newton_steps=config["newton_steps"],
                cg_iters=config["cg_iters"],
                batch_size=config["batch_size"]
            )

            # Update pruned model
            with torch.no_grad():
                named_params[param_name].data.copy_(updated_weight.to(device))

            print(f"  ✓ Completed\n")

        except Exception as e:
            print(f"  [ERROR] Failed: {e}\n")
            import traceback
            traceback.print_exc()

    # Step 7: Evaluate reconstructed model
    print("=" * 60)
    print("STEP 7: Evaluating reconstructed model")
    print("=" * 60)

    latency_recon = measure_latency(pruned_model, device=device)
    acc_recon = evaluate_top1(
        pruned_model, test_loader, device=device,
        max_batches=config["eval_batches"], progress=True
    )

    print(f"Reconstructed Accuracy: {acc_recon*100:.2f}%")
    print(f"Reconstructed Latency: {latency_recon*1000:.2f} ms/image")
    print(f"Accuracy recovery: {(acc_recon - acc_pruned_before)*100:.2f}%\n")

    # Step 8: Save model
    print("=" * 60)
    print("STEP 8: Saving reconstructed model")
    print("=" * 60)

    os.makedirs(config["output_model_dir"], exist_ok=True)
    pruned_model.save_pretrained(config["output_model_dir"])

    pt_path = os.path.join(config["output_model_dir"], "reconstructed_model.pt")
    torch.save(pruned_model.state_dict(), pt_path)

    print(f"✓ Model saved to: {config['output_model_dir']}")
    print(f"  - config.json")
    print(f"  - model.safetensors")
    print(f"  - reconstructed_model.pt\n")

    # Summary
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<25} {'Accuracy':<15} {'Latency (ms)':<15}")
    print("-" * 60)
    print(f"{'Base Model':<25} {acc_base*100:>6.2f}%        {latency_base*1000:>8.2f}")
    print(f"{'Pruned (before)':<25} {acc_pruned_before*100:>6.2f}%        {latency_pruned_before*1000:>8.2f}")
    print(f"{'Reconstructed':<25} {acc_recon*100:>6.2f}%        {latency_recon*1000:>8.2f}")
    print("-" * 60)
    print(f"\nMetrics:")
    print(f"  Accuracy Recovery: {(acc_recon - acc_pruned_before)*100:+.2f}%")
    print(f"  Final vs Base: {(acc_recon - acc_base)*100:+.2f}%")
    print(f"  Speedup: {latency_base/latency_recon:.2f}x")
    print(f"  Sparsity: {overall_sparsity:.2f}%")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()