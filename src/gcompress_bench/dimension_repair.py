"""
Dimension Repair Module for G-Compress

Implements shape alignment constraints and repair strategies based on C2/C3 findings:
- H1: Tensor Core requires K % 16 == 0 for optimal performance (58% slowdown otherwise)
- H4: Vectorized loading requires K % 8 == 0 for float4 (50% throughput loss otherwise)
- H3: SDPA bandwidth efficiency requires head_dim % 8 == 0 (40% efficiency loss otherwise)

Reference: results/C23/20260124_220005_C23_hardware_layer/
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class AlignmentStrategy(Enum):
    """Dimension repair strategies with different memory-performance tradeoffs."""

    MINIMAL = "minimal"          # Pad to next 8-aligned (minimum overhead)
    OPTIMAL = "optimal"          # Pad to next 16-aligned or predefined fast path
    PREDEFINED = "predefined"    # Pad to nearest {64, 96, 112, 128, ...}
    TRADEOFF = "tradeoff"        # Choose based on overhead threshold


@dataclass
class ShapeContract:
    """
    Formalized alignment constraints for GPU-efficient tensor dimensions.

    Based on C23 Hardware Layer Analysis findings:
    - minimal_alignment: 8 (for vectorized float4 loads)
    - optimal_alignment: 16 (for Tensor Core tiles)
    - recommended_values: {64, 96, 112, 128} (known fast paths)

    Attributes:
        minimal_alignment: Minimum alignment for correctness (8 for SDPA)
        optimal_alignment: Alignment for best Tensor Core utilization (16)
        recommended_values: Known dimensions with optimized kernel paths
        max_overhead_pct: Maximum acceptable memory overhead (default 20%)
    """
    minimal_alignment: int = 8
    optimal_alignment: int = 16
    recommended_values: Tuple[int, ...] = (32, 64, 96, 112, 128, 160, 192, 224, 256)
    max_overhead_pct: float = 20.0

    def is_aligned(self, dim: int, level: str = "minimal") -> bool:
        """Check if dimension meets alignment requirements."""
        if level == "minimal":
            return dim % self.minimal_alignment == 0
        elif level == "optimal":
            return dim % self.optimal_alignment == 0
        elif level == "predefined":
            return dim in self.recommended_values
        return False

    def alignment_gap(self, dim: int) -> Dict[str, int]:
        """Calculate gap to each alignment level."""
        return {
            "to_8": (self.minimal_alignment - dim % self.minimal_alignment) % self.minimal_alignment,
            "to_16": (self.optimal_alignment - dim % self.optimal_alignment) % self.optimal_alignment,
            "to_predefined": min(
                (v - dim for v in self.recommended_values if v >= dim),
                default=0
            ),
        }

    def memory_overhead(self, original: int, padded: int) -> float:
        """Calculate memory overhead percentage."""
        if original <= 0:
            return 0.0
        return 100.0 * (padded - original) / original


def repair_dimension(
    head_dim: int,
    strategy: Union[str, AlignmentStrategy] = AlignmentStrategy.MINIMAL,
    max_overhead_pct: float = 20.0,
    contract: Optional[ShapeContract] = None,
) -> int:
    """
    Repair head_dim to the nearest aligned value based on strategy.

    Args:
        head_dim: Original dimension to repair
        strategy: Repair strategy (minimal, optimal, predefined, tradeoff)
        max_overhead_pct: Maximum acceptable memory overhead for tradeoff strategy
        contract: Custom ShapeContract (uses default if None)

    Returns:
        Repaired dimension that satisfies alignment constraints

    Examples:
        >>> repair_dimension(107, strategy="minimal")
        112  # Pad to next 8-aligned
        >>> repair_dimension(107, strategy="optimal")
        112  # Pad to next 16-aligned
        >>> repair_dimension(107, strategy="predefined")
        112  # Nearest predefined value
        >>> repair_dimension(125, strategy="tradeoff", max_overhead_pct=5.0)
        128  # 2.4% overhead acceptable

    Performance Impact (from C23):
        - Non-aligned → 8-aligned: +50% throughput (vectorized loads)
        - 8-aligned → 16-aligned: +15% throughput (Tensor Core tiles)
    """
    if contract is None:
        contract = ShapeContract()

    if isinstance(strategy, str):
        strategy = AlignmentStrategy(strategy)

    # Already aligned to optimal - no repair needed
    if contract.is_aligned(head_dim, "optimal") or head_dim in contract.recommended_values:
        return head_dim

    if strategy == AlignmentStrategy.MINIMAL:
        # Pad to next 8-aligned
        return ((head_dim + 7) // 8) * 8

    elif strategy == AlignmentStrategy.OPTIMAL:
        # Pad to next 16-aligned
        return ((head_dim + 15) // 16) * 16

    elif strategy == AlignmentStrategy.PREDEFINED:
        # Find nearest predefined value >= head_dim
        for v in contract.recommended_values:
            if v >= head_dim:
                return v
        # Fallback: pad to 16-aligned
        return ((head_dim + 15) // 16) * 16

    elif strategy == AlignmentStrategy.TRADEOFF:
        # Try progressively stricter alignments until overhead exceeds threshold
        candidates = [
            ((head_dim + 7) // 8) * 8,    # 8-aligned
            ((head_dim + 15) // 16) * 16,  # 16-aligned
        ]
        # Add predefined values
        for v in contract.recommended_values:
            if v >= head_dim:
                candidates.append(v)
                break

        # Sort by padded size
        candidates = sorted(set(candidates))

        # Return first candidate within overhead threshold
        for padded in candidates:
            overhead = contract.memory_overhead(head_dim, padded)
            if overhead <= max_overhead_pct:
                return padded

        # If all exceed threshold, return minimal (8-aligned)
        return ((head_dim + 7) // 8) * 8

    return head_dim


@dataclass
class RepairResult:
    """Result of dimension repair for a model."""
    original_dims: Dict[str, int]
    repaired_dims: Dict[str, int]
    strategy: str
    total_original_params: int = 0
    total_repaired_params: int = 0
    memory_overhead_pct: float = 0.0

    @property
    def affected_layers(self) -> List[str]:
        """List of layers that were modified."""
        return [k for k, v in self.original_dims.items()
                if v != self.repaired_dims.get(k, v)]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Dimension Repair Summary (strategy={self.strategy})",
            f"  Affected layers: {len(self.affected_layers)}/{len(self.original_dims)}",
            f"  Memory overhead: {self.memory_overhead_pct:.2f}%",
            "",
            "  Dimension changes:",
        ]
        for name in sorted(self.affected_layers)[:10]:  # Show first 10
            orig = self.original_dims[name]
            new = self.repaired_dims[name]
            lines.append(f"    {name}: {orig} → {new}")
        if len(self.affected_layers) > 10:
            lines.append(f"    ... and {len(self.affected_layers) - 10} more")
        return "\n".join(lines)


class DimensionRepairer:
    """
    Repairs tensor dimensions in a model to satisfy hardware alignment constraints.

    Designed for post-compression models like PaLU where SVD creates irregular dimensions.
    Applies zero-padding to weight matrices and adjusts subsequent layers accordingly.

    Example usage:
        >>> repairer = DimensionRepairer(strategy="minimal")
        >>> model, result = repairer.repair_model(palu_model)
        >>> print(result.summary())
    """

    def __init__(
        self,
        strategy: Union[str, AlignmentStrategy] = AlignmentStrategy.MINIMAL,
        max_overhead_pct: float = 20.0,
        contract: Optional[ShapeContract] = None,
    ):
        self.strategy = AlignmentStrategy(strategy) if isinstance(strategy, str) else strategy
        self.max_overhead_pct = max_overhead_pct
        self.contract = contract or ShapeContract()

    def analyze_model(self, model: nn.Module) -> Dict[str, int]:
        """
        Analyze model to find dimensions that need repair.

        Focuses on attention projection layers where PaLU compression
        creates irregular head_dim values.

        Returns:
            Dict mapping layer names to their current head_dim values
        """
        dims = {}
        for name, module in model.named_modules():
            # Look for linear layers in attention projections
            if isinstance(module, nn.Linear):
                # Check for common attention projection patterns
                if any(proj in name.lower() for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                    # For compressed models, the inner dimension may be irregular
                    in_features = module.in_features
                    out_features = module.out_features

                    # K/V projections: out_features is the compressed dimension
                    if 'k_proj' in name.lower() or 'v_proj' in name.lower():
                        dims[name] = out_features
                    # For O projection, in_features should match K/V compressed dim
                    elif 'o_proj' in name.lower():
                        dims[name] = in_features

        return dims

    def compute_repair_plan(
        self,
        model: nn.Module,
    ) -> Dict[str, Tuple[int, int]]:
        """
        Compute repair plan without modifying the model.

        Returns:
            Dict mapping layer names to (original_dim, repaired_dim) tuples
        """
        dims = self.analyze_model(model)
        plan = {}

        for name, dim in dims.items():
            repaired = repair_dimension(
                dim,
                strategy=self.strategy,
                max_overhead_pct=self.max_overhead_pct,
                contract=self.contract,
            )
            plan[name] = (dim, repaired)

        return plan

    def repair_linear_layer(
        self,
        layer: nn.Linear,
        dim_axis: str,  # "in" or "out"
        target_dim: int,
    ) -> nn.Linear:
        """
        Repair a linear layer by padding weights to target dimension.

        Args:
            layer: Original linear layer
            dim_axis: Which dimension to pad ("in" or "out")
            target_dim: Target dimension after padding

        Returns:
            New linear layer with padded weights
        """
        weight = layer.weight.data  # [out_features, in_features]
        bias = layer.bias.data if layer.bias is not None else None

        out_features, in_features = weight.shape

        if dim_axis == "out":
            # Pad output dimension (rows)
            if target_dim > out_features:
                pad_size = target_dim - out_features
                weight_pad = torch.zeros(pad_size, in_features,
                                        dtype=weight.dtype, device=weight.device)
                weight = torch.cat([weight, weight_pad], dim=0)
                if bias is not None:
                    bias_pad = torch.zeros(pad_size, dtype=bias.dtype, device=bias.device)
                    bias = torch.cat([bias, bias_pad], dim=0)
                out_features = target_dim

        elif dim_axis == "in":
            # Pad input dimension (columns)
            if target_dim > in_features:
                pad_size = target_dim - in_features
                weight_pad = torch.zeros(out_features, pad_size,
                                        dtype=weight.dtype, device=weight.device)
                weight = torch.cat([weight, weight_pad], dim=1)
                in_features = target_dim

        # Create new layer
        new_layer = nn.Linear(in_features, out_features, bias=bias is not None)
        new_layer.weight.data = weight
        if bias is not None:
            new_layer.bias.data = bias

        return new_layer

    def repair_model(
        self,
        model: nn.Module,
        inplace: bool = False,
    ) -> Tuple[nn.Module, RepairResult]:
        """
        Repair all attention dimensions in a model.

        Note: This is a simplified implementation. Full integration with
        PaLU models requires handling the specific SVD decomposition structure.

        Args:
            model: Model to repair
            inplace: If True, modify model in place; otherwise return a copy

        Returns:
            Tuple of (repaired_model, RepairResult)
        """
        if not inplace:
            import copy
            model = copy.deepcopy(model)

        plan = self.compute_repair_plan(model)

        original_dims = {}
        repaired_dims = {}

        for name, (orig, target) in plan.items():
            original_dims[name] = orig
            repaired_dims[name] = target

            if orig == target:
                continue

            # Navigate to the layer
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            layer_name = parts[-1]
            layer = getattr(parent, layer_name)

            if not isinstance(layer, nn.Linear):
                continue

            # Determine which dimension to pad
            if 'k_proj' in name.lower() or 'v_proj' in name.lower():
                dim_axis = "out"  # Pad output dimension
            elif 'o_proj' in name.lower():
                dim_axis = "in"   # Pad input dimension
            else:
                continue

            # Apply repair
            new_layer = self.repair_linear_layer(layer, dim_axis, target)
            setattr(parent, layer_name, new_layer)

        # Calculate memory overhead
        total_orig = sum(original_dims.values())
        total_repaired = sum(repaired_dims.values())
        overhead = 100.0 * (total_repaired - total_orig) / total_orig if total_orig > 0 else 0.0

        result = RepairResult(
            original_dims=original_dims,
            repaired_dims=repaired_dims,
            strategy=self.strategy.value,
            memory_overhead_pct=overhead,
        )

        return model, result


def create_repair_hooks(
    model: nn.Module,
    strategy: Union[str, AlignmentStrategy] = AlignmentStrategy.MINIMAL,
) -> Dict[str, callable]:
    """
    Create forward hooks that apply runtime padding to attention inputs.

    This is an alternative to weight modification - instead of padding weights,
    we pad the input tensors at runtime. This is useful for:
    - Quick experimentation without modifying model weights
    - Validating padding effectiveness before permanent repair

    Returns:
        Dict of hook handles that can be removed later
    """
    repairer = DimensionRepairer(strategy=strategy)
    plan = repairer.compute_repair_plan(model)
    hooks = {}

    def make_padding_hook(orig_dim: int, target_dim: int, dim_axis: str):
        def hook(module, inputs, outputs):
            if dim_axis == "out" and outputs.shape[-1] == orig_dim:
                # Pad last dimension of output
                pad_size = target_dim - orig_dim
                padding = torch.zeros(
                    *outputs.shape[:-1], pad_size,
                    dtype=outputs.dtype, device=outputs.device
                )
                return torch.cat([outputs, padding], dim=-1)
            return outputs
        return hook

    for name, (orig, target) in plan.items():
        if orig == target:
            continue

        # Navigate to layer
        parts = name.split('.')
        layer = model
        for part in parts:
            layer = getattr(layer, part)

        if 'k_proj' in name.lower() or 'v_proj' in name.lower():
            hook_fn = make_padding_hook(orig, target, "out")
            handle = layer.register_forward_hook(hook_fn)
            hooks[name] = handle

    return hooks
