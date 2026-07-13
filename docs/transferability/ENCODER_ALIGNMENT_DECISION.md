# Encoder Alignment Decision

Current decision: `add one minimal full-MLLM measurement path after the current ensemble equal-compute gate`.

The completed single-proxy matrix shows that global encoder CKA does not explain
transfer, while NO@1 and the item-level target-distance chain contain signal.
This satisfies the predefined gate that encoder-level measurements alone are
insufficient to explain the observed multimodal transfer behavior.

The next alignment implementation should expose vision-encoder output,
connector/projector output, target-token loss and image gradient for one model
that fits the 16 GB compatibility budget. It should measure encoder versus
connector alignment separately and preserve a disabled baseline.

This decision does **not** authorize DynVLA yet. Connector perturbation remains
gated on showing that connector-level measurements add explanatory value beyond
encoder NO@1 and target-distance changes. PCGrad and generic loss weighting
remain out of scope.

The decision should be revisited after:

- CKA/NO versus transfer correlations and uncertainty;
- `D_proxy -> D_target -> success` evidence;
- model-family and CKA-sample-size sensitivity;
- disagreement between encoder predictions and frozen API replay, if available.

Possible decisions are: remain encoder-only; add one minimal full-MLLM
measurement path; or justify a connector-level pilot. DynVLA is not the default
next step.
