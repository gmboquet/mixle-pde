# Security

Treat model, mesh, matrix, observation, checkpoint, and result artifacts as untrusted. Validate schema, finite size,
indices, exact-rational syntax, digests, and optional dependency availability before allocation or execution. Adapters
do not evaluate code, import paths from records, load native plugins, or access networks.
