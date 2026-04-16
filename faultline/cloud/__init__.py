"""Cloud sync for Faultlines SaaS dashboard.

Optional, opt-in module. When ``FAULTLINE_API_KEY`` is set:
  - ``faultlines analyze --push`` uploads the feature map to the SaaS
  - The MCP server batches tool-call events and sends them every minute

Without an API key, all cloud calls become silent no-ops. The CLI and
MCP server work entirely locally and never reach out to faultlines.dev.
"""

from faultline.cloud.sync import push_feature_map, send_mcp_events_batch

__all__ = ["push_feature_map", "send_mcp_events_batch"]
