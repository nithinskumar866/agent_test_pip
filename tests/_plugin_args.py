"""How test sessions started by agenttest's own tests should load the plugin."""

from importlib.metadata import entry_points


def plugin_args() -> list[str]:
    """``-p agenttest.plugin`` when running from a source checkout; nothing when installed.

    An installed agenttest is loaded through its ``pytest11`` entry point, and
    naming it again with ``-p`` would register it twice.
    """
    installed = any(ep.value == "agenttest.plugin" for ep in entry_points(group="pytest11"))
    return [] if installed else ["-p", "agenttest.plugin"]
