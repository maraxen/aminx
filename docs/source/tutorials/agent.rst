Using aminx from an agent
=========================

This page covers the ``agent`` extra, publishing the aminx plugin, and the MCP launch mode.

Install
-------

MCP tools need the agent extra. The library and the ``aminx`` CLI do not.

.. code-block:: bash

   pip install "aminx[agent]"

The extra installs ``cisternal`` (``cisternal>=0.1.1a15,<0.2``). The ``aminx-mcp`` console script is the MCP server entry point. ``aminx.agent.require_agent_extra`` refuses to start that server unless both ``cisternal`` and ``fastmcp`` import. The error tells you to install ``aminx[agent]``.

Publish the plugin
------------------

Publish into the shared cisternal marketplace:

.. code-block:: bash

   python -m aminx.agent.plugin publish

``--scope`` is the Claude install scope: ``user`` (the default), ``project``, or ``local``.

``--marketplace PATH`` sets the marketplace root. Omit it and publish uses the shared marketplace root.

``--dry-run`` resolves the launch and prints the JSON record. It does not write the bundle or install the plugin. The record includes ``"dry_run": true``.

``--allow-unpublished`` lets a ``uvx`` publish continue when the release is missing from PyPI or the index cannot be reached. Without it, publish stops in those cases.

Default uvx launch
-------------------

The default mode is ``uvx``. Publish writes this command, with ``<version>`` the installed aminx version and any ``+local`` suffix removed:

.. code-block:: text

   uvx --from aminx[agent]==<version> aminx-mcp

Publish checks ``https://pypi.org/pypi/aminx/<version>/json``. HTTP 200 continues. HTTP 404 stops with an error that the version is not on the index and that ``--launch venv`` is the alternative. Any other failure to reach the index also stops, because ``uvx`` needs the index too. ``--allow-unpublished`` skips both stops.

Cold start: the first run resolves the uvx environment, and the first tool call per checkpoint and padded shape compiles JAX. See "Run time and padding" below.

Local venv
----------

Resolution order, first hit wins. ``uvx`` and ``venv`` are the only modes. A relative ``mcp_python`` is joined to a base directory and resolved to an absolute path. An absolute path is kept. When the mode is ``venv`` and no interpreter was set at any layer, the interpreter is ``sys.executable`` (the Python that is running publish).

The venv command is that absolute interpreter path followed by ``-m aminx.agent.mcp``. Claude Code does not activate virtualenvs, so a bare ``python`` on ``PATH`` is not used. Publish runs ``import aminx.agent.mcp`` with that interpreter and fails if the import does not exit 0. The error names the interpreter and includes the tail of stderr.

1. Command line. ``--python`` is resolved against the current working directory.

   .. code-block:: bash

      python -m aminx.agent.plugin publish --launch venv --python /path/to/venv/bin/python

2. Environment. ``AMINX_MCP_PYTHON`` is resolved against the current working directory.

   .. code-block:: bash

      AMINX_MCP_LAUNCH=venv AMINX_MCP_PYTHON=/path/to/venv/bin/python python -m aminx.agent.plugin publish

3. Nearest ``pyproject.toml``, walked upward from the working directory. A relative ``mcp_python`` is resolved against the directory that contains that file.

   .. code-block:: toml

      [tool.aminx.agent]
      mcp_launch = "venv"
      mcp_python = ".venv/bin/python"

4. User config, ``${XDG_CONFIG_HOME:-~/.config}/aminx/config.toml``. A relative ``mcp_python`` is resolved against the directory that contains ``config.toml``.

   .. code-block:: toml

      [agent]
      mcp_launch = "venv"
      mcp_python = "/path/to/venv/bin/python"

A malformed TOML file, a non-string value, or an unknown mode raises ``ValueError`` at publish time.

Check the active mode
---------------------

``info`` prints JSON and does not write, install, or check PyPI:

.. code-block:: bash

   python -m aminx.agent.plugin info

The object has ``launch_mode``, ``launch_source``, ``command``, ``python``, ``marketplace``, ``marketplace_source``, and ``bundle_source``. Illustrative output (the version, marketplace path, and the two source strings depend on the machine):

.. code-block:: json

   {
     "launch_mode": "uvx",
     "launch_source": "default",
     "command": ["uvx", "--from", "aminx[agent]==0.1.0", "aminx-mcp"],
     "python": null,
     "marketplace": "/home/user/.cisternal/claude-plugin-marketplace",
     "marketplace_source": "<marketplace source>",
     "bundle_source": "<bundle source>"
   }

``python`` is ``null`` for ``uvx``. For ``venv`` it is the absolute interpreter path. ``launch_source`` names the layer that chose the mode (``argument``, ``$AMINX_MCP_LAUNCH``, the ``pyproject.toml`` origin, the user config path, or ``default``).

Switch back to uvx
------------------

Run publish again with ``--launch uvx``. If ``AMINX_MCP_LAUNCH``, ``pyproject.toml``, or the user config still selects ``venv``, remove that setting. Otherwise the next publish that does not pass ``--launch`` will select ``venv`` again.

.. code-block:: bash

   python -m aminx.agent.plugin publish --launch uvx

Side files
----------

When a tool call omits ``output_dir``, side files go to ``$AMINX_AGENT_OUTPUT_DIR`` if that variable is set, otherwise ``$XDG_CACHE_HOME/aminx/agent``, otherwise ``~/.cache/aminx/agent``. Model-run results include ``spec``, ``spec_sha256``, and ``provenance``.

Run time and padding
--------------------

Each structure is padded to ``max_length`` (default 512) by the loader, and decode time grows with the length the model runs at. ``sample`` and ``score`` calls are trimmed by the runner: each batch runs at the next bucket on xtrax's ``BUCKET_LADDER`` (64, 128, 256, 512, ...) that holds its structures, so ``max_length`` is only a cap and the returned ``spec`` keeps the caller's value. The runner does not trim ``inspect``, ``jacobian``, ``pass_mode="inter"``, averaged-feature or multi-state scoring, or ``length_bucketing=False``. For those, when ``options`` does not set ``max_length`` and every input parses, the server fits it to the inputs: the longest structure (or the summed length for ``inter``), rounded up to the next ladder bucket and never above the spec's ``max_length``. The fitted value is in the returned ``spec``, so a run can be repeated exactly.

One server process keeps JAX's compiled functions, so a later call with the same checkpoint and padded shape does not compile again.
