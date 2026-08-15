# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Oliver Jandette

"""Top-level entry point. Delegates to the CLI defined in kindle_math_converter.main."""
from kindle_math_converter.main import cli

if __name__ == "__main__":
    cli()
