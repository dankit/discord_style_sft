from __future__ import annotations

from discord_sft.cli.env import _load_local_dotenv
from discord_sft.cli.parser import build_parser


def main() -> None:
    _load_local_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


__all__ = ["main", "build_parser"]
