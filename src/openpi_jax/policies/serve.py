"""Websocket policy server for closed-loop robot inference.

A robot client connects, streams observations, and receives action chunks. Wire format is msgpack-encoded
dicts. Run via ``openpi-serve --config pi0_aloha_sim --checkpoint <path>``.

The serving loop and (de)serialization are sketched here; restoring real params and the model trunk are TODOs.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class ServeArgs:
    config: str = "pi0_aloha_sim"
    checkpoint: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000


def build_policy(args: ServeArgs):
    """Construct a ``Policy`` from a config + checkpoint.

    TODO: restore params via ``training.checkpoints.restore``, build the input/output transform stacks
    (tokenizer + normalizer loaded from assets), and return a ready ``Policy``.
    """
    raise NotImplementedError("serving build_policy — see docs/ROADMAP.md")


async def _serve(args: ServeArgs) -> None:
    import msgpack
    import websockets

    policy = build_policy(args)

    async def handler(websocket):
        async for message in websocket:
            obs = msgpack.unpackb(message, raw=False)
            result = policy.infer(obs)
            await websocket.send(msgpack.packb(result, use_bin_type=True))

    print(f"[openpi-jax] serving '{args.config}' on ws://{args.host}:{args.port}")
    async with websockets.serve(handler, args.host, args.port, max_size=None):
        import asyncio

        await asyncio.Future()  # run forever


def main() -> None:
    import asyncio

    import tyro

    args = tyro.cli(ServeArgs)
    asyncio.run(_serve(args))


if __name__ == "__main__":
    main()
