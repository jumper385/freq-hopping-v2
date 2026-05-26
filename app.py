import uuid
import argparse

from src.PlutoSDR import PlutoSDR
from src.bb.pipeline import build_pipelines


def det_msg(msg: str, dev_id: uuid.UUID) -> str | None:
    if str(dev_id) in msg:
        return None
    return msg.split("|")[-1].strip()


def gen_msg(msg: str, dev_id: uuid.UUID) -> str:
    return f"{dev_id} | {msg}"


def run(sdr, tx_pipe, rx_pipe, send_mode: bool) -> None:
    dev_id = uuid.uuid4()
    print(f"Device ID: {dev_id}")

    def send(msg: str) -> None:
        payload = gen_msg(msg, dev_id)
        print(payload)
        sdr.transmit(tx_pipe.encode(payload.encode("utf8")))

    if send_mode:
        while True:
            sig = input("MSG > ")
            send(sig)
    else:
        while True:
            frame = sdr.receive(flush=0)
            result = rx_pipe.decode_averaged(frame)
            if result is not None:
                rx_msg = result.decode("utf8", errors="ignore")
                rx_msg = det_msg(rx_msg, dev_id)
                if rx_msg:
                    print(rx_msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal App")
    parser.add_argument("--uri", help="IIO URI", type=str, required=True)
    parser.add_argument("--msg", help="Send mode: type messages interactively",
                        action="store_true")
    args = parser.parse_args()

    sdr = PlutoSDR(
        uri=args.uri,
        buffer_size=1024 * 32,
        tx_gain=0,
        rx_gain=30,
        center_freq=915_000_000,
    )

    tx_pipe, rx_pipe = build_pipelines(
        seq_len=1024,
        pre_len=32,
        center_guard=32,
        pilot_spacing=3,
        qam_bits=4,
        tx_scale=2**14,
        tile=24,
    )

    print("Starting...")
    run(sdr, tx_pipe, rx_pipe, send_mode=args.msg)


if __name__ == "__main__":
    main()
