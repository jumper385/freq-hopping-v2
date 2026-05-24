import argparse
from src.reticulum.RNSComms import RNSComms


parser = argparse.ArgumentParser(
        description="Minimal Impl"
)

parser.add_argument("--config",
                    action="store",
                    default=None,
                    help="path to alternative Reticulum config directory",
                    type=str)

args = parser.parse_args()

if args.config:
    configarg = args.config
else:
    configarg = None

rns_comms = RNSComms(configarg)
rns_comms.dest_setup()
rns_comms.announce()

while True:
    message = input("ENTER MESSAGE > ")
    rns_comms.announce()
    for dest_hash in list(rns_comms.peers.keys()):
        rns_comms.send(dest_hash, message)
