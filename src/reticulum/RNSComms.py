import RNS
import time


class RNSComms:
    def __init__(self, configpath=None):

        self.configpath = configpath

        # remember to share app name to others
        self.app_name = "example_utilities"
        self.reticulum = None
        self.identity = None
        self.destination = None
        self.peers = {}

        # announce filtering setup
        self.aspect = "minimalsample"
        self.aspect_filter = f"{self.app_name}.{self.aspect}"

        # rx/tx
        self.active_links = {}

    def dest_setup(self):

        # setup configpath
        if self.configpath:
            RNS.log(f"Using Reticulum config path: {self.configpath}")
        else:
            RNS.log("Using Default Reticulum Path")

        # setup reticulum
        self.reticulum = RNS.Reticulum(self.configpath)
        RNS.Transport.register_announce_handler(self)

        # setup random identity
        self.identity = RNS.Identity()

        # setup destination
        self.destination = RNS.Destination(
                self.identity,
                RNS.Destination.IN,
                RNS.Destination.SINGLE,
                self.app_name,
                "minimalsample")

        self.destination.set_proof_strategy(RNS.Destination.PROVE_ALL)
        self.destination.set_link_established_callback(self.link_established)

    # --- RX ---
    def link_established(self, link):
        """When remote peer opens a link to us"""
        RNS.log(f"Incoming link established {link}")
        link.set_packet_callback(self.packet_received)
        link.set_remote_identified_callback(self.peer_identified)

    def peer_identified(self, link, identity):
        RNS.log(f"Peer Identified: {RNS.prettyhexrep(identity.hash)}")

    def packet_received(self, message, packet):
        text = message.decode("utf-8", errors="replace")
        RNS.log(f"Packet Received {text}")
        packet.prove()

    # --- TX ---
    def send(self, destination_hash: bytes, message: str):
        RNS.log(f"Attempting to send to {RNS.prettyhexrep(destination_hash)}: {message}")
        if destination_hash not in self.active_links:
            self._open_link(destination_hash)

        link = self.active_links.get(destination_hash)
        if link and link.status == RNS.Link.ACTIVE:
            packet = RNS.Packet(link, message.encode("utf-8"))
            packet.send()
            RNS.log(f"Sent to {RNS.prettyhexrep(destination_hash)}: {message}")
        else:
            RNS.log(f"Link not ready yet - Try again later")

    def _open_link(self, destination_hash: bytes):
        remote_identity = RNS.Identity.recall(destination_hash)

        if remote_identity is None:
            RNS.log("Identity not known yet - wait for their announce")
            return

        remote_dest = RNS.Destination(
                remote_identity,
                RNS.Destination.OUT,
                RNS.Destination.SINGLE,
                self.app_name, 
                "minimalsample"
                )

        link = RNS.Link(remote_dest)
        link.set_link_established_callback(lambda l: RNS.log(f"Outgoing link established {RNS.prettyhexrep(destination_hash)}"))
        link.set_link_closed_callback(lambda l: self.active_links.pop(destination_hash, None))
        self.active_links[destination_hash] = link

    # --- ANNOUNCEMENT ---
    def announce(self):

        if not self.destination:
            RNS.log("Destination not setup")
            raise SystemError("No RNS Destination")

        RNS.log(f"Destination at {RNS.prettyhexrep(self.destination.hash)}")

        self.destination.announce()
        RNS.log(f"Sent Announcement from {RNS.prettyhexrep(self.destination.hash)}")

    def received_announce(self, destination_hash, announced_identity, app_data):
        # required by the announcement registration
        self.peers[destination_hash] = time.time()

        RNS.log(f"Announcement Received from {RNS.prettyhexrep(destination_hash)}")

        if app_data:
            RNS.log(f"app_data: {app_data.decode("utf-8", errors="replace")}")
