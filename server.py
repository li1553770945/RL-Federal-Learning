from plistlib import Dict

import flwr as fl
from flwr.server
server_address = "[::]:9000"
fl.server.start_server(server_address=server_address,config=fl.server.ServerConfig(num_rounds=3))

