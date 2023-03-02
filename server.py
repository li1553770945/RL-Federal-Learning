import flwr as fl
from client_manager import RLManager
server_address = "[::]:9000"
fl.server.start_server(server_address=server_address,config=fl.server.ServerConfig(num_rounds=3),client_manager=RLManager())

