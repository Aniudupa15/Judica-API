import socket
hostname = socket.gethostname()  # Get the local machine name
print(socket.gethostbyname(hostname))  # Get the local IP address