## Quick Start

### Server Mode

1. Start the container:
    ```
    docker-compose up -d
    ```

This starts the gRPC server on 9012.

### Dev Mode

On the host:

1. Start the container:
    ```
    docker-compose -f docker-compose.dev.yml up -d
    ```
2. Enter the container:
    ```
    docker ps
    docker exec -it <container ID> bash
    ```
    
Inside the container:
    
1. Start the gRPC server on 9012:
    ```
    python3 /files/server.py --port 9012
    ```

## Useful Links

* Deepface project: https://github.com/serengil/deepface