import asyncio
import json
import websockets

# Function to handle WebSocket connections
async def handle_connection(websocket):
    print("Client connected!")
    
    try:
        # Keep the connection open and process messages
        async for message in websocket:
            try:
                # Parse the JSON data
                data = json.loads(message)
                
                # Process the received data
                print("\n--- Received Motion Data ---")
                print(f"Shoulder: x={data['shoulder']['x']:.3f}, y={data['shoulder']['y']:.3f}, z={data['shoulder']['z']:.3f}")
                print(f"Elbow: x={data['elbow']['x']:.3f}, y={data['elbow']['y']:.3f}, z={data['elbow']['z']:.3f}")
                print(f"Wrist: x={data['wrist']['x']:.3f}, y={data['wrist']['y']:.3f}, z={data['wrist']['z']:.3f}")
                print(f"Wrist Roll: {data['roll']:.2f} degrees")
                
                # Here you can add your custom processing logic
                # For example, trigger actions based on specific movements
                
                # Optionally send a response back
                await websocket.send(json.dumps({"status": "received"}))
                
            except json.JSONDecodeError:
                print(f"Received non-JSON message: {message}")
            except KeyError as e:
                print(f"Missing expected data field: {e}")
    
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")

# Start the WebSocket server
async def main():
    # Start server on the same port your sketch.js connects to (8765)
    server = await websockets.serve(handle_connection, "localhost", 8765)
    print("WebSocket server started at ws://localhost:8765")
    
    # Keep the server running
    await asyncio.Future()  # Run forever

# Run the server
if __name__ == "__main__":
    asyncio.run(main())