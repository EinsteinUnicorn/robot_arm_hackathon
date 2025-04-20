import asyncio
import json
import websockets

import math
import numpy as np

def compute_angles(shoulder, elbow, wrist, roll_deg):
    # Arm segment lengths
    L1 = np.linalg.norm(np.array([elbow[k] - shoulder[k] for k in 'xyz']))  # upper arm
    L2 = np.linalg.norm(np.array([wrist[k] - elbow[k] for k in 'xyz']))     # forearm

    # Vector from shoulder to wrist
    dx = wrist['x'] - shoulder['x']
    dy = wrist['y'] - shoulder['y']
    dz = wrist['z'] - shoulder['z']
    r = math.sqrt(dx**2 + dy**2)
    D = math.sqrt(dx**2 + dy**2 + dz**2)

    # Base rotation (θ1)
    theta1 = math.atan2(dy, dx)

    # Clamp cos(θ3) to avoid NaNs due to float error
    cos_theta3 = (D**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_theta3 = max(min(cos_theta3, 1), -1)
    theta3 = math.acos(cos_theta3)  # Elbow angle

    # Shoulder angle (θ2)
    phi = math.atan2(dz, r)
    psi = math.acos((L1**2 + D**2 - L2**2) / (2 * L1 * D))
    theta2 = phi + psi

    # Wrist rotation = provided roll
    theta4 = math.radians(roll_deg)

    # Convert to degrees
    return {
        'theta1': math.degrees(theta1),
        'theta2': math.degrees(theta2),
        'theta3': math.degrees(theta3),
        'theta4': roll_deg
    }

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
                
                computed_angles = compute_angles(data['shoulder'], data['elbow'], data['wrist'], data['roll'])
                print(computed_angles)

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