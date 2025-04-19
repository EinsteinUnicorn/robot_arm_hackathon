import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://127.0.0.1:8765"
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket server")

            # Send dummy motion data
            test_data = {
                "shoulder": {"x": 0.1, "y": 0.2, "z": 0.3},
                "elbow": {"x": 0.4, "y": 0.5, "z": 0.6},
                "wrist": {"x": 0.7, "y": 0.8, "z": 0.9},
                "roll": 42.0
            }

            await websocket.send(json.dumps(test_data))
            print("✅ Sent test data")

            # Await response
            response = await websocket.recv()
            print(f"✅ Got response: {response}")

    except Exception as e:
        print("❌ WebSocket test failed:", e)

# Run the test
asyncio.run(test_ws())
