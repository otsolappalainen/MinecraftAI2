# dummy_websocket_server.py
import asyncio
import websockets
import json

async def handler(websocket, path):
    while True:
        try:
            message = await websocket.recv()
            action = json.loads(message).get('action', 'no_op')
            response = {
                'broken_blocks': [],
                'alive': True,
                'x': -16.0,
                'y': -63.0,
                'health': 20.0,
                'z': -106.0,
                'pitch': 0.0,
                'inventory': {str(i): 'Item' for i in range(36)},
                'yaw': 0.0,
                'hunger': 20
            }
            await websocket.send(json.dumps(response))
        except websockets.ConnectionClosed:
            break

async def main():
    async with websockets.serve(handler, "localhost", 8080):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())