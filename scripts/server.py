import json
import zmq
import asyncio
import qtm_rt

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")


def on_packet(packet):
    """Callback function that is called everytime a data packet arrives from QTM"""
    header, markers = packet.get_3d_markers()

    dict_marker = {
        i: {"x": marker.x, "y": marker.y, "z": marker.z}
        for i, marker in enumerate(markers)
    }
    marker_json = json.dumps(dict_marker)

    # Send packet to client
    message = socket.recv_string()
    print(f'Received request "{message}"')
    socket.send_string(marker_json)

    print(f"Component info: {header}")
    print(dict_marker)


async def setup():
    """Setup connection to QTM"""
    connection = await qtm_rt.connect("127.0.0.1")
    if connection is None:
        return

    await connection.stream_frames(components=["3d"], on_packet=on_packet)


if __name__ == "__main__":
    asyncio.ensure_future(setup())
    asyncio.get_event_loop().run_forever()
