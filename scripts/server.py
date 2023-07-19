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
    force_plates = packet.get_force()
    dict_force = {f"plate_{plate.id}": forces
                  for plate, forces in force_plates}
    # E.g.: {'plate_1': [RTForce.x, RTForce.y, ...], 
    #        'plate_2': [...]}

    dict_marker = {
        f"marker_{i}": {"x": marker.x, "y": marker.y, "z": marker.z}
        for i, marker in enumerate(markers)
    }
    combined_dict = {**dict_marker, **dict_force}
    marker_json = json.dumps(combined_dict)

    print(f"Component info: {header}")
    print(dict_marker)

    # Send packet to client
    message = socket.recv_string()
    print(f'Received request "{message}"')
    socket.send_string(marker_json)


async def setup():
    """Setup connection to QTM"""
    connection = await qtm_rt.connect("127.0.0.1")
    if connection is None:
        return

    await connection.stream_frames(components=["3d"], on_packet=on_packet)


if __name__ == "__main__":
    asyncio.ensure_future(setup())
    asyncio.get_event_loop().run_forever()
