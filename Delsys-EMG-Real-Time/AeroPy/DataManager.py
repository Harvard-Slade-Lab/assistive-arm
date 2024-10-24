"""
This is the class that handles the data that is output from the Delsys Trigno Base.
Create an instance of this and pass it a reference to the Trigno base for initialization.
See CollectDataController.py for a usage example.
"""
import socket
import numpy as np
import queue

class DataKernel():
    def __init__(self, trigno_base, host='10.250.1.229', port=3000):
        self.TrigBase = trigno_base
        self.packetCount = 0
        self.sampleCount = 0
        self.allcollectiondata = [[] for _ in self.TrigBase.channel_guids]
        self.channel1time = []
        self.host = host
        self.port = port
        self.socket = None

    def connect_to_server(self):
        """Connect to the Raspberry Pi server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def send_data(self, data):
        """Send data to the Raspberry Pi."""
        try:
            if self.socket:
                byte_data = data.encode('utf-8')
                self.socket.sendall(byte_data)
        except BrokenPipeError:
            print("Connection closed by the server.")
    
    def close_connection(self):
        """Close the connection to the server."""
        if self.socket:
            self.socket.close()

    def processData(self, data_queue):
        """Processes the data from the DelsysAPI and places it in the data_queue argument"""
        outArr = self.GetData()
        if outArr is not None:
            for i in range(len(outArr)):
                # Check for non-empty data and ensure the expected structure
                if outArr[i] and len(outArr[i]) > 0 and isinstance(outArr[i][0], np.ndarray):
                    try:
                        self.allcollectiondata[i].extend(outArr[i][0].tolist())
                    except IndexError:
                        # print(f"Index {i} out of range for allcollectiondata with length {len(self.allcollectiondata)}")
                        continue
                else:
                    print(f"Channel {i} has unexpected data structure or is empty.")
                    continue

            try:
                data = []
                for idx in range(len(outArr)):
                    if outArr[idx] and isinstance(outArr[idx][0], np.ndarray):
                        data.append(outArr[idx][0].tolist())
                if data:
                    data_queue.put(data)
                    self.packetCount += len(outArr[0])
                    self.sampleCount += len(outArr[0][0])
            except IndexError:
                print("IndexError encountered while putting data into queue.")
                pass

    def GetData(self):
        """Check if data is ready from DelsysAPI."""
        dataReady = self.TrigBase.TrigBase.CheckDataQueue()
        if dataReady:
            DataOut = self.TrigBase.TrigBase.PollData()
            outArr = [[] for _ in self.TrigBase.channel_guids]

            for guid in DataOut.Keys:
                index = self.TrigBase.guid_to_index.get(guid, None)
                if index is not None:
                    chan_data = DataOut[guid]
                    outArr[index].append(np.asarray(chan_data, dtype='object'))

            return outArr
        else:
            return None

