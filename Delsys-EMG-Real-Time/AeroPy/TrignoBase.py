import threading
import time
from pythonnet import load
load("coreclr")
import clr

# Import required libraries for data handling and plotting
from AeroPy.DataManager import DataKernel
import matplotlib.pyplot as plt
import numpy as np

# Ensure correct path handling by using raw strings
clr.AddReference(r"resources\DelsysAPI")
clr.AddReference("System.Collections")

from Aero import AeroPy

key = "MIIBKjCB4wYHKoZIzj0CATCB1wIBATAsBgcqhkjOPQEBAiEA/////wAAAAEAAAAAAAAAAAAAAAD///////////////8wWwQg/////wAAAAEAAAAAAAAAAAAAAAD///////////////wEIFrGNdiqOpPns+u9VXaYhrxlHQawzFOw9jvOPD4n0mBLAxUAxJ02CIbnBJNqZnjhE50mt4GffpAEIQNrF9Hy4SxCR/i85uVjpEDydwN9gS3rM6D0oTlF2JjClgIhAP////8AAAAA//////////+85vqtpxeehPO5ysL8YyVRAgEBA0IABJ9i3whTdIzDlkXEBJqMK0ked1eSLc+CFj72nP2tWC/x4SJx5dzsXBEb+/xhEeb6DK1FXdYpV9jDRpQ4Civ/k1M="
license = "<License>"\
    "<Id>3e889b4f-a926-43c9-b633-3a3e2698c017</Id>"\
    "<Type>Standard</Type>"\
    "<Quantity>10</Quantity>"\
    "<LicenseAttributes>"\
        "<Attribute name='Software'></Attribute>"\
    "</LicenseAttributes>"\
    "<ProductFeatures>"\
        "<Feature name='Sales'>True</Feature>"\
        "<Feature name='Billing'>False</Feature>"\
    "</ProductFeatures>"\
    "<Customer>"\
        "<Name>Nathan Irniger</Name>"\
        "<Email>nathani@g.harvard.edu</Email>"\
    "</Customer>"\
    "<Expiration>Sun, 31 Dec 2034 05:00:00 GMT</Expiration>"\
    "<Signature>MEYCIQCjMbLHwVD1gqzSQMiEGa19b98bTVFKhgU3oVwDvawvggIhALwvVE6XxSItA/VgyfPhZ3n/g0yIfz2j5F+pjfH2KQ/Q</Signature>"\
"</License>"

class TrignoBase:
    def __init__(self, collection_data_handler):
        self.TrigBase = AeroPy()
        self.collection_data_handler = collection_data_handler
        self.channelcount = 0
        self.pairnumber = 0
        self.channel_guids = []
        self.channelLabels = []
        self.emgChannelLabels = []
        self.accChannelLabels = []
        self.gyroChannelLabels = []
        self.orChannelsLabels = []
        self.emgChannelsIdx = []
        self.accChannelsIdx = []
        self.gyroChannelsIdx = []
        self.orChannelsIdx = []
        self.emgSampleRates = []
        self.accSampleRates = []
        self.gyroSampleRates = []
        self.orSampleRates = []
        self.sensor_label_to_channels = {}
        self.guid_to_index = {}

    # -- AeroPy Methods --
    def PipelineState_Callback(self):
        return self.TrigBase.GetPipelineState()

    def Connect_Callback(self):
        """Callback to connect to the base"""
        self.TrigBase.ValidateBase(key, license)

    def Pair_Callback(self):
        return self.TrigBase.PairSensor(self.pair_number)

    def CheckPairStatus(self):
        return self.TrigBase.CheckPairStatus()

    def CheckPairComponentAdded(self):
        return self.TrigBase.CheckPairComponentAdded()

    def Scan_Callback(self):
        """Callback to tell the base to scan for any available sensors"""
        try:
            f = self.TrigBase.ScanSensors().Result
        except Exception as e:
            print("Python demo attempt another scan...")
            time.sleep(2)
            self.Scan_Callback()

        self.all_scanned_sensors = self.TrigBase.GetScannedSensorsFound()
        print("Sensors Found:\n")
        for sensor in self.all_scanned_sensors:
            print(f"({sensor.PairNumber}) {sensor.FriendlyName}\n{sensor.Configuration.ModeString}\n")

        self.SensorCount = len(self.all_scanned_sensors)
        for i in range(self.SensorCount):
            self.TrigBase.SelectSensor(i)

    def Start_Callback(self, start_trigger, stop_trigger):
        """Callback to start the data stream from Sensors"""
        self.start_trigger = start_trigger
        self.stop_trigger = stop_trigger

        configured = self.ConfigureCollectionOutput()
        if configured:
            self.TrigBase.Start(self.collection_data_handler.streamYTData)
            self.collection_data_handler.threadManager(self.start_trigger, self.stop_trigger)


    def ConfigureCollectionOutput(self):
        if not self.start_trigger:
            self.collection_data_handler.pauseFlag = False

        self.collection_data_handler.data_handler.packetCount = 0
        self.collection_data_handler.data_handler.allcollectiondata = [[]]

        if self.TrigBase.GetPipelineState() == 'Armed':
            for i in range(len(self.channelobjects)):
                self.collection_data_handler.data_handler.allcollectiondata.append([])
            return True
        elif self.TrigBase.GetPipelineState() == 'Connected':
            self.channelcount = 0
            self.TrigBase.Configure(self.start_trigger, self.stop_trigger)
            configured = self.TrigBase.IsPipelineConfigured()
            if configured:
                self.channelobjects = []
                self.plotCount = 0
                self.emgChannelsIdx = []
                self.accChannelsIdx = []
                self.gyroChannelsIdx = []
                self.orChannelsIdx = []
                self.channelLabels = []
                self.emgChannelLabels = []
                self.accChannelLabels = []
                self.gyroChannelLabels = []
                self.orChannelsLabels = []
                self.emgSampleRates = {}      # Changed to dictionary
                self.accSampleRates = {}      # Changed to dictionary
                self.gyroSampleRates = {}     # Changed to dictionary
                self.orSampleRates = {}       # Changed to dictionary
                self.emgChannelSensors = []   # List to map EMG channels to sensors
                self.accChannelSensors = []   # List to map ACC channels to sensors
                self.gyroChannelSensors = []  # List to map GYRO channels to sensors
                self.orChannelSensors = []    # List to map OR channels to sensors
                self.channel_guids = []
                self.sensor_label_to_channels = {}
                self.guid_to_index = {}
                globalChannelIdx = 0

                for i in range(self.SensorCount):
                    selectedSensor = self.TrigBase.GetSensorObject(i)
                    sensor_label = selectedSensor.PairNumber
                    self.sensor_label_to_channels[sensor_label] = {'EMG': [], 'ACC': [], 'GYRO': [], 'OR': []}

                    print(f"({sensor_label}) {selectedSensor.FriendlyName}")

                    if len(selectedSensor.TrignoChannels) > 0:
                        print("--Channels")
                        for channel in range(len(selectedSensor.TrignoChannels)):
                            channel_obj = selectedSensor.TrignoChannels[channel]
                            sample_rate = round(channel_obj.SampleRate, 3)
                            channel_name = channel_obj.Name
                            channel_label = f"{channel_name}"
                            print(f"----{channel_label} ({sample_rate} Hz)")
                            self.channelcount += 1
                            self.channelobjects.append(channel)
                            self.collection_data_handler.data_handler.allcollectiondata.append([])

                            # Append channel label
                            self.channelLabels.append(channel_label)

                            if hasattr(channel_obj, 'Id'):
                                guid = channel_obj.Id  # GUID object
                                self.channel_guids.append(guid)  # Store the GUID object directly
                                self.guid_to_index[guid] = globalChannelIdx  # Map GUID to index
                            else:
                                self.channel_guids.append(channel_label)

                            if "EMG" in channel_name:
                                self.emgChannelsIdx.append(globalChannelIdx)
                                self.emgChannelLabels.append(channel_label)
                                self.emgChannelSensors.append(sensor_label)  # Map EMG channel to sensor label
                                self.emgSampleRates[sensor_label] = sample_rate  # Map sensor label to sample rate
                                self.sensor_label_to_channels[sensor_label]['EMG'].append({'index': globalChannelIdx, 'label': channel_label, 'sample_rate': sample_rate})
                                self.plotCount += 1
                            elif "IMP" in channel_name:
                                # Exclude IMP channels
                                pass
                            elif "ACC" in channel_name:
                                self.accChannelsIdx.append(globalChannelIdx)
                                self.accChannelLabels.append(channel_label)
                                self.accChannelSensors.append(sensor_label)  # Map ACC channel to sensor label
                                if sensor_label not in self.accSampleRates:
                                    self.accSampleRates[sensor_label] = sample_rate
                                self.sensor_label_to_channels[sensor_label]['ACC'].append({'index': globalChannelIdx, 'label': channel_label, 'sample_rate': sample_rate})
                                self.plotCount += 1
                            elif "GYRO" in channel_name:
                                self.gyroChannelsIdx.append(globalChannelIdx)
                                self.gyroChannelLabels.append(channel_label)
                                self.gyroChannelSensors.append(sensor_label)  # Map GYRO channel to sensor label
                                if sensor_label not in self.gyroSampleRates:
                                    self.gyroSampleRates[sensor_label] = sample_rate
                                self.sensor_label_to_channels[sensor_label]['GYRO'].append({'index': globalChannelIdx, 'label': channel_label, 'sample_rate': sample_rate})
                                self.plotCount += 1
                            elif "OR" in channel_name:
                                self.orChannelsIdx.append(globalChannelIdx)
                                self.orChannelsLabels.append(channel_label)
                                self.orChannelSensors.append(sensor_label)
                                if sensor_label not in self.orSampleRates:
                                    self.orSampleRates[sensor_label] = sample_rate
                                self.sensor_label_to_channels[sensor_label]['OR'].append({'index': globalChannelIdx, 'label': channel_label, 'sample_rate': sample_rate})
                                self.plotCount += 1
                            else:
                                # Other channels, ignore or handle as needed
                                pass
                            globalChannelIdx += 1

                    else:
                        print(f"No channels found for sensor {sensor_label}")

                if self.collection_data_handler.EMGplot:
                    self.collection_data_handler.EMGplot.initiateCanvas(None, None, self.plotCount, 1, 20000)

                return True
        else:
            return False

    def Stop_Callback(self):
        """Callback to stop the data stream"""
        self.collection_data_handler.pauseFlag = True
        self.TrigBase.Stop()
        print("Data Collection Complete")

    # ---------------------------------------------------------------------------------
    # ---- Helper Functions
    def getSampleModes(self, sensorIdx):
        """Gets the list of sample modes available for selected sensor"""
        sampleModes = self.TrigBase.AvailibleSensorModes(sensorIdx)
        return sampleModes

    def getCurMode(self, sensorIdx):
        """Gets the current mode of the sensors"""
        curModes = self.TrigBase.GetCurrentSensorMode(sensorIdx)
        return curModes

    def setSampleMode(self, curSensor, setMode):
        """Sets the sample mode for the selected sensor"""
        self.TrigBase.SetSampleMode(curSensor, setMode)