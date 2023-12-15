# Assistive Arm Project at Harvard

## Our setup
The general setup can be a bit cumbersome, but essentially what we need is the 
following: 

* 1 Raspberry Pi 4 Model B 2GB
* 1 Desktop for retrieving Motion Capture data
* 1 Macbook Pro
  
### Networking
Raspberry Pi and Desktop are connected to the same network via Ethernet, and 
communicate between them via a TCP port using ZMQ.

Coding is done from the Macbook Pro via SSH directly on the Raspberry Pi.

### IP addresses
On the Harvard secure network, both desktop and raspberry pi MUST be connected
to the network on the same way (WIFI or ETH), otherwise they won't be able to
communicate between them.

You might need to modify the IP address and make it static so that they are on 
the same network.

### Sit-to-stand
[![Video](https://img.youtube.com/vi/tL9eCLn0_UM/maxresdefault.jpg)](https://youtube.com/shorts/tL9eCLn0_UM?feature=share)
