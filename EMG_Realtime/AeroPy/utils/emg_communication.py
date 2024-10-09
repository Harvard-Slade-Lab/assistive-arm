# client.py
import socket
import time
import argparse

DEFAULT_HOST = '10.250.176.251'
DEFAULT_PORT = 65432

def run_test(host, port, message_size, total_messages):
    message = b'a' * message_size
    sent_messages = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        start_time = time.time()

        for _ in range(total_messages):
            try:
                s.sendall(message)
                sent_messages += 1
            except BrokenPipeError:
                print("Connection closed by the server.")
                break

        end_time = time.time()

    duration = end_time - start_time
    messages_per_second = sent_messages / duration if duration > 0 else 0

    print(f'Sent {sent_messages} messages in {duration:.2f} seconds.')
    print(f'Messages per second: {messages_per_second:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Socket Connection Speed Test Client')
    parser.add_argument('--host', type=str, default=DEFAULT_HOST, help='Server IP address')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='Server port')
    parser.add_argument('--size', type=int, default=100, help='Size of each message in bytes')
    parser.add_argument('--messages', type=int, default=10000, help='Total number of messages to send')

    args = parser.parse_args()

    run_test(args.host, args.port, args.size, args.messages)