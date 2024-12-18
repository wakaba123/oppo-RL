import socket
import time

# 服务器地址和端口（确保与C++服务器代码中的地址和端口一致）
# SERVER_IP = "192.168.2.103"  # 
SERVER_IP = "127.0.0.1"  # 本地运行时为localhost
SERVER_PORT = 8080

def send_socket_message(msg):
    try:
        # 创建 socket 对象
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 连接到服务器
        client_socket.connect((SERVER_IP, SERVER_PORT))
        # 发送消息
        message = str(msg)  # 可以是任意内容
        client_socket.sendall(message.encode())
        print("Message sent to server.")
        # 关闭 socket 连接
        client_socket.close()
    except Exception as e:
        print(f"An error occurred: {e}")

def send_socket_message(msg):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_IP, SERVER_PORT))
    message = str(msg)  
    client_socket.sendall(message.encode())
    response = client_socket.recv(1024)
    res = response.decode('utf-8')
    client_socket.close()
    return res

if __name__ == "__main__":
    while True:
        print("Sending first message...")
        send_socket_message(0)  # 第一次发送消息
        time.sleep(10)
        print("Sending second message...")
        # send_socket_message(1)  # 第二次发送消息
        res = send_socket_message(1)
        print(res)
        break
