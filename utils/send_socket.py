import socket
import time

def send_socket_data(message, host='192.168.2.103', port=8888):
    try:
        # 创建一个 socket 对象
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 连接到服务器
        client_socket.connect((host, port))
        
        # 发送数据
        client_socket.sendall(message.encode('utf-8'))
        
        # 接收来自服务器的响应
        response = client_socket.recv(1024)
        return response.decode('utf-8')
        
    except socket.error as e:
        print(f"Socket error: {e}")
    finally:
        # 关闭连接
        client_socket.close()

# print(send_socket_data('1,825600,672000'))
t = 0
print('big_cpu_freq,little_cpu_freq,fps,mem,little_util,big_util,ipc,cache_miss')
wangzhe = 'SurfaceView[com.tencent.tmgp.sgame/com.tencent.tmgp.sgame.SGameActivity]\(BLAST\)#6327'
douyin = 'SurfaceView[com.ss.android.ugc.aweme/com.ss.android.ugc.aweme.splash.SplashActivity]\(BLAST\)#6485'
while t < 200:
    print(send_socket_data('0'))
    t+=1
    time.sleep(0.1)
    
