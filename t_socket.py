# Based on https://github.com/ekbanasolutions/numpy-using-socket/blob/master/npsocket/npsocket.py

import socket
import logging
import pickle
import struct
import logging
from threading import Thread
import socketserver

def GetHostIP():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

class Socket():
    def __init__(self):

        self.address = ''
        self.port = 0
        self.type = None  # server or client
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


    def connect(self, address, port):
        """
        :param address: host address of the socket e.g 'localhost' or your ip
        :type address: str
        :param port: port in which the socket should be intialized. e.g 4000
        :type port: int
        :return: None
        :rtype: None
        """

        self.type = 'client'

        self.address = address
        self.port = port
        self.socket.connect((self.address, self.port))

        self.interface = self.socket
        self.addr = None

        self.payload_size = struct.calcsize("Q")  ### CHANGED
        self.data = b''

    def send(self, np_array):
        """
        :param np_array: Numpy array to send to the listening socket
        :type np_array: ndarray
        :return: None
        :rtype: None
        """
        data = pickle.dumps(np_array)

        # Send message length first
        message_size = struct.pack("Q", len(data))  ### CHANGED

        # Then data
        self.interface.sendall(message_size + data)

    def runServer(self, port):
        """
        :param port: port to listen
        :type port: int
        :return: numpy array
        :rtype: ndarray
        """

        self.type = 'server'

        self.address = ''
        self.port = port
        self.socket.bind((self.address, self.port))
        logging.info('Socket binding complete')

        self.payload_size = struct.calcsize("Q")  ### CHANGED
        self.data = b''

        self.socket.listen(10)
        logging.info('Server is listening port ' + str(self.port) + ', with max connection 5.')
        self.resetConnection()



    def receive(self):

        while len(self.data) < self.payload_size:
            self.data += self.interface.recv(4096)

        packed_msg_size = self.data[:self.payload_size]
        self.data = self.data[self.payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Retrieve all data based on message size
        while len(self.data) < msg_size:
            self.data += self.interface.recv(4096)

        frame_data = self.data[:msg_size]
        self.data = self.data[msg_size:]

        # Extract frame
        frame = pickle.loads(frame_data)
        return frame

    def resetConnection(self):

        if self.type == 'server':
            self.interface, self.addr = self.socket.accept()
            logging.info('Connected by ' + str(self.addr))


    def __del__(self):
        if self.socket:
            try:
                self.socket.shutdown(2)
                self.socket.close()
            except Exception as e:
                logging.critical(str(e))



def PackAndSend(np_array, interface):

    data = pickle.dumps(np_array)
    message_size = struct.pack("Q", len(data))  ### CHANGED

    interface.sendall(message_size + data)

def ReceiveAndUnpack(interface):

    payload_size = struct.calcsize("Q")  ### CHANGED
    data = b''

    while len(data) < payload_size:
        data += interface.recv(4096)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    # Retrieve all data based on message size
    while len(data) < msg_size:
        data += interface.recv(4096)

    array_data = data[:msg_size]

    # Extract frame
    array = pickle.loads(array_data)

    return array

class ThreadingSocket():

    def __init__(self, type, address='', port=0):

        self.address = address
        self.port = port
        self.type = type  # server or client

        self.suggested_batch_size = 1024
        self.suggested_batch_interval = 60

    class ThreadingServerWithCallBack(socketserver.BaseRequestHandler):

        def callback(self, data):

            return data

        def handle(self):

            logging.info('Connected by ' + str(self.client_address))

            data = ReceiveAndUnpack(self.request)
            send_data = self.callback(data)

            logging.info('Reply to ' + str(self.client_address))
            PackAndSend(send_data, self.request)


    class MyThreadingConnection(Thread):

        def __init__(self, address, port, data):

            super().__init__()

            self.socket = None

            self.send_data = data
            self.address = address
            self.port = port

            return

        def __del__(self):

            if not self.socket is None:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()

        def run(self):

            try:
                self.socket = socket.create_connection((self.address, self.port))
                PackAndSend(self.send_data, self.socket)
                self.recv_data = ReceiveAndUnpack(self.socket)
            except BaseException as e:
                logging.error(e)
                logging.error('Catch exception and close socket ' + str(self) + '.')
            finally:
                if not self.socket is None:
                    self.socket.shutdown(socket.SHUT_RDWR)
                    self.socket.close()
                    self.socket = None

        def shutdown(self):

            if not self.socket is None:
                self.socket.shutdown(socket.SHUT_RDWR)


        def result(self):
            try:
                if self.is_alive():
                    self.join()
                return self.recv_data
            except BaseException as e:
                logging.error(e)
                return None


    def connect(self, data_dict):

        assert self.type == 'client', 'Socket type has to be \'client\'.'

        connection = self.MyThreadingConnection(self.address, self.port, data_dict)
        connection.start()

        return connection



    # def run(self, ThreadingServerWithCallBack):
    def run(self):

        assert self.type == 'server', 'Socket type has to be \'server\'.'

        self_ip = GetHostIP()

        self.server = socketserver.ThreadingTCPServer((self.address, self.port), self.ThreadingServerWithCallBack)
        logging.critical('Server starts ------ ' + str(self_ip) + ':' + str(self.port))
        self.server.serve_forever()


    def __del__(self):
        if hasattr(self, 'server'):
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception as e:
                logging.error(e)
