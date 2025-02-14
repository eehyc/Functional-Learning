# Functional-Learning

1. The project supports TIS camera by defaults. For other cameras, you need to adapt the driver and import as t_camera.
2. You need to launch a server thread and a client thread. The server thread control the optical device. The client thread send request of sensing to the server thread, and process the received data (training).
3. Launch run.py with the following example command to start server thread: --oDevice regular_2LC --startCheckpointStage MNIST_2LC/ --checkpointStage MNIST_2LC/ --task ['train'] --dWorkers 2 --startStep -0 --seed 4 --activateHardware=True --hardwareSocket server -calibrateLineality=False -data MNIST
4. Launch run.py with the following example command to start client thread: --oDevice regular_2LC --startCheckpointStage MNIST_2LC/ --checkpointStage MNIST_2LC/ --task ['train'] --dWorkers 2 --startStep -0 --seed 4 --activateHardware=False --hardwareSocket [the ip of the server thread] -calibrateLineality=False -data MNIST
5. output/MNIST/Data/ODevice_i_2L_i-regular_2LC contain an example propagation data, i.e., the matrices descriping the rough neuron connections. You need to regenerate the propagation data in new device.
