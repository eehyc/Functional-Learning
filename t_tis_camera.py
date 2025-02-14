import os
import sys
import logging
tis_visible = os.environ.get('TIS_VISIBLE')


if not tis_visible or tis_visible == 0:

    def Active():
        logging.info('Does not activate TIS module.')
        return False


else:

    def Active():
        logging.info('Activate TIS module.')
        return True


    import ctypes as C
    import IC_TIS.tisgrabber as IC
    import cv2
    import numpy as np
    import time, sys
    import os
    import t_io
    import math
    import t_component_interface



    def delta(Camera):

        # # Create the function pointer.
        # Callbackfunc = IC.TIS_GrabberDLL.FRAMEREADYCALLBACK(Callback)
        #
        # Userdata = CallbackUserdata()
        #
        # # # Create the camera object.
        # # Camera = IC.TIS_CAM()
        # #
        # # # Open a camera.
        # # Camera.ShowDeviceSelectionDialog()
        #
        # # Noe pass the function pointer and our user data to the library.
        # Camera.SetFrameReadyCallback(Callbackfunc, Userdata)

        # # Handle each incoming frame automatically.
        # Camera.SetContinuousMode(0)
        #
        # Camera.SetPropertySwitch("Trigger", "Enable", 1)
        # # Start live video.
        # Camera.StartLive(1)

        try:
            while (True):
                Camera.PropertyOnePush("Trigger", "Software Trigger")
                time.sleep(0.5)

        except KeyboardInterrupt:
            Camera.StopLive()


    class Instance():

        def __init__(self, FLAGS, params, reconfigure = False, live = True):


            self.camera = IC.TIS_CAM()


            path = params['dirList']['data']
            dict_file = path + 'camera_property_dict.json'
            setting_file = path + 'camera.setting'


            if reconfigure or not os.path.exists(dict_file) or not os.path.exists(setting_file):
                print('Please configure the Camera.')
                self.configureCamera(FLAGS, params)
                self.camera.SaveDeviceStateToFile(setting_file)
                t_io.WriteJson(dict_file, self.property_dict)
            else:
                logging.info('Loaded camera configure files.')

                self.camera.LoadDeviceStateFromFile(setting_file)

                self.property_dict = t_io.ReadJson(dict_file)
                self.setCameraProperty(self.property_dict)
                self.getCameraProperty(self.property_dict)



            if not self.camera.IsDevValid():
                print('Unable to initialize TIS camera.')
                sys.exit()

            self.dynamicRange = FLAGS.dynamicRange
            self.live = live

            self.startLive()


        def tuneExposure(self, scale = 1):

            self.camera.StopLive()

            temp_dict = self.property_dict
            if not scale == 1:
                temp_dict = self.property_dict.copy()
                original_exp = temp_dict['Exposure']
                temp_dict['Exposure'] = original_exp * scale

            self.setCameraProperty(temp_dict)

            self.startLive()


        def startLive(self):

            res = str(self.camera.GetVideoFormats()[0].split()[1])

            if self.dynamicRange > 8:
                self.camera.SetFormat(IC.SinkFormats.Y16)
                self.camera.SetVideoFormat("Y16 " +  res)
            else:
                self.camera.SetFormat(IC.SinkFormats.RGB24)
                self.camera.SetVideoFormat("RGB32 " + res)

            self.callback_func = IC.TIS_GrabberDLL.FRAMEREADYCALLBACK(Instance.__callbackFunc)
            self.callback_data = Instance.__callbackData()


            # Now pass the function pointer and our user data to the library.
            self.camera.SetFrameReadyCallback(self.callback_func, self.callback_data)


            # Handle each incoming frame automatically.
            self.camera.SetContinuousMode(0)
            self.camera.SetPropertySwitch("Trigger", "Enable", 1)

            if (self.live):
                self.live_window = 1
                self.camera.StartLive(self.live_window)
            else:
                self.live_window = 0
                self.camera.StartLive(self.live_window)
            #self.camera.StopLive()

            exp_time = self.property_dict['Exposure']
            if exp_time > 1:
               self.time_out = exp_time * 5
            else:
                self.time_out = 5

            logging.info('Start camera live.')



        def destroy(self):
            self.camera.StopLive()

        def setCameraProperty(self, dict):

            # Brightness
            self.camera.SetPropertySwitch("Brightness", "Auto", 0)
            self.camera.SetPropertyValue("Brightness", "Value", int(dict['Brightness']))

            # Gamma
            self.camera.SetPropertySwitch("Gamma", "Auto", 0)
            self.camera.SetPropertyValue("Gamma", "Value", int(dict['Gamma']))

            # White Balance
            self.camera.SetPropertySwitch("WhiteBalance", "Auto", 0)
            self.camera.SetPropertyValue("WhiteBalance", "White Balance Red", (dict['WhiteBalanceRed']))
            self.camera.SetPropertyValue("WhiteBalance", "White Balance Green", (dict['WhiteBalanceRed']))
            self.camera.SetPropertyValue("WhiteBalance", "White Balance Blue", (dict['WhiteBalanceRed']))

            # Gain
            self.camera.SetPropertySwitch("Gain", "Auto", 0)
            self.camera.SetPropertyValue("Gain", "Value", (dict['Gain']))

            # Exposure
            self.camera.SetPropertySwitch("Exposure", "Auto", 0)
            self.camera.SetPropertyAbsoluteValue("Exposure", "Value", (dict['Exposure']))


        def getCameraProperty(self, dict):

            # Brightness
            dict['Brightness'] = self.camera.GetPropertyValue("Brightness", "Value")
            logging.info("Brightness : %s" % str(dict['Brightness']))

            # Gamma
            dict['Gamma'] = self.camera.GetPropertyValue("Gamma", "Value")
            logging.info("Gamma : %s" % str(dict['Gamma']))

            # White Balance
            dict['WhiteBalanceRed'] = self.camera.GetPropertyValue("WhiteBalance", "White Balance Red")
            dict['WhiteBalanceGreen'] = self.camera.GetPropertyValue("WhiteBalance", "White Balance Green")
            dict['WhiteBalanceBlue'] = self.camera.GetPropertyValue("WhiteBalance", "White Balance Blue")
            logging.info("White balance : %s %s %s" % (str(dict['WhiteBalanceRed']), str(dict['WhiteBalanceGreen']), str(dict['WhiteBalanceBlue'])))

            # Gain
            dict['Gain'] = self.camera.GetPropertyValue("Gain", "Value")
            logging.info("Gain : %s" % str(dict['Gain']))

            # Exposure
            exposureTime = [0]
            self.camera.GetPropertyAbsoluteValue("Exposure", "Value", exposureTime)
            dict['Exposure'] = (exposureTime[0])
            logging.info("Exposure time abs: %s" % str(dict['Exposure']))



        class __callbackData(C.Structure):

            def __init__(self):
                self.ready = False
                self.img = None

            def release(self):
                self.ready = False
                self.img = None

        def __callbackFunc(hGrabber, pBuffer, framenumber, pData):
            """ This is an example callback function
                 The image is saved in test.jpg and the pData.Value1 is
                 incremented by one.

            :param: hGrabber: This is the real pointer to the grabber object.
            :param: pBuffer : Pointer to the first pixel's first byte
            :param: framenumber : Number of the frame since the stream started
            :param: pData : Pointer to additional user data structure
            """

            # pData.buffer_size = 1600 * 1200 * 3
            # pData.height = 1200
            # pData.width = 1600
            # pData.iBitsPerPixel = 3
            # pData.image = C.cast(pBuffer, C.POINTER(C.c_ubyte * pData.buffer_size))
            #
            #
            # cvMat = np.ndarray(buffer = pData.image.contents,
            #                 dtype = np.uint8,
            #                 shape = (pData.height,
            #                         pData.width,
            #                         pData.iBitsPerPixel))
            #
            # pData.image = cvMat

            pData.ready = True

        def snapImage(self):

            ret = self.camera.PropertyOnePush("Trigger","Software Trigger")



            black = False
            t0 = time.clock()
            while not self.callback_data.ready:
                #time.sleep(1.1 * exp_time)
                if time.clock() - t0 > self.time_out and not self.callback_data.ready:
                    print('TIS_Camera : snapping image timeout.')
                    input('wait for command')
                    black = True

            if self.callback_data.ready == True or black == True:
                #raw_image1 = self.callback_data.image

                # self.camera.SnapImage()
                if self.dynamicRange > 8:
                    raw_image = self.camera.GetImageEx()
                else:
                    raw_image = self.camera.SnapImage()

                if black == True:
                    raw_image.fill(0)


                if self.dynamicRange > 8:
                    assert(raw_image.dtype == np.uint16)

                if raw_image.shape[-1] == 3 and raw_image.dtype == np.uint8:

                    image = np.empty_like(raw_image)
                    image[:,:,0] = raw_image[:,:,2]
                    image[:,:,1] = raw_image[:,:,1]
                    image[:,:,2] = raw_image[:,:,0]

                    image = cv2.flip(image, 0)
                else:
                    # debayer
                    image = cv2.cvtColor(raw_image, cv2.COLOR_BAYER_GB2RGB)

                self.callback_data.release()

                return image
            else:
                exit(0)


        def configureCamera(self, FLAGS, params):

            self.camera.ShowDeviceSelectionDialog()
            #self.camera.SetFrameRate(1.0)
            self.property_dict = {}
            self.getCameraProperty(self.property_dict)


            # self.property_dict['X0'] = 0
            # self.property_dict['X1'] = -1
            # self.property_dict['Y0'] = 0
            # self.property_dict['Y1'] = -1


            # command = input("Pleas input the optional commands (X0=,X1=,Y0=,Y1=) : ")
            # if "=" in command:
            #     command.replace(' ', '')
            #     pairs = command.split(',')
            #     for p in pairs:
            #         pp = p.split("=")
            #         key = pp[0]
            #         value = pp[1]
            #         if key == 'X0':
            #             self.property_dict['X0'] = value
            #         elif key == 'X1':
            #             self.property_dict['X1'] = value
            #         elif key == 'Y0':
            #             self.property_dict['Y0'] = value
            #         elif key == 'Y1':
            #             self.property_dict['Y1'] = value


            # with open(dict_file, 'w', newline='') as csv_file:
            #     writer = csv.writer(csv_file)
            #     for key, value in self.property_dict.items():
            #         writer.writerow([key, value])


            # reset the dict
            # if os.path.exists(file):
            #     with open(file, 'rb') as csv_file:
            #         reader = csv.reader(csv_file)
            #         property_dict = dict(reader)
            #
            #         self.setCameraProperty(property_dict)

            # get initial state

            # try:
            #     while (True):
            #
            #         command = input("Input the command (WhiteBalanceGreen=/Gain=/Exposure=XXX) : ")
            #


            #
            #         # refresh and print statistics
            #         image = self.snapImage(property_dict['Exposure'])
            #         temp = np.reshape(image, [-1, 3])
            #         print('Min : ', temp.min(axis=0))
            #         print('Max : ', temp.max(axis=0))
            #
            #         # Apply some OpenCV function on this image
            #         # image = cv2.flip(image, 0)
            #         #
            #         cv2.imshow('Captured', image)
            #         cv2.waitKey(10)
            #
            # except KeyboardInterrupt:
            #     self.camera.StopLive()
            #     cv2.destroyWindow('Window')

        class TISCameraComponentInterface(t_component_interface.ImageComponentInterface):

            def __init__(self, driver):
                self.driver = driver

            def name(self):
                return 'tis_camera'

            def hwc(self):
                Camera = self.driver.camera
                Imageformat = Camera.GetImageDescription()[:3]

                width = Imageformat[0]
                height = Imageformat[1]
                channels = 3 if (Imageformat[2] == 16) else Imageformat[2] / 8

                return (height, width, channels)

            def sense(self):
                img = self.driver.snapImage()
                img = t_io.ToHDR(img)
                #img = np.power(img, 1.0/2.2)
                return img

            def tuneSensitivity(self, scale):

                self.driver.camera.tuneExposure(scale)


            def emit(self, img):
                assert(0)

        def interface(self):
            return self.TISCameraComponentInterface(self)