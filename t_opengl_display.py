import os
import logging

gl_visible = os.environ.get('OPENGL_VISIBLE')
firstRun = True

if not gl_visible or gl_visible == 0:
    pass

    def Print():
        global firstRun
        if firstRun:
            logging.info('Does not activate OpenGL.')
            firstRun = False

else:

    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    import glfw
    import t_io
    import numpy as np
    import cv2
    import csv

    import time, sys

    import t_component_interface

    class Instance:

        def __init__(self, display_list):

            if not glfw.init():
                print('Unable to initialize glfw.')
                sys.exit()

            self.monitors = glfw.get_monitors()

            if not len(self.monitors) == len(display_list):
                print('Monitor and display list are mismatched.')
                print(self.monitors)
                print(display_list)
                sys.exit()

            windows = {}
            width_offset = 0
            for i in range(len(display_list)):
                mode = glfw.get_video_mode(self.monitors[i])
                width, height = mode.size

                if display_list[i] == None:
                    width_offset += width
                    continue


                glfw.window_hint(glfw.DECORATED, False)
                # glfw.window_hint(glfw.GREEN_BITS, 8)
                # glfw.window_hint(glfw.RED_BITS, 8)
                # glfw.window_hint(glfw.BLUE_BITS, 8)
                # glfw.window_hint(glfw.REFRESH_RATE, mode.refresh_rate)
                window = glfw.create_window(width, height, display_list[i], None, None)
                glfw.set_window_pos(window, width_offset, 0)
                width_offset += width

                glfw.make_context_current(window)

                textID = glGenTextures(1)

                glBindTexture(GL_TEXTURE_2D, textID)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

                gamma = 1.0
                window.gamma = [gamma, gamma, gamma]
                window.lut = self.__makeGammaLUT(window)
                windows[display_list[i]] = window

            self.windows = windows
            self.spectrum_correlation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            self.LC_delay = 0.1


        def writeProperty(self, params, FLAGS):

            path = params['dirList']['data']
            dict_file = path + 'displays_property_dict.json'

            property_dict = {}
            property_dict['spectrum_correlation'] = self.spectrum_correlation
            for key, window in self.windows.items():

                w_dict = {}
                #gamma
                w_dict['gamma'] = window.gamma

                #lut
                w_dict['lut'] = window.lut
                property_dict[key] = w_dict

            t_io.WriteJson(dict_file, property_dict)

        def readProperty(self, params, FLAGS):

            path = params['dirList']['data']
            dict_file = path + 'displays_property_dict.json'

            if not os.path.exists(dict_file):
                print('Can not read the specified file : ', dict_file)
                exit(0)
            else:

                property_dict = t_io.ReadJson(dict_file)
                if 'spectrum_correlation' in property_dict:
                    self.spectrum_correlation = property_dict['spectrum_correlation']


                for device, w_dict in property_dict.items():
                    if device in self.windows:
                        window = self.windows[device]
                        #gamma
                        window.gamma = w_dict['gamma']
                        #lut
                        lut = w_dict['lut']
                        if not lut is None:
                            lut = np.array(w_dict['lut']).astype(np.uint8)
                        window.lut = lut


        # Pop the window to the top of the desktop.
        def focus(self):
            for item in self.windows.items():
                window = item[1]
                glfw.focus_window(window)

        def displaySize(self, display):

            window = self.windows[display]
            glfw.make_context_current(window)
            window_width, window_height = glfw.get_framebuffer_size(window)

            return (window_width, window_height)

        # Update window framebuffer.
        def refresh(self, display_or_window):

            if isinstance(display_or_window, str):
                window = self.windows[display_or_window]
            else:
                window = display_or_window

            glfw.make_context_current(window)

            glDrawBuffer(GL_FRONT)

            window_width, window_height = glfw.get_framebuffer_size(window)

            glClear(GL_COLOR_BUFFER_BIT)

            glDisable(GL_DEPTH_TEST)
            glDisable(GL_LIGHTING)
            glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            glDisable(GL_FRAMEBUFFER_SRGB)
            glEnable(GL_TEXTURE_2D)

            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            glViewport(0, 0, window_width, window_height)

            # glBegin(GL_QUADS)
            # glTexCoord2f(0.0, 0.0)
            # glVertex3f(-1.0, -1.0, 0.5)
            # glTexCoord2f(1.0, 0.0)
            # glVertex3f(1.0, -1.0, 0.5)
            # glTexCoord2f(1.0, 1.0)
            # glVertex3f(1.0, 1.0, 0.5)
            # glTexCoord2f(0.0, 1.0)
            # glVertex3f(-1.0, 1.0, 0.5)
            # glEnd()

            # alignment issue
            glBegin(GL_QUADS)
            glTexCoord2f(0.0, 0.0)
            glVertex3f(-1.0, 1.0, 0.5)
            glTexCoord2f(1.0, 0.0)
            glVertex3f(-1.0, -1.0, 0.5)
            glTexCoord2f(1.0, 1.0)
            glVertex3f(1.0, -1.0, 0.5)
            glTexCoord2f(0.0, 1.0)
            glVertex3f(1.0, 1.0, 0.5)
            glEnd()

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()

            glDisable(GL_TEXTURE_2D)

            #glFlush()
            glFinish()

            # glfw.swap_buffers(window)
            glfw.poll_events()

            #glfw.focus_window(window)

        def refreshAll(self):

            for item in self.windows.items():
                window = item[1]
                self.refresh(window)
                #glfw.focus_window(window)

            self.delay()

        def delay(self):
            if self.LC_delay > 0:
                time.sleep(self.LC_delay)

        def __makeGammaLUT(self, window):

            gamma = window.gamma
            table = np.empty((1, 256, 3), dtype=np.uint8)

            for c in range(3):
                invGamma = 1.0 / gamma[c]
                table[:, :,c] = np.array([((i / 255.0) ** invGamma) * 255
                                  for i in np.arange(0, 256)]).astype("uint8")

            return table




        def switchTexture(self, display, image):

            window = self.windows[display]

            if not image.dtype == np.uint8:
                image = t_io.ToLDR(image)

            if not window.lut is None:
                image = cv2.LUT(image, window.lut)

            glfw.make_context_current(window)
            # alignment
            image = image.transpose(1, 0, 2)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, image.shape[1], image.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, image)

        def offsetGamma(self, display, gamma):

            window = self.windows[display]

            for l in range(len(window.gamma)):
                window.gamma[l] += gamma[l]

            window.lut = self.__makeGammaLUT(window)

        def calibrateLinearLUT(self, display, x_range, y_rgb, gt_y_rgb, channel_s):

            l = x_range

            window = self.windows[display]
            #window.lut = np.zeros((1, 256, 3),  dtype=np.uint8)

            for c in range(3):
                if channel_s[c] == 0:
                    continue
                y_array = y_rgb[c]
                gt_y_array = gt_y_rgb[c]

                assert (x_range == len(y_array))
                assert (x_range == len(gt_y_array))

                new_lut = np.zeros((1, 256), dtype=np.uint8)

                for i in range(0, l):

                    abs_difference = np.abs(gt_y_array[i] - y_array)

                    min_idx = np.where(abs_difference==np.min(abs_difference))
                    min_idx = min_idx[0][0]
                    new_lut[0, i] = min_idx


                window.lut[:,:,c] = np.sort(new_lut)
                window.lut_range = x_range

        def destroy(self):

            for item in self.windows.items():
                window = item[1]
                glfw.destroy_window(window)

            glfw.terminate()
            self.windows = None
            self.monitors = None

        class OpenGLComponentInterface(t_component_interface.ImageComponentInterface):

            def __init__(self, display, driver):
                self.display = display
                self.driver = driver

            def name(self):
                return self.display

            def hwc(self):
                size = self.driver.displaySize(self.display)
                return (size[1], size[0], 3)


            def emit(self, img):
                self.driver.switchTexture(self.display, img)
                self.driver.refresh(self.display)

            def delay(self):
                self.driver.delay()

        def interface(self, display):
            return self.OpenGLComponentInterface(display, self)




